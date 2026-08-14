"""Read-only access to chat.db, or a copied snapshot if the live file is locked."""

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from config import APPLE_EPOCH, DB_PATH, PAGE_SIZE, SNAPSHOT_DB
from contacts import person_key, resolve_contact

_snapshot_ready = False


class DbUnavailable(Exception):
    """chat.db exists but this process cannot open it (TCC or a WAL lock)."""


def _open_sqlite(path):
    uri = Path(path).as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    return conn


def _copy_chat_snapshot():
    """Messages.app keeps chat.db in WAL mode. Copy it into .cache so we can
    open a stable snapshot without writing into ~/Library/Messages."""
    os.makedirs(os.path.dirname(SNAPSHOT_DB), exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        src = DB_PATH + suffix
        if os.path.exists(src):
            shutil.copy2(src, SNAPSHOT_DB + suffix)


def get_conn():
    global _snapshot_ready
    try:
        return _open_sqlite(DB_PATH)
    except sqlite3.Error:
        pass
    try:
        if not _snapshot_ready:
            _copy_chat_snapshot()
            _snapshot_ready = True
        return _open_sqlite(SNAPSHOT_DB)
    except (sqlite3.Error, OSError) as e:
        raise DbUnavailable(
            f"Cannot open {DB_PATH}. Grant Full Disk Access to the app running "
            f"this terminal (Cursor or Terminal), then restart. ({e})"
        ) from e


def live_db_error():
    """Reason that this process cannot read chat.db, or None if it can. macOS
    denies the read with EPERM until the app that runs the server has Full Disk
    Access, so a plain open() separates that block from a SQLite-level fault."""
    try:
        with open(DB_PATH, "rb") as f:
            f.read(16)
    except OSError as e:
        return str(e)
    return None


def apple_date(ns):
    if ns is None:
        return ""
    return datetime.fromtimestamp(ns / 1_000_000_000 + APPLE_EPOCH).strftime("%Y-%m-%d %H:%M")


def date_to_apple_ns(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int((dt.timestamp() - APPLE_EPOCH) * 1_000_000_000)


def parse_attributed_body(data):
    """Extract plain text from the NSKeyedArchiver blob Messages stores in
    attributedBody for messages that have no plain-text `text` column."""
    if not data:
        return None
    try:
        start = data.index(b"NSString") + len(b"NSString")
        i = start + 5
        length_byte = data[i]
        if length_byte == 0x81:
            length = int.from_bytes(data[i + 1 : i + 3], "little")
            i += 3
        else:
            length = length_byte
            i += 1
        return data[i : i + length].decode("utf-8", errors="replace")
    except Exception:
        return None


def message_text(row):
    raw = row["text"] or parse_attributed_body(row["attributedBody"])
    if not raw:
        return None
    # U+FFFC is the attachment placeholder iMessage stores in `text`. Alone it
    # renders as an empty bubble; strip it (and leftover whitespace) so image
    # messages only show the attachment.
    text = raw.replace("\ufffc", "").strip()
    return text or None


REACTION_EXCLUDE_SQL = "(m.associated_message_type IS NULL OR m.associated_message_type NOT BETWEEN 2000 AND 3999)"

REACTION_LABELS = {
    2000: ("❤️", "Loved"),
    2001: ("\U0001f44d", "Liked"),
    2002: ("\U0001f44e", "Disliked"),
    2003: ("\U0001f602", "Laughed at"),
    2004: ("‼️", "Emphasized"),
    2005: ("❓", "Questioned"),
    2007: ("\U0001f3f7️", "Reacted with a sticker to"),
}


def reaction_parts(assoc_type, emoji):
    """The (emoji, verb) pair of a tapback. The badge shows the emoji alone."""
    if assoc_type == 2006 and emoji:
        return (emoji, "Reacted")
    return REACTION_LABELS.get(assoc_type, ("\U0001f44d", "Reacted"))


def strip_guid_prefix(guid):
    return guid.rsplit("/", 1)[-1] if guid else guid


CHAT_GROUPS = None


def build_chat_groups(conn):
    """Group the one-to-one chats that hold one person's history. Messages.app
    opens a second chat row when somebody moves between SMS and iMessage, or
    texts from a second number that sits on the same contact card. Only chats
    with exactly one participant qualify, so no group chat ever merges."""
    handles = {}
    for r in conn.execute(
        """SELECT chj.chat_id, h.id as handle
           FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id"""
    ):
        handles.setdefault(r["chat_id"], set()).add(r["handle"])
    by_person = {}
    for chat_id, chat_handles in handles.items():
        if len(chat_handles) != 1:
            continue
        key = person_key(next(iter(chat_handles)))
        if key:
            by_person.setdefault(key, []).append(chat_id)
    groups = {}
    for chat_ids in by_person.values():
        if len(chat_ids) > 1:
            merged = tuple(sorted(chat_ids))
            for chat_id in merged:
                groups[chat_id] = merged
    return groups


def merged_chat_ids(conn, chat_id):
    global CHAT_GROUPS
    if CHAT_GROUPS is None:
        CHAT_GROUPS = build_chat_groups(conn)
    return CHAT_GROUPS.get(chat_id, (chat_id,))


def chat_filter(conn, chat_id, alias="cmj"):
    """SQL and parameters that select every message of the merged chat."""
    chat_ids = merged_chat_ids(conn, chat_id)
    return f"{alias}.chat_id IN ({','.join('?' * len(chat_ids))})", list(chat_ids)


def load_reactions(conn, chat_id, guids):
    """Tapbacks are stored as their own message rows pointing at a target guid
    via associated_message_guid. A later 3xxx-type row for the same
    (target, reactor) pair means the reaction was removed, so state must be
    replayed in date order rather than just collected."""
    if not guids:
        return {}
    guid_set = set(guids)
    chat_sql, chat_params = chat_filter(conn, chat_id)
    rows = conn.execute(
        f"""SELECT m.associated_message_guid, m.associated_message_type, m.associated_message_emoji,
                  m.is_from_me, m.date, h.id as handle
           FROM message m
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           LEFT JOIN handle h ON h.ROWID = m.handle_id
           WHERE {chat_sql} AND m.associated_message_type BETWEEN 2000 AND 3999
           ORDER BY m.date ASC""",
        chat_params,
    ).fetchall()

    state = {}
    for r in rows:
        target = strip_guid_prefix(r["associated_message_guid"])
        if target not in guid_set:
            continue
        reactor_key = "me" if r["is_from_me"] else (r["handle"] or "unknown")
        t = r["associated_message_type"]
        if t >= 3000:
            state.pop((target, reactor_key), None)
        else:
            who = "You" if r["is_from_me"] else (resolve_contact(r["handle"]) or r["handle"] or "Unknown")
            emoji, verb = reaction_parts(t, r["associated_message_emoji"])
            state[(target, reactor_key)] = (emoji, verb, who)

    out = {}
    for (target, _reactor_key), reaction in state.items():
        out.setdefault(target, []).append(reaction)
    return out


MSG_SELECT = """SELECT m.ROWID as id, m.guid, m.text, m.attributedBody, m.date, m.is_from_me, h.id as handle
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            LEFT JOIN handle h ON h.ROWID = m.handle_id"""


def fetch_messages(conn, chat_id, after=None, before=None, start_ns=None, limit=PAGE_SIZE, from_end=False):
    """Page by (date, ROWID). `after`/`before` are (date_ns, rowid) cursors."""
    chat_sql, chat_params = chat_filter(conn, chat_id)
    where = [chat_sql, REACTION_EXCLUDE_SQL]
    params = list(chat_params)
    order = "m.date ASC, m.ROWID ASC"
    if after:
        date_ns, rowid = after
        where.append("(m.date > ? OR (m.date = ? AND m.ROWID > ?))")
        params.extend([date_ns, date_ns, rowid])
    elif before:
        date_ns, rowid = before
        where.append("(m.date < ? OR (m.date = ? AND m.ROWID < ?))")
        params.extend([date_ns, date_ns, rowid])
        order = "m.date DESC, m.ROWID DESC"
    elif start_ns is not None:
        where.append("m.date >= ?")
        params.append(start_ns)
    elif from_end:
        order = "m.date DESC, m.ROWID DESC"
    rows = conn.execute(
        f"{MSG_SELECT} WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?",
        params + [limit],
    ).fetchall()
    if before or from_end:
        rows = list(reversed(rows))
    return rows


def sender_context(conn, rowid):
    r = conn.execute(
        """SELECT m.date, m.is_from_me, h.id as handle FROM message m
           LEFT JOIN handle h ON h.ROWID = m.handle_id WHERE m.ROWID=?""",
        (rowid,),
    ).fetchone()
    if not r:
        return None, None
    return apple_date(r["date"])[:10], (r["is_from_me"], r["handle"])


def has_neighbor(conn, chat_id, date_ns, rowid, direction):
    if direction == "before":
        clause = "(m.date < ? OR (m.date = ? AND m.ROWID < ?))"
    else:
        clause = "(m.date > ? OR (m.date = ? AND m.ROWID > ?))"
    chat_sql, chat_params = chat_filter(conn, chat_id)
    return conn.execute(
        f"""SELECT 1 FROM message m JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE {chat_sql} AND {REACTION_EXCLUDE_SQL} AND {clause} LIMIT 1""",
        chat_params + [date_ns, date_ns, rowid],
    ).fetchone() is not None


def load_attachments(conn, message_ids):
    if not message_ids:
        return {}
    qmarks = ",".join("?" * len(message_ids))
    out = {}
    for a in conn.execute(
        f"""SELECT maj.message_id, att.ROWID as att_id, att.mime_type, att.filename
            FROM message_attachment_join maj JOIN attachment att ON att.ROWID = maj.attachment_id
            WHERE maj.message_id IN ({qmarks})
              AND (att.filename IS NULL OR att.filename NOT LIKE '%.pluginPayloadAttachment')""",
        message_ids,
    ):
        out.setdefault(a["message_id"], []).append(a)
    return out


def fetch_messages_around(conn, chat_id, target_id, half=75):
    target = conn.execute("SELECT date FROM message WHERE ROWID=?", (target_id,)).fetchone()
    if not target:
        return []
    tgt_date = target["date"]
    chat_sql, chat_params = chat_filter(conn, chat_id)
    before_rows = conn.execute(
        f"""{MSG_SELECT}
           WHERE {chat_sql} AND (m.date < ? OR (m.date = ? AND m.ROWID <= ?)) AND {REACTION_EXCLUDE_SQL}
           ORDER BY m.date DESC, m.ROWID DESC LIMIT ?""",
        chat_params + [tgt_date, tgt_date, target_id, half],
    ).fetchall()
    after_rows = conn.execute(
        f"""{MSG_SELECT}
           WHERE {chat_sql} AND (m.date > ? OR (m.date = ? AND m.ROWID > ?)) AND {REACTION_EXCLUDE_SQL}
           ORDER BY m.date ASC, m.ROWID ASC LIMIT ?""",
        chat_params + [tgt_date, tgt_date, target_id, half],
    ).fetchall()
    return list(reversed(before_rows)) + list(after_rows)
