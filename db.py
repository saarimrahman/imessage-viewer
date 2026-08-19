"""Read-only access to chat.db, or a copied snapshot if the live file is locked."""

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from config import APPLE_EPOCH, DB_PATH, MEDIA_PAGE_SIZE, PAGE_SIZE, SNAPSHOT_DB
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


# Group tapbacks store the participant index in the guid prefix (`p:0/…`,
# `p:3/…`). 32 covers an iMessage group at the usual size limit.
REACTION_GUID_PREFIXES = 32


def reaction_lookup_keys(guids):
    """Bare guid plus `p:N/guid` so the associated_message_guid index can hit."""
    keys = []
    seen = set()
    for guid in guids:
        if not guid or guid in seen:
            continue
        seen.add(guid)
        keys.append(guid)
        keys.extend(f"p:{i}/{guid}" for i in range(REACTION_GUID_PREFIXES))
    return keys


def load_reactions(conn, chat_id, guids):
    """Tapbacks are stored as their own message rows pointing at a target guid
    via associated_message_guid. A later 3xxx-type row for the same
    (target, reactor) pair means the reaction was removed, so state must be
    replayed in date order rather than just collected."""
    if not guids:
        return {}
    guid_set = set(guids)
    keys = reaction_lookup_keys(guid_set)
    if not keys:
        return {}
    chat_sql, chat_params = chat_filter(conn, chat_id)
    qmarks = ",".join("?" * len(keys))
    rows = conn.execute(
        f"""SELECT m.associated_message_guid, m.associated_message_type, m.associated_message_emoji,
                  m.is_from_me, m.date, h.id as handle
           FROM message m
           LEFT JOIN handle h ON h.ROWID = m.handle_id
           WHERE m.associated_message_type BETWEEN 2000 AND 3999
             AND m.associated_message_guid IN ({qmarks})
             AND EXISTS (
               SELECT 1 FROM chat_message_join cmj
               WHERE cmj.message_id = m.ROWID AND {chat_sql}
             )
           ORDER BY m.date ASC""",
        keys + chat_params,
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


# Start from chat_message_join so ORDER BY message_date can use
# chat_message_join_idx_message_date_id_chat_id instead of sorting the thread.
MSG_SELECT = """SELECT m.ROWID as id, m.guid, m.text, m.attributedBody, m.date, m.is_from_me, h.id as handle
            FROM chat_message_join cmj
            JOIN message m ON m.ROWID = cmj.message_id
            LEFT JOIN handle h ON h.ROWID = m.handle_id"""


def _cmj_cursor(date_op, id_op=None):
    id_op = id_op or date_op
    return (
        f"(cmj.message_date {date_op} ? OR "
        f"(cmj.message_date = ? AND cmj.message_id {id_op} ?))"
    )


def fetch_messages(conn, chat_id, after=None, before=None, start_ns=None, limit=PAGE_SIZE, from_end=False):
    """Page by (date, ROWID). `after`/`before` are (date_ns, rowid) cursors."""
    chat_sql, chat_params = chat_filter(conn, chat_id)
    where = [chat_sql, REACTION_EXCLUDE_SQL]
    params = list(chat_params)
    order = "cmj.message_date ASC, cmj.message_id ASC"
    if after:
        date_ns, rowid = after
        where.append(_cmj_cursor(">"))
        params.extend([date_ns, date_ns, rowid])
    elif before:
        date_ns, rowid = before
        where.append(_cmj_cursor("<"))
        params.extend([date_ns, date_ns, rowid])
        order = "cmj.message_date DESC, cmj.message_id DESC"
    elif start_ns is not None:
        where.append("cmj.message_date >= ?")
        params.append(start_ns)
    elif from_end:
        order = "cmj.message_date DESC, cmj.message_id DESC"
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
        return None, None, None
    return apple_date(r["date"])[:10], (r["is_from_me"], r["handle"]), r["date"]


def has_neighbor(conn, chat_id, date_ns, rowid, direction):
    clause = _cmj_cursor("<" if direction == "before" else ">")
    chat_sql, chat_params = chat_filter(conn, chat_id)
    return conn.execute(
        f"""SELECT 1 FROM chat_message_join cmj
            JOIN message m ON m.ROWID = cmj.message_id
            WHERE {chat_sql} AND {REACTION_EXCLUDE_SQL} AND {clause} LIMIT 1""",
        chat_params + [date_ns, date_ns, rowid],
    ).fetchone() is not None


def chat_date_bounds(conn, chat_id):
    """First and last message_date in the merged thread. Uses the join table
    so a long chat does not have to touch every `message` row."""
    chat_sql, chat_params = chat_filter(conn, chat_id)
    return conn.execute(
        f"SELECT min(cmj.message_date) as lo, max(cmj.message_date) as hi "
        f"FROM chat_message_join cmj WHERE {chat_sql}",
        chat_params,
    ).fetchone()


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
           WHERE {chat_sql} AND {_cmj_cursor("<", "<=")} AND {REACTION_EXCLUDE_SQL}
           ORDER BY cmj.message_date DESC, cmj.message_id DESC LIMIT ?""",
        chat_params + [tgt_date, tgt_date, target_id, half],
    ).fetchall()
    after_rows = conn.execute(
        f"""{MSG_SELECT}
           WHERE {chat_sql} AND {_cmj_cursor(">")} AND {REACTION_EXCLUDE_SQL}
           ORDER BY cmj.message_date ASC, cmj.message_id ASC LIMIT ?""",
        chat_params + [tgt_date, tgt_date, target_id, half],
    ).fetchall()
    return list(reversed(before_rows)) + list(after_rows)


MEDIA_FILE_SQL = "(att.filename IS NULL OR att.filename NOT LIKE '%.pluginPayloadAttachment')"
MEDIA_VISUAL_SQL = """(
  ifnull(att.mime_type, '') LIKE 'image/%'
  OR ifnull(att.mime_type, '') LIKE 'video/%'
  OR lower(ifnull(att.filename, '')) LIKE '%.heic'
  OR lower(ifnull(att.filename, '')) LIKE '%.heif'
  OR lower(ifnull(att.filename, '')) LIKE '%.mov'
  OR lower(ifnull(att.filename, '')) LIKE '%.mp4'
)"""
MEDIA_FROM = """FROM message_attachment_join maj
            JOIN attachment att ON att.ROWID = maj.attachment_id
            JOIN message m ON m.ROWID = maj.message_id
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID"""
MEDIA_SELECT = (
    "SELECT att.ROWID as att_id, att.mime_type, att.filename, "
    "m.ROWID as msg_id, m.date, cmj.chat_id as chat_id "
    + MEDIA_FROM
)


def _media_filters(conn, chat_id=None, visual_only=False):
    where = [MEDIA_FILE_SQL]
    params = []
    if visual_only:
        where.append(MEDIA_VISUAL_SQL)
    if chat_id is not None:
        chat_sql, chat_params = chat_filter(conn, chat_id)
        where.append(chat_sql)
        params.extend(chat_params)
    return where, params


def _media_cursor(op):
    return f"(m.date {op} ? OR (m.date = ? AND att.ROWID {op} ?))"


def fetch_media(
    conn,
    chat_id=None,
    after=None,
    before=None,
    start_ns=None,
    limit=MEDIA_PAGE_SIZE,
    from_end=False,
    visual_only=False,
):
    """Page by (date, attachment id). `after`/`before` are (date_ns, att_id)."""
    where, params = _media_filters(conn, chat_id, visual_only)
    order = "m.date ASC, att.ROWID ASC"
    if after:
        date_ns, att_id = after
        where.append(_media_cursor(">"))
        params.extend([date_ns, date_ns, att_id])
    elif before:
        date_ns, att_id = before
        where.append(_media_cursor("<"))
        params.extend([date_ns, date_ns, att_id])
        order = "m.date DESC, att.ROWID DESC"
    elif start_ns is not None:
        where.append("m.date >= ?")
        params.append(start_ns)
    elif from_end:
        order = "m.date DESC, att.ROWID DESC"
    rows = conn.execute(
        f"{MEDIA_SELECT} WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?",
        params + [limit],
    ).fetchall()
    if before or from_end:
        rows = list(reversed(rows))
    return rows


def has_media_neighbor(conn, date_ns, att_id, direction, chat_id=None, visual_only=False):
    clause = _media_cursor("<" if direction == "before" else ">")
    where, params = _media_filters(conn, chat_id, visual_only)
    where.append(clause)
    params.extend([date_ns, date_ns, att_id])
    return conn.execute(
        f"SELECT 1 {MEDIA_FROM} WHERE {' AND '.join(where)} LIMIT 1",
        params,
    ).fetchone() is not None


def media_date_bounds(conn, chat_id=None, visual_only=False):
    where, params = _media_filters(conn, chat_id, visual_only)
    return conn.execute(
        f"SELECT min(m.date) as lo, max(m.date) as hi {MEDIA_FROM} WHERE {' AND '.join(where)}",
        params,
    ).fetchone()


def media_count(conn, chat_id=None, visual_only=False):
    where, params = _media_filters(conn, chat_id, visual_only)
    return conn.execute(
        f"SELECT count(DISTINCT att.ROWID) {MEDIA_FROM} WHERE {' AND '.join(where)}",
        params,
    ).fetchone()[0]
