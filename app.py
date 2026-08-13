#!/usr/bin/env python3
"""Local read-only viewer for macOS Messages (chat.db). No dependencies beyond
the standard library. Run: python3 app.py, then open http://127.0.0.1:8765/"""

import glob
import html
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import hashlib
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
CONTACTS_GLOBS = [
    os.path.expanduser("~/Library/Application Support/AddressBook/AddressBook-v22.abcddb"),
    os.path.expanduser("~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb"),
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
PORT = 8765
PAGE_SIZE = 150
THUMB_SIZE = 512
APPLE_EPOCH = 978307200  # 2001-01-01 relative to Unix epoch
CACHE_CONTROL = "public, max-age=31536000, immutable"

os.makedirs(CACHE_DIR, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


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
    return row["text"] or parse_attributed_body(row["attributedBody"])


REACTION_EXCLUDE_SQL = "(m.associated_message_type IS NULL OR m.associated_message_type NOT BETWEEN 2000 AND 3999)"

REACTION_LABELS = {
    2000: "❤️ Loved",
    2001: "\U0001f44d Liked",
    2002: "\U0001f44e Disliked",
    2003: "\U0001f602 Laughed at",
    2004: "‼️ Emphasized",
    2005: "❓ Questioned",
    2007: "\U0001f3f7️ Reacted with a sticker to",
}


def reaction_label(assoc_type, emoji):
    if assoc_type == 2006 and emoji:
        return f"{emoji} Reacted"
    return REACTION_LABELS.get(assoc_type, "Reacted")


def strip_guid_prefix(guid):
    return guid.rsplit("/", 1)[-1] if guid else guid


def load_reactions(conn, chat_id, guids):
    """Tapbacks are stored as their own message rows pointing at a target guid
    via associated_message_guid. A later 3xxx-type row for the same
    (target, reactor) pair means the reaction was removed, so state must be
    replayed in date order rather than just collected."""
    if not guids:
        return {}
    guid_set = set(guids)
    rows = conn.execute(
        """SELECT m.associated_message_guid, m.associated_message_type, m.associated_message_emoji,
                  m.is_from_me, m.date, h.id as handle
           FROM message m
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           LEFT JOIN handle h ON h.ROWID = m.handle_id
           WHERE cmj.chat_id=? AND m.associated_message_type BETWEEN 2000 AND 3999
           ORDER BY m.date ASC""",
        (chat_id,),
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
            state[(target, reactor_key)] = (reaction_label(t, r["associated_message_emoji"]), who)

    out = {}
    for (target, _reactor_key), (label, who) in state.items():
        out.setdefault(target, []).append((label, who))
    return out


def format_day_label(day_str):
    d = datetime.strptime(day_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    return d.strftime("%A, %B %-d, %Y")


def convert_heic(path):
    """HEIC/HEIF only render natively in Safari. Convert to JPEG on demand
    and cache the result so other browsers can display it."""
    digest = hashlib.sha1(path.encode()).hexdigest()
    out_path = os.path.join(CACHE_DIR, digest + ".jpg")
    if not os.path.exists(out_path):
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", path, "--out", out_path],
            capture_output=True,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
    return out_path


def make_thumb(path):
    """Max-edge 512px JPEG for chat bubbles and the media grid. Lightbox still
    uses the original via /attachment."""
    digest = hashlib.sha1(f"thumb{THUMB_SIZE}:{path}".encode()).hexdigest()
    out_path = os.path.join(CACHE_DIR, digest + ".jpg")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    result = subprocess.run(
        ["sips", "-Z", str(THUMB_SIZE), "-s", "format", "jpeg", path, "--out", out_path],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.exists(out_path):
        return None
    return out_path


def is_heic(path, mime):
    return mime in ("image/heic", "image/heif") or path.lower().endswith((".heic", ".heif"))


# ---------- Contacts ----------

def normalize_phone(s):
    digits = re.sub(r"\D", "", s or "")
    return digits[-10:] if len(digits) >= 10 else digits


def normalize_email(s):
    return (s or "").strip().lower()


def load_contacts():
    lookup = {}
    photos = {}
    paths = []
    for pattern in CONTACTS_GLOBS:
        paths.extend(glob.glob(pattern))

    for path in paths:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            names = {}
            record_photos = {}
            for r in conn.execute(
                "SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION, ZNICKNAME, "
                "ZTHUMBNAILIMAGEDATA, ZIMAGEDATA FROM ZABCDRECORD"
            ):
                full = " ".join(p for p in [r["ZFIRSTNAME"], r["ZLASTNAME"]] if p).strip()
                name = full or r["ZNICKNAME"] or r["ZORGANIZATION"]
                if name:
                    names[r["Z_PK"]] = name
                raw = r["ZTHUMBNAILIMAGEDATA"] or r["ZIMAGEDATA"]
                if raw and len(raw) > 100:
                    image = raw[1:]  # AddressBook prefixes a version byte
                    if image.startswith(b"\x89PNG"):
                        record_photos[r["Z_PK"]] = ("image/png", image)
                    elif image.startswith(b"\xff\xd8\xff"):
                        record_photos[r["Z_PK"]] = ("image/jpeg", image)
            for r in conn.execute("SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER"):
                name = names.get(r["ZOWNER"])
                key = normalize_phone(r["ZFULLNUMBER"])
                if name and key:
                    lookup["phone:" + key] = name
                    if r["ZOWNER"] in record_photos:
                        photos["phone:" + key] = record_photos[r["ZOWNER"]]
            for r in conn.execute("SELECT ZOWNER, ZADDRESS FROM ZABCDEMAILADDRESS"):
                name = names.get(r["ZOWNER"])
                if name and r["ZADDRESS"]:
                    email_key = "email:" + normalize_email(r["ZADDRESS"])
                    lookup[email_key] = name
                    if r["ZOWNER"] in record_photos:
                        photos[email_key] = record_photos[r["ZOWNER"]]
            conn.close()
        except Exception:
            continue

    # A photo reused byte-for-byte under two different names is a generic
    # placeholder (e.g. macOS's own "photo sync failed" icon), not a real
    # picture of either person, so drop it everywhere it appears.
    names_by_photo = {}
    for key, (_, image) in photos.items():
        names_by_photo.setdefault(image, set()).add(lookup[key])
    photos = {k: v for k, v in photos.items() if len(names_by_photo[v[1]]) == 1}

    return lookup, photos


CONTACTS, CONTACT_PHOTOS = load_contacts()
AVATAR_INDEX = {hashlib.sha1(k.encode()).hexdigest()[:16]: mime_bytes for k, mime_bytes in CONTACT_PHOTOS.items()}


def resolve_contact(identifier):
    if not identifier:
        return None
    if "@" in identifier:
        return CONTACTS.get("email:" + normalize_email(identifier))
    return CONTACTS.get("phone:" + normalize_phone(identifier))


def avatar_id(identifier):
    if not identifier:
        return None
    key = "email:" + normalize_email(identifier) if "@" in identifier else "phone:" + normalize_phone(identifier)
    if key not in CONTACT_PHOTOS:
        return None
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_participants(conn, chat_ids):
    if not chat_ids:
        return {}
    qmarks = ",".join("?" * len(chat_ids))
    out = {}
    for row in conn.execute(
        f"""SELECT chj.chat_id, h.id as handle
            FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id
            WHERE chj.chat_id IN ({qmarks})""",
        chat_ids,
    ):
        out.setdefault(row["chat_id"], []).append(row["handle"])
    return out


AVATAR_COLORS = ["#ff9500", "#ff3b30", "#af52de", "#5856d6", "#007aff", "#34c759", "#ff2d55", "#5ac8fa"]


def avatar_html(name, identifier=None):
    aid = avatar_id(identifier)
    if aid:
        return f'<img class="avatar" src="/avatar/{aid}">'
    initial = (name or "?").strip()[:1].upper() or "?"
    color = AVATAR_COLORS[sum(ord(c) for c in (name or "")) % len(AVATAR_COLORS)]
    return f'<span class="avatar" style="background:{color}">{html.escape(initial)}</span>'


def chat_label(display_name, identifier, participants):
    if display_name:
        return display_name
    resolved_self = resolve_contact(identifier)
    if resolved_self:
        return resolved_self
    if participants:
        return ", ".join(resolve_contact(p) or p for p in participants)
    return identifier or "Unknown"


# ---------- Messages ----------

MSG_SELECT = """SELECT m.ROWID as id, m.guid, m.text, m.attributedBody, m.date, m.is_from_me, h.id as handle
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            LEFT JOIN handle h ON h.ROWID = m.handle_id"""


def fetch_messages(conn, chat_id, after=None, before=None, start_ns=None, limit=PAGE_SIZE):
    """Page by (date, ROWID). `after`/`before` are (date_ns, rowid) cursors."""
    where = ["cmj.chat_id=?", REACTION_EXCLUDE_SQL]
    params = [chat_id]
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
    rows = conn.execute(
        f"{MSG_SELECT} WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?",
        params + [limit],
    ).fetchall()
    if before:
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
    return conn.execute(
        f"""SELECT 1 FROM message m JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE cmj.chat_id=? AND {REACTION_EXCLUDE_SQL} AND {clause} LIMIT 1""",
        (chat_id, date_ns, date_ns, rowid),
    ).fetchone() is not None


def load_attachments(conn, message_ids):
    if not message_ids:
        return {}
    qmarks = ",".join("?" * len(message_ids))
    out = {}
    for a in conn.execute(
        f"""SELECT maj.message_id, att.ROWID as att_id, att.mime_type, att.filename
            FROM message_attachment_join maj JOIN attachment att ON att.ROWID = maj.attachment_id
            WHERE maj.message_id IN ({qmarks})""",
        message_ids,
    ):
        out.setdefault(a["message_id"], []).append(a)
    return out


def render_message_blocks(
    rows,
    att_by_msg,
    reactions_by_guid=None,
    highlight_id=None,
    prev_day=None,
    prev_sender=None,
    next_day=None,
    next_sender=None,
):
    reactions_by_guid = reactions_by_guid or {}
    blocks = []
    n = len(rows)

    for idx, r in enumerate(rows):
        day = apple_date(r["date"])[:10]
        sender_key = (r["is_from_me"], r["handle"])

        if day != prev_day:
            if next_day is None or day != next_day:
                blocks.append(f'<div class="dateSep">{format_day_label(day)}</div>')
            prev_day = day
            prev_sender = None

        next_row = rows[idx + 1] if idx + 1 < n else None
        if next_row is not None:
            next_same_group = (
                (next_row["is_from_me"], next_row["handle"]) == sender_key
                and apple_date(next_row["date"])[:10] == day
            )
        elif next_sender is not None:
            next_same_group = next_sender == sender_key and next_day == day
        else:
            next_same_group = False
        is_last_in_group = not next_same_group
        is_first_in_group = sender_key != prev_sender

        who = "me" if r["is_from_me"] else "them"
        highlight_cls = " highlight" if highlight_id and r["id"] == highlight_id else ""
        group_cls = "" if not is_first_in_group else " group-start"
        tail_cls = " tail" if is_last_in_group else ""

        sender_label = ""
        if who == "them" and is_first_in_group:
            sender_label = html.escape(resolve_contact(r["handle"]) or r["handle"] or "Unknown")

        text = message_text(r)
        parts = []
        if sender_label:
            parts.append(f'<div class="sender">{sender_label}</div>')
        body = ""
        if text:
            body += f'<div class="bubble{tail_cls}">{html.escape(text)}</div>'
        for a in att_by_msg.get(r["id"], []):
            mime = a["mime_type"] or mimetypes.guess_type(a["filename"] or "")[0] or ""
            if mime.startswith("image/"):
                body += (
                    f'<img class="att" src="/thumb/{a["att_id"]}" '
                    f'data-full-src="/attachment/{a["att_id"]}" loading="lazy" data-msg-id="{r["id"]}">'
                )
            elif mime.startswith("video/"):
                body += f'<video class="att" src="/attachment/{a["att_id"]}" controls data-msg-id="{r["id"]}"></video>'
            else:
                fname = html.escape(os.path.basename(a["filename"] or "file"))
                body += f'<a class="att-file" href="/attachment/{a["att_id"]}">{fname}</a>'
        if not body:
            body = f'<div class="bubble{tail_cls}" style="opacity:.5">[no content]</div>'

        reactions = reactions_by_guid.get(r["guid"])
        if reactions:
            grouped = {}
            for label, who_reacted in reactions:
                grouped.setdefault(label, []).append(who_reacted)
            for label, names in grouped.items():
                body += f'<div class="reaction-pill">{html.escape(label)} &middot; {html.escape(", ".join(names))}</div>'

        parts.append(body)
        if is_last_in_group:
            parts.append(f'<div class="ts">{apple_date(r["date"])[11:]}</div>')
        blocks.append(f'<div class="row {who}{group_cls}{highlight_cls}" id="msg-{r["id"]}">{"".join(parts)}</div>')
        prev_sender = sender_key
    return "".join(blocks)


def fetch_messages_around(conn, chat_id, target_id, half=75):
    target = conn.execute("SELECT date FROM message WHERE ROWID=?", (target_id,)).fetchone()
    if not target:
        return []
    tgt_date = target["date"]
    before_rows = conn.execute(
        f"""{MSG_SELECT}
           WHERE cmj.chat_id=? AND (m.date < ? OR (m.date = ? AND m.ROWID <= ?)) AND {REACTION_EXCLUDE_SQL}
           ORDER BY m.date DESC, m.ROWID DESC LIMIT ?""",
        (chat_id, tgt_date, tgt_date, target_id, half),
    ).fetchall()
    after_rows = conn.execute(
        f"""{MSG_SELECT}
           WHERE cmj.chat_id=? AND (m.date > ? OR (m.date = ? AND m.ROWID > ?)) AND {REACTION_EXCLUDE_SQL}
           ORDER BY m.date ASC, m.ROWID ASC LIMIT ?""",
        (chat_id, tgt_date, tgt_date, target_id, half),
    ).fetchall()
    return list(reversed(before_rows)) + list(after_rows)


HEATMAP_COLORS = ["#e1e0d9", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]


def build_heatmap_html(conn, chat_id=None):
    if chat_id is not None:
        counts = conn.execute(
            """SELECT date(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as day, count(*) c
               FROM message m JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
               WHERE cmj.chat_id=? AND """ + REACTION_EXCLUDE_SQL + """
               GROUP BY day""",
            (chat_id,),
        ).fetchall()
    else:
        counts = conn.execute(
            """SELECT date(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as day, count(*) c
               FROM message m WHERE """ + REACTION_EXCLUDE_SQL + """
               GROUP BY day"""
        ).fetchall()
    if not counts:
        return ""

    day_counts = {r["day"]: r["c"] for r in counts}
    start = datetime.strptime(min(day_counts), "%Y-%m-%d").date()
    end = datetime.strptime(max(day_counts), "%Y-%m-%d").date()
    start_aligned = start - timedelta(days=(start.isoweekday() % 7))  # back up to the preceding Sunday
    max_count = max(day_counts.values())

    def bucket(c):
        if c == 0:
            return 0
        if c <= max_count * 0.25:
            return 1
        if c <= max_count * 0.5:
            return 2
        if c <= max_count * 0.75:
            return 3
        return 4

    total_days = (end - start_aligned).days + 1
    weeks = total_days // 7 + 1

    cells_by_week = []
    cur = start_aligned
    for _ in range(weeks):
        week = []
        for _ in range(7):
            week.append(cur.strftime("%Y-%m-%d") if cur >= start else None)
            cur += timedelta(days=1)
        cells_by_week.append(week)

    month_spans = []
    seen_months = set()
    first_span = True
    for wi, week in enumerate(cells_by_week):
        first_valid = next((d for d in week if d), None)
        if not first_valid:
            continue
        d = datetime.strptime(first_valid, "%Y-%m-%d").date()
        key = (d.year, d.month)
        if key not in seen_months and d.day <= 7:
            seen_months.add(key)
            label = d.strftime("%b '%y") if d.month == 1 or first_span else d.strftime("%b")
            month_spans.append((wi, label))
            first_span = False

    month_html = "".join(
        f'<span style="grid-column:{wi + 1}">{label}</span>' for wi, label in month_spans
    )

    cell_html_parts = []
    for week in cells_by_week:
        for day_str in week:
            if day_str is None:
                cell_html_parts.append('<div class="hcell"></div>')
                continue
            c = day_counts.get(day_str, 0)
            color = HEATMAP_COLORS[bucket(c)]
            title = f"{day_str}: {c} message{'s' if c != 1 else ''}"
            clickable = c and chat_id is not None
            onclick = f' onclick="location.href=\'/chat/{chat_id}?date={day_str}\'"' if clickable else ""
            cls = "hcell clickable" if clickable else "hcell"
            cell_html_parts.append(f'<div class="{cls}" style="background:{color}" title="{title}"{onclick}></div>')

    return f"""<div class="heatmap-wrap"><div class="heatmap-scroll">
<div class="heatmap-months">{month_html}</div>
<div class="heatmap-grid">{''.join(cell_html_parts)}</div>
</div></div>"""


# ---------- Rendering ----------

PAGE_CSS = """
body { font-family: -apple-system, sans-serif; background: #f2f2f7; margin: 0; color: #1c1c1e; }
header { background: #fff; padding: 12px 20px; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 10; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
header a { color: #007aff; text-decoration: none; }
header input[type=date] { border: 1px solid #ccc; border-radius: 6px; padding: 5px 8px; font-size: 13px; }
main { max-width: 900px; margin: 0 auto; padding: 16px; }
input#filter { width: 100%; padding: 10px; font-size: 15px; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; margin-bottom: 12px; }
table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }
tr.chatrow { cursor: pointer; border-bottom: 1px solid #eee; }
tr.chatrow:hover { background: #f5f8ff; }
td { padding: 10px 14px; font-size: 14px; }
td.name { font-weight: 600; }
td.count, td.last { color: #666; text-align: right; white-space: nowrap; }
body.chatpage { background: #fff; }
.bubblewrap { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; }
.row { display: flex; flex-direction: column; max-width: 70%; margin-top: 1px; }
.row.group-start { margin-top: 12px; }
.row.me { align-self: flex-end; align-items: flex-end; }
.row.them { align-self: flex-start; align-items: flex-start; }
.sender { font-size: 11px; color: #888; margin: 0 10px 2px; }
.bubble { position: relative; padding: 8px 14px; border-radius: 18px; font-size: 15px; line-height: 1.35; white-space: pre-wrap; word-wrap: break-word; }
.row.me .bubble { background: #0b84ff; color: #fff; }
.row.them .bubble { background: #e5e5ea; color: #000; }
.bubble.tail::after { content: ""; position: absolute; bottom: 0; width: 12px; height: 14px; }
.row.me .bubble.tail::after { right: -5px; background: #0b84ff; clip-path: polygon(0 30%, 100% 100%, 0 100%); }
.row.them .bubble.tail::after { left: -5px; background: #e5e5ea; clip-path: polygon(100% 30%, 100% 100%, 0 100%); }
.ts { font-size: 10px; color: #aaa; margin: 3px 10px 0; }
.dateSep { align-self: center; font-size: 12px; color: #888; padding: 4px 12px; margin: 16px 0 6px; }
.reaction-pill { font-size: 11px; background: rgba(0,0,0,0.06); color: #555; padding: 2px 9px; border-radius: 10px; margin-top: 4px; display: inline-block; }
img.att, video.att { max-width: 280px; max-height: 280px; border-radius: 16px; display: block; margin-top: 4px; }
a.att-file { display: block; margin-top: 4px; font-size: 13px; }
#sentinel, #sentinel-top { text-align: center; padding: 16px; color: #999; font-size: 13px; min-height: 1px; }
.row.highlight .bubble, .row.highlight img.att, .row.highlight video.att { outline: 3px solid #ffcc00; }
.searchbox { padding: 5px 8px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px; width: 140px; }
.mediagrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 3px; }
.tile { display: block; aspect-ratio: 1; overflow: hidden; border-radius: 4px; background: #e5e5ea; }
.tile img, .tile video { width: 100%; height: 100%; object-fit: cover; }
.filetile { display: flex; align-items: center; justify-content: center; height: 100%; font-size: 11px; padding: 6px; text-align: center; color: #666; }
.media-month { margin-bottom: 18px; }
.media-month-h { font-size: 15px; font-weight: 600; color: #1c1c1e; margin: 0 0 8px 2px; position: sticky; top: 44px; background: #f2f2f7; padding: 6px 0; z-index: 5; }
.media-rail { position: fixed; right: 4px; top: 70px; bottom: 16px; width: 46px; z-index: 30; display: flex; }
.media-rail-track { position: relative; flex: 1; cursor: pointer; touch-action: none; }
.media-rail-tick { position: absolute; right: 6px; transform: translateY(-50%); font-size: 10px; color: #9a9a9a; pointer-events: none; white-space: nowrap; font-variant-numeric: tabular-nums; }
.media-rail-dot { position: absolute; right: 2px; width: 5px; height: 5px; border-radius: 50%; background: #0b84ff; transform: translateY(-50%); pointer-events: none; transition: top 0.05s linear; }
.media-rail-label { position: absolute; right: 100%; margin-right: 10px; transform: translateY(-50%); background: #0b0b0b; color: #fff; font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 6px; white-space: nowrap; opacity: 0; transition: opacity 0.15s; pointer-events: none; }
.media-rail.active .media-rail-label { opacity: 1; }
body.mediapage main, body.chatpage main { padding-right: 56px; }
.searchresult { display: block; background: #fff; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; text-decoration: none; color: inherit; }
.searchresult:hover { background: #f5f8ff; }
.sr-meta { font-size: 11px; color: #888; margin-bottom: 4px; }
.sr-text { font-size: 14px; white-space: pre-wrap; }
.avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 14px; font-weight: 600; margin-right: 4px; flex-shrink: 0; object-fit: cover; }
tr.chatrow td.name { display: flex; align-items: center; gap: 10px; }
.heatmap-wrap { background: #fff; border-radius: 8px; padding: 10px 14px 12px; margin-bottom: 12px; overflow-x: auto; }
.heatmap-months { display: grid; grid-auto-flow: column; grid-auto-columns: 14px; font-size: 10px; color: #999; height: 14px; margin-bottom: 3px; white-space: nowrap; }
.heatmap-grid { display: grid; grid-template-rows: repeat(7, 11px); grid-auto-flow: column; grid-auto-columns: 11px; gap: 3px; }
.hcell { width: 11px; height: 11px; border-radius: 2px; }
.hcell.clickable { cursor: pointer; }
.hcell.clickable:hover { outline: 1px solid #007aff; }
.section-h { font-size: 15px; margin: 22px 0 10px; color: #0b0b0b; }
.section-sub { font-size: 12px; color: #898781; margin: -6px 0 10px; }
.panel { background: #fff; border-radius: 8px; padding: 14px; }
.kpirow { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.stattile { background: #fff; border-radius: 8px; padding: 16px; text-align: center; }
.stat-value { font-size: 30px; font-weight: 600; color: #0b0b0b; }
.stat-label { font-size: 12px; color: #898781; margin-top: 4px; }
.lbrow { display: grid; grid-template-columns: 170px 1fr 56px; align-items: center; gap: 10px; margin-bottom: 10px; opacity: 0; animation: rowIn 0.4s ease forwards; }
.lbname { display: flex; align-items: center; gap: 8px; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lbtrack { background: #ebedf0; border-radius: 4px; height: 14px; overflow: hidden; }
.lbbar { height: 100%; width: 0; background: #2a78d6; border-radius: 4px; transition: width 1s cubic-bezier(.22,1,.36,1); }
.lbcount { font-size: 12px; color: #52514e; font-variant-numeric: tabular-nums; text-align: right; }
@keyframes rowIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }
.ccard { display: flex; justify-content: space-between; align-items: center; background: #fff; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; opacity: 0; animation: rowIn 0.4s ease forwards; }
.lbrow.clickable, .ccard.clickable { cursor: pointer; }
.lbrow.clickable { margin-left: -8px; margin-right: -8px; padding: 4px 8px; border-radius: 6px; }
.lbrow.clickable:hover, .ccard.clickable:hover { background: #f5f8ff; }
.ccard-left { display: flex; align-items: center; gap: 10px; }
.ccard-name { font-size: 14px; font-weight: 600; color: #0b0b0b; }
.ccard-sub { font-size: 12px; color: #898781; margin-top: 2px; }
.ccard-count { font-size: 15px; font-weight: 600; color: #0b0b0b; font-variant-numeric: tabular-nums; }
.trendwrap { position: relative; background: #fff; border-radius: 8px; padding: 12px 14px; }
.trendsvg { width: 100%; height: 220px; overflow: visible; display: block; }
.axisline { stroke: #c3c2b7; stroke-width: 1; }
.axislabel { font-size: 9px; fill: #898781; }
.areapath { fill: #2a78d6; fill-opacity: 0.10; stroke: none; }
.linepath { fill: none; stroke: #2a78d6; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.pt { fill: #2a78d6; stroke: #fff; stroke-width: 2; }
.hitcol { fill: transparent; cursor: crosshair; }
.trendtip { position: absolute; pointer-events: none; background: #0b0b0b; color: #fff; font-size: 11px; padding: 4px 8px; border-radius: 6px; opacity: 0; transform: translate(-50%,-130%); white-space: nowrap; transition: opacity 0.1s; }
img.att, .tile img, .tile video { cursor: pointer; }
.lightbox-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.92); z-index: 1000; display: none; align-items: center; justify-content: center; }
.lightbox-overlay.open { display: flex; }
.lightbox-content { max-width: 92vw; max-height: 92vh; display: flex; align-items: center; justify-content: center; }
.lightbox-content img, .lightbox-content video { max-width: 92vw; max-height: 92vh; object-fit: contain; border-radius: 6px; }
.lightbox-close { position: absolute; top: 18px; right: 24px; color: #fff; font-size: 30px; cursor: pointer; background: none; border: none; line-height: 1; padding: 4px 10px; }
.lightbox-nav { position: absolute; top: 50%; transform: translateY(-50%); color: #fff; font-size: 26px; background: rgba(255,255,255,0.1); border: none; width: 44px; height: 44px; border-radius: 50%; cursor: pointer; }
.lightbox-nav:hover { background: rgba(255,255,255,0.2); }
.lightbox-prev { left: 18px; }
.lightbox-next { right: 18px; }
.ctx-menu { position: fixed; background: #fff; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); padding: 4px; z-index: 1100; min-width: 150px; display: none; }
.ctx-menu.open { display: block; }
.ctx-menu-item { padding: 8px 12px; font-size: 13px; cursor: pointer; border-radius: 5px; }
.ctx-menu-item:hover { background: #f2f2f7; }
"""


SORT_OPTIONS = {
    "recent": "Most recent activity",
    "count": "Most messages",
    "name": "Name (A-Z)",
    "oldest": "Oldest conversation",
}


def render_chat_list(sort="recent"):
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.ROWID as id, c.chat_identifier, c.display_name,
                  count(m.ROWID) as msg_count, max(m.date) as last_date, min(m.date) as first_date
           FROM chat c
           JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID
           JOIN message m ON m.ROWID = cmj.message_id
           GROUP BY c.ROWID"""
    ).fetchall()

    need_participants = [
        r["id"] for r in rows if not r["display_name"] and not resolve_contact(r["chat_identifier"])
    ]
    participants_map = load_participants(conn, need_participants)
    conn.close()

    def is_known(display_name, chat_identifier, participants):
        if display_name or resolve_contact(chat_identifier):
            return True
        return bool(participants) and any(resolve_contact(p) for p in participants)

    items = [
        {
            "id": r["id"],
            "name": chat_label(r["display_name"], r["chat_identifier"], participants_map.get(r["id"])),
            "identifier": r["chat_identifier"],
            "count": r["msg_count"],
            "last": r["last_date"],
            "first": r["first_date"],
            "known": is_known(r["display_name"], r["chat_identifier"], participants_map.get(r["id"])),
        }
        for r in rows
    ]

    if sort == "name":
        items.sort(key=lambda x: x["name"].lower())
    elif sort == "count":
        items.sort(key=lambda x: -x["count"])
    elif sort == "oldest":
        items.sort(key=lambda x: x["first"])
    else:
        sort = "recent"
        items.sort(key=lambda x: -x["last"])

    trs = []
    for it in items:
        trs.append(
            f'<tr class="chatrow" onclick="location.href=\'/chat/{it["id"]}\'" '
            f'data-search="{html.escape(it["name"].lower())}" data-known="{"1" if it["known"] else "0"}">'
            f'<td class="name">{avatar_html(it["name"], it["identifier"])}{html.escape(it["name"])}</td>'
            f'<td class="count">{it["count"]}</td>'
            f'<td class="last">{apple_date(it["last"])}</td>'
            f"</tr>"
        )

    opts_html = "".join(
        f'<option value="{k}"{" selected" if k == sort else ""}>{v}</option>'
        for k, v in SORT_OPTIONS.items()
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Messages</title><style>{PAGE_CSS}</style></head><body>
<header><b>Messages</b> &middot; {len(rows)} conversations
<a href="/stats">Stats</a>
<select id="sortSelect" onchange="location.href='/?sort='+this.value" style="margin-left:auto; padding:6px 8px; border:1px solid #ccc; border-radius:6px;">{opts_html}</select>
</header>
<main>
<input id="filter" placeholder="Filter conversations..." oninput="filterRows()">
<label style="display:flex; align-items:center; gap:6px; font-size:13px; color:#555; margin:-6px 0 12px;">
<input type="checkbox" id="knownOnly" onchange="filterRows()"> Known contacts only
</label>
<table><tbody id="rows">{''.join(trs)}</tbody></table>
</main>
<script>
function filterRows() {{
  const q = document.getElementById('filter').value.toLowerCase();
  const knownOnly = document.getElementById('knownOnly').checked;
  document.querySelectorAll('#rows tr').forEach(tr => {{
    const matchesText = tr.dataset.search.includes(q);
    const matchesKnown = !knownOnly || tr.dataset.known === '1';
    tr.style.display = matchesText && matchesKnown ? '' : 'none';
  }});
}}
</script>
</body></html>"""


def render_chat(chat_id, date_str=None, around_id=None):
    conn = get_conn()
    chat = conn.execute(
        "SELECT ROWID as id, chat_identifier, display_name FROM chat WHERE ROWID=?", (chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        return None

    bounds = conn.execute(
        """SELECT min(m.date) as lo, max(m.date) as hi FROM message m
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID WHERE cmj.chat_id=?""",
        (chat_id,),
    ).fetchone()
    media_count = conn.execute(
        """SELECT count(*) FROM message_attachment_join maj
           JOIN chat_message_join cmj ON cmj.message_id = maj.message_id
           JOIN attachment att ON att.ROWID = maj.attachment_id
           WHERE cmj.chat_id=? AND att.filename NOT LIKE '%.pluginPayloadAttachment'""",
        (chat_id,),
    ).fetchone()[0]

    if around_id:
        rows = fetch_messages_around(conn, chat_id, around_id)
    else:
        start_ns = None
        if date_str:
            try:
                start_ns = date_to_apple_ns(date_str)
            except ValueError:
                start_ns = None
        rows = fetch_messages(conn, chat_id, start_ns=start_ns, limit=PAGE_SIZE)

    ids = [r["id"] for r in rows]
    att_by_msg = load_attachments(conn, ids)
    reactions_by_guid = load_reactions(conn, chat_id, [r["guid"] for r in rows])
    blocks_html = render_message_blocks(rows, att_by_msg, reactions_by_guid, highlight_id=around_id)

    has_older = has_newer = False
    if rows:
        has_older = has_neighbor(conn, chat_id, rows[0]["date"], rows[0]["id"], "before")
        has_newer = has_neighbor(conn, chat_id, rows[-1]["date"], rows[-1]["id"], "after")

    participants = None
    if not chat["display_name"] and not resolve_contact(chat["chat_identifier"]):
        participants = load_participants(conn, [chat_id]).get(chat_id)
    title = chat_label(chat["display_name"], chat["chat_identifier"], participants)
    conn.close()

    first_id = rows[0]["id"] if rows else 0
    first_date = str(rows[0]["date"]) if rows else "0"
    last_id = rows[-1]["id"] if rows else 0
    last_date = str(rows[-1]["date"]) if rows else "0"
    min_date = apple_date(bounds["lo"])[:10] if bounds["lo"] else ""
    max_date = apple_date(bounds["hi"])[:10] if bounds["hi"] else ""
    if around_id and rows:
        around_row = next((r for r in rows if r["id"] == around_id), rows[0])
        cur_date = apple_date(around_row["date"])[:10]
    elif date_str:
        cur_date = date_str
    else:
        cur_date = min_date

    around_js = json.dumps(around_id)
    top_label = "" if has_older else "Beginning of conversation"
    bottom_label = "Loading more..." if has_newer else "End of conversation"
    rail_html = ""
    rail_script = ""
    if min_date and max_date:
        rail_html = """<div class="media-rail" id="chatRail">
<div class="media-rail-track" id="chatRailTrack"><div class="media-rail-dot" id="chatRailDot"></div></div>
<div class="media-rail-label" id="chatRailLabel"></div>
</div>"""
        rail_script = f"<script>{chat_rail_script(chat_id, min_date, max_date, cur_date)}</script>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{PAGE_CSS}</style></head><body class="chatpage">
<header>
<a href="/">&larr; All conversations</a>
<b>{html.escape(title)}</b>
<a href="/chat/{chat_id}/media">Media ({media_count})</a>
<form method="get" action="/chat/{chat_id}/search">
<input class="searchbox" type="text" name="q" placeholder="Search this conversation...">
</form>
<span style="margin-left:auto; display:flex; align-items:center; gap:6px;">
<label for="datepicker" style="font-size:13px; color:#666;">Jump to date</label>
<input type="date" id="datepicker" value="{cur_date}" min="{min_date}" max="{max_date}">
</span>
</header>
<main>
<div id="sentinel-top">{top_label}</div>
<div class="bubblewrap" id="messages">{blocks_html}</div>
<div id="sentinel">{bottom_label}</div>
</main>
{rail_html}
<script>
let firstDate = {json.dumps(first_date)};
let firstId = {first_id};
let lastDate = {json.dumps(last_date)};
let lastId = {last_id};
let hasOlder = {json.dumps(has_older)};
let hasNewer = {json.dumps(has_newer)};
let loading = false;
const chatId = {chat_id};
const aroundId = {around_js};
const topSentinel = document.getElementById('sentinel-top');
const sentinel = document.getElementById('sentinel');
const wrap = document.getElementById('messages');

const olderObs = new IntersectionObserver(entries => {{
  if (entries[0].isIntersecting) loadOlder();
}}, {{rootMargin: '600px'}});
const newerObs = new IntersectionObserver(entries => {{
  if (entries[0].isIntersecting) loadNewer();
}}, {{rootMargin: '600px'}});

function armObservers() {{
  if (hasOlder) olderObs.observe(topSentinel);
  if (hasNewer) newerObs.observe(sentinel);
}}

function isNear(el) {{
  const r = el.getBoundingClientRect();
  return r.top < window.innerHeight + 600 && r.bottom > -600;
}}
function fillIfNeeded() {{
  if (hasNewer && isNear(sentinel)) loadNewer();
  else if (hasOlder && isNear(topSentinel)) loadOlder();
}}

async function loadNewer() {{
  if (loading || !hasNewer) return;
  loading = true;
  const res = await fetch(`/chat/${{chatId}}/more?after_date=${{lastDate}}&after_id=${{lastId}}`);
  const data = await res.json();
  if (data.html) {{
    wrap.insertAdjacentHTML('beforeend', data.html);
    lastDate = data.last_date;
    lastId = data.last_id;
  }}
  hasNewer = data.has_more_newer;
  loading = false;
  sentinel.textContent = hasNewer ? 'Loading more...' : 'End of conversation';
  if (!hasNewer) newerObs.unobserve(sentinel);
  fillIfNeeded();
}}

async function loadOlder() {{
  if (loading || !hasOlder) return;
  loading = true;
  const cursorId = firstId;
  const prevHeight = document.documentElement.scrollHeight;
  const prevScroll = window.scrollY;
  const res = await fetch(`/chat/${{chatId}}/more?before_date=${{firstDate}}&before_id=${{firstId}}`);
  const data = await res.json();
  if (data.html) {{
    wrap.insertAdjacentHTML('afterbegin', data.html);
    firstDate = data.first_date;
    firstId = data.first_id;
    if (data.strip_group_start) {{
      const el = document.getElementById('msg-' + cursorId);
      if (el) {{
        el.classList.remove('group-start');
        const sender = el.querySelector('.sender');
        if (sender) sender.remove();
      }}
    }}
    window.scrollTo(0, prevScroll + (document.documentElement.scrollHeight - prevHeight));
  }}
  hasOlder = data.has_more_older;
  loading = false;
  topSentinel.textContent = hasOlder ? '' : 'Beginning of conversation';
  if (!hasOlder) olderObs.unobserve(topSentinel);
  fillIfNeeded();
}}

document.getElementById('datepicker').addEventListener('change', e => {{
  if (e.target.value) location.href = `/chat/${{chatId}}?date=${{e.target.value}}`;
}});

if (aroundId) {{
  window.addEventListener('DOMContentLoaded', () => {{
    const el = document.getElementById('msg-' + aroundId);
    if (el) {{
      el.scrollIntoView({{block: 'center'}});
      setTimeout(() => el.classList.remove('highlight'), 3000);
    }}
    requestAnimationFrame(() => requestAnimationFrame(armObservers));
  }});
}} else {{
  armObservers();
}}
</script>
<script>{lightbox_script(chat_id)}</script>
{rail_script}
</body></html>"""


def month_label(ym):
    return datetime.strptime(ym, "%Y-%m").strftime("%B %Y")


MEDIA_RAIL_SCRIPT = """
(function() {
  const sections = Array.from(document.querySelectorAll('.media-month'));
  const rail = document.getElementById('mediaRail');
  if (!sections.length || !rail) return;
  const track = document.getElementById('mediaRailTrack');
  const dot = document.getElementById('mediaRailDot');
  const label = document.getElementById('mediaRailLabel');
  let dragging = false;

  function docHeight() {
    return Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  }

  function layoutTicks() {
    track.querySelectorAll('.media-rail-tick').forEach(t => t.remove());
    const total = document.documentElement.scrollHeight;
    let lastYear = null;
    sections.forEach(sec => {
      const year = sec.dataset.year;
      if (year !== lastYear) {
        lastYear = year;
        const tick = document.createElement('div');
        tick.className = 'media-rail-tick';
        tick.style.top = (sec.offsetTop / total * 100) + '%';
        tick.textContent = year;
        track.appendChild(tick);
      }
    });
  }

  function sectionAtFrac(frac) {
    const targetTop = frac * docHeight();
    let cur = sections[0];
    for (const sec of sections) {
      if (sec.offsetTop <= targetTop + 60) cur = sec; else break;
    }
    return cur;
  }

  function updateDot() {
    const frac = window.scrollY / docHeight();
    dot.style.top = (frac * 100) + '%';
  }

  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => { updateDot(); ticking = false; });
      ticking = true;
    }
  });
  window.addEventListener('resize', layoutTicks);

  track.addEventListener('mousemove', e => {
    const rect = track.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    const sec = sectionAtFrac(frac);
    label.textContent = sec.dataset.label;
    label.style.top = (frac * 100) + '%';
    rail.classList.add('active');
    if (dragging) window.scrollTo(0, frac * docHeight());
  });
  track.addEventListener('mouseleave', () => { if (!dragging) rail.classList.remove('active'); });
  track.addEventListener('mousedown', e => {
    dragging = true;
    const rect = track.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    window.scrollTo(0, frac * docHeight());
  });
  window.addEventListener('mouseup', () => { dragging = false; });

  window.addEventListener('load', () => { layoutTicks(); updateDot(); });
  layoutTicks();
  updateDot();
})();
"""


def chat_rail_script(chat_id, min_date, max_date, cur_date):
    """Year scrubber mapped to the conversation's date range. Click/drag jumps
    to that day; the dot marks the current window, not page scroll."""
    return f"""
(function() {{
  const rail = document.getElementById('chatRail');
  if (!rail) return;
  const track = document.getElementById('chatRailTrack');
  const dot = document.getElementById('chatRailDot');
  const label = document.getElementById('chatRailLabel');
  const chatId = {chat_id};
  const curDate = {json.dumps(cur_date)};
  const t0 = Date.parse({json.dumps(min_date)} + 'T00:00:00');
  const t1 = Date.parse({json.dumps(max_date)} + 'T00:00:00');
  const span = Math.max(1, t1 - t0);
  let dragging = false;
  let lastFrac = 0;

  function fracOf(dateStr) {{
    return Math.min(1, Math.max(0, (Date.parse(dateStr + 'T00:00:00') - t0) / span));
  }}
  function dateAt(frac) {{
    const d = new Date(t0 + frac * span);
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return d.getFullYear() + '-' + m + '-' + day;
  }}
  const startYear = new Date(t0).getFullYear();
  const endYear = new Date(t1).getFullYear();
  for (let y = startYear; y <= endYear; y++) {{
    const frac = fracOf(y + '-01-01');
    const tick = document.createElement('div');
    tick.className = 'media-rail-tick';
    tick.style.top = (frac * 100) + '%';
    tick.textContent = y;
    track.appendChild(tick);
  }}

  function setDot(frac) {{
    dot.style.top = (frac * 100) + '%';
  }}
  function showLabel(frac) {{
    lastFrac = frac;
    label.textContent = dateAt(frac);
    label.style.top = (frac * 100) + '%';
    rail.classList.add('active');
  }}
  function fracFromEvent(e) {{
    const rect = track.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
  }}

  setDot(fracOf(curDate));

  track.addEventListener('mousemove', e => {{
    const frac = fracFromEvent(e);
    showLabel(frac);
  }});
  track.addEventListener('mouseleave', () => {{ if (!dragging) rail.classList.remove('active'); }});
  track.addEventListener('mousedown', e => {{
    dragging = true;
    showLabel(fracFromEvent(e));
    e.preventDefault();
  }});
  window.addEventListener('mouseup', () => {{
    if (!dragging) return;
    dragging = false;
    rail.classList.remove('active');
    const next = dateAt(lastFrac);
    if (next !== curDate) location.href = '/chat/' + chatId + '?date=' + next;
  }});
}})();
"""


def lightbox_script(chat_id):
    return f"""
(function() {{
  const chatId = {chat_id};
  const SELECTOR = '.tile, img.att, video.att';

  function mediaInfo(el) {{
    const inner = el.classList.contains('tile') ? el.querySelector('img,video') : el;
    if (!inner) return null;
    return {{ src: inner.getAttribute('data-full-src') || el.getAttribute('data-full-src') || inner.getAttribute('src'), isVideo: inner.tagName === 'VIDEO', msgId: el.dataset.msgId, node: el }};
  }}

  function collectItems() {{
    return Array.from(document.querySelectorAll(SELECTOR)).map(mediaInfo).filter(Boolean);
  }}

  const overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  overlay.innerHTML =
    '<button class="lightbox-close" aria-label="Close">&times;</button>' +
    '<button class="lightbox-nav lightbox-prev" aria-label="Previous">&#8249;</button>' +
    '<div class="lightbox-content"></div>' +
    '<button class="lightbox-nav lightbox-next" aria-label="Next">&#8250;</button>';
  document.body.appendChild(overlay);
  const content = overlay.querySelector('.lightbox-content');

  const menu = document.createElement('div');
  menu.className = 'ctx-menu';
  menu.innerHTML = '<div class="ctx-menu-item" id="ctxShowInChat">Show in chat</div>';
  document.body.appendChild(menu);
  let menuMsgId = null;
  let items = [];
  let curIndex = 0;

  function render(i) {{
    curIndex = (i + items.length) % items.length;
    const it = items[curIndex];
    content.innerHTML = it.isVideo
      ? `<video src="${{it.src}}" controls autoplay></video>`
      : `<img src="${{it.src}}">`;
  }}

  function openAt(node) {{
    items = collectItems();
    const idx = items.findIndex(it => it.node === node);
    if (idx === -1) return;
    render(idx);
    overlay.classList.add('open');
  }}

  function close() {{
    overlay.classList.remove('open');
    content.innerHTML = '';
  }}

  overlay.querySelector('.lightbox-close').addEventListener('click', close);
  overlay.addEventListener('click', e => {{ if (e.target === overlay) close(); }});
  overlay.querySelector('.lightbox-prev').addEventListener('click', () => render(curIndex - 1));
  overlay.querySelector('.lightbox-next').addEventListener('click', () => render(curIndex + 1));
  document.addEventListener('keydown', e => {{
    if (!overlay.classList.contains('open')) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowLeft') render(curIndex - 1);
    else if (e.key === 'ArrowRight') render(curIndex + 1);
  }});

  document.body.addEventListener('click', e => {{
    const el = e.target.closest(SELECTOR);
    if (!el || !mediaInfo(el)) return;
    e.preventDefault();
    openAt(el);
  }});

  document.body.addEventListener('contextmenu', e => {{
    const el = e.target.closest(SELECTOR);
    if (!el || !mediaInfo(el)) return;
    e.preventDefault();
    menuMsgId = el.dataset.msgId;
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.classList.add('open');
  }});

  document.getElementById('ctxShowInChat').addEventListener('click', () => {{
    if (menuMsgId) location.href = '/chat/' + chatId + '?around=' + menuMsgId;
  }});
  document.addEventListener('click', e => {{
    if (!menu.contains(e.target)) menu.classList.remove('open');
  }});
}})();
"""


def render_media(chat_id):
    conn = get_conn()
    chat = conn.execute(
        "SELECT ROWID as id, chat_identifier, display_name FROM chat WHERE ROWID=?", (chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        return None

    media = conn.execute(
        """SELECT att.ROWID as att_id, att.mime_type, att.filename, m.ROWID as msg_id, m.date
           FROM message_attachment_join maj
           JOIN attachment att ON att.ROWID = maj.attachment_id
           JOIN message m ON m.ROWID = maj.message_id
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           WHERE cmj.chat_id = ? AND att.filename NOT LIKE '%.pluginPayloadAttachment'
           ORDER BY m.date DESC""",
        (chat_id,),
    ).fetchall()

    participants = None
    if not chat["display_name"] and not resolve_contact(chat["chat_identifier"]):
        participants = load_participants(conn, [chat_id]).get(chat_id)
    title = chat_label(chat["display_name"], chat["chat_identifier"], participants)
    conn.close()

    sections = []
    cur_ym = None
    tiles = []
    for a in media:
        ym = apple_date(a["date"])[:7]
        if ym != cur_ym:
            if tiles:
                sections.append((cur_ym, tiles))
            cur_ym = ym
            tiles = []
        mime = a["mime_type"] or mimetypes.guess_type(a["filename"] or "")[0] or ""
        link = f'/chat/{chat_id}?around={a["msg_id"]}'
        if mime.startswith("image/"):
            inner = f'<img src="/thumb/{a["att_id"]}" data-full-src="/attachment/{a["att_id"]}" loading="lazy">'
        elif mime.startswith("video/"):
            inner = f'<video src="/attachment/{a["att_id"]}" preload="metadata" muted></video>'
        else:
            fname = html.escape(os.path.basename(a["filename"] or "file"))
            inner = f'<div class="filetile">{fname}</div>'
        tiles.append(f'<a class="tile" href="{link}" title="{apple_date(a["date"])}" data-msg-id="{a["msg_id"]}">{inner}</a>')
    if tiles:
        sections.append((cur_ym, tiles))

    if sections:
        section_html = "".join(
            f'<section class="media-month" data-year="{ym[:4]}" data-label="{month_label(ym)}">'
            f'<h3 class="media-month-h">{month_label(ym)}</h3>'
            f'<div class="mediagrid">{"".join(tile_htmls)}</div></section>'
            for ym, tile_htmls in sections
        )
        rail_html = f"""<div class="media-rail" id="mediaRail">
<div class="media-rail-track" id="mediaRailTrack"><div class="media-rail-dot" id="mediaRailDot"></div></div>
<div class="media-rail-label" id="mediaRailLabel"></div>
</div>"""
    else:
        section_html = '<p style="color:#999">No media in this conversation.</p>'
        rail_html = ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Media: {html.escape(title)}</title><style>{PAGE_CSS}</style></head><body class="mediapage">
<header><a href="/chat/{chat_id}">&larr; {html.escape(title)}</a> <b>Media</b> &middot; {len(media)} items</header>
<main>{section_html}</main>
{rail_html}
<script>{MEDIA_RAIL_SCRIPT}</script>
<script>{lightbox_script(chat_id)}</script>
</body></html>"""


def render_search(chat_id, query):
    conn = get_conn()
    chat = conn.execute(
        "SELECT ROWID as id, chat_identifier, display_name FROM chat WHERE ROWID=?", (chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        return None

    rows = conn.execute(
        f"""SELECT m.ROWID as id, m.text, m.attributedBody, m.date, m.is_from_me, h.id as handle
           FROM message m
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           LEFT JOIN handle h ON h.ROWID = m.handle_id
           WHERE cmj.chat_id=? AND {REACTION_EXCLUDE_SQL}
           ORDER BY m.date DESC""",
        (chat_id,),
    ).fetchall()

    participants = None
    if not chat["display_name"] and not resolve_contact(chat["chat_identifier"]):
        participants = load_participants(conn, [chat_id]).get(chat_id)
    title = chat_label(chat["display_name"], chat["chat_identifier"], participants)
    conn.close()

    q = (query or "").strip().lower()
    matches = []
    if q:
        for r in rows:
            text = message_text(r)
            if text and q in text.lower():
                matches.append((r, text))

    items = []
    for r, text in matches[:300]:
        who = "Me" if r["is_from_me"] else (resolve_contact(r["handle"]) or r["handle"] or "Unknown")
        items.append(
            f'<a class="searchresult" href="/chat/{chat_id}?around={r["id"]}">'
            f'<div class="sr-meta">{html.escape(who)} &middot; {apple_date(r["date"])}</div>'
            f'<div class="sr-text">{html.escape(text)}</div></a>'
        )

    note = ""
    if len(matches) > 300:
        note = f'<p style="color:#999">Showing first 300 of {len(matches)} matches.</p>'

    body = "".join(items) if items else ('<p style="color:#999">No matches.</p>' if q else "")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Search: {html.escape(title)}</title><style>{PAGE_CSS}</style></head><body>
<header>
<a href="/chat/{chat_id}">&larr; {html.escape(title)}</a>
<form method="get" action="/chat/{chat_id}/search" style="margin-left:auto; display:flex; gap:6px;">
<input class="searchbox" type="text" name="q" value="{html.escape(query or '')}" placeholder="Search this conversation..." style="width:220px;">
<button type="submit" style="padding:6px 12px; border:none; border-radius:6px; background:#007aff; color:#fff;">Search</button>
</form>
</header>
<main>
<p style="color:#666">{len(matches)} match{'es' if len(matches) != 1 else ''} for &ldquo;{html.escape(query or '')}&rdquo;</p>
{note}
{body}
</main>
</body></html>"""


# ---------- Stats ----------

LONG_TERM_MIN_SPAN_DAYS = 365
LONG_TERM_MAX_GAP_DAYS = 90
FELL_OFF_MIN_COUNT = 30
FELL_OFF_MIN_GAP_DAYS = 180


def apple_to_datetime(ns):
    return datetime.fromtimestamp(ns / 1_000_000_000 + APPLE_EPOCH)


def best_chat_per_handle(conn):
    """Find the chat to open when someone clicks a contact: the direct 1:1
    chat if one exists, otherwise their highest-volume shared chat."""
    rows = conn.execute(
        """SELECT h.id as handle, cmj.chat_id as chat_id,
                  (SELECT count(*) FROM chat_handle_join x WHERE x.chat_id = cmj.chat_id) as n_participants,
                  count(*) as msg_count
           FROM message m
           JOIN handle h ON h.ROWID = m.handle_id
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           WHERE m.is_from_me = 0
           GROUP BY h.id, cmj.chat_id"""
    ).fetchall()
    best = {}
    for r in rows:
        is_dm = r["n_participants"] == 1
        cur = best.get(r["handle"])
        if cur is None or (is_dm, r["msg_count"]) > (cur[1], cur[2]):
            best[r["handle"]] = (r["chat_id"], is_dm, r["msg_count"])
    return {handle: chat_id for handle, (chat_id, _, _) in best.items()}


def compute_contact_stats(conn):
    counts = conn.execute(
        f"""SELECT h.id as handle, count(*) c, min(m.date) as first_d, max(m.date) as last_d
            FROM message m JOIN handle h ON h.ROWID = m.handle_id
            WHERE m.is_from_me=0 AND {REACTION_EXCLUDE_SQL}
            GROUP BY h.id"""
    ).fetchall()
    chat_counts = {
        r["handle"]: r["chat_count"]
        for r in conn.execute(
            """SELECT h.id as handle, count(distinct chj.chat_id) as chat_count
               FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id
               GROUP BY h.id"""
        ).fetchall()
    }
    chat_per_handle = best_chat_per_handle(conn)

    now = datetime.now()
    items = []
    for r in counts:
        first_dt = apple_to_datetime(r["first_d"])
        last_dt = apple_to_datetime(r["last_d"])
        items.append(
            {
                "name": resolve_contact(r["handle"]) or r["handle"],
                "handle": r["handle"],
                "count": r["c"],
                "chat_count": chat_counts.get(r["handle"], 1),
                "chat_id": chat_per_handle.get(r["handle"]),
                "first_dt": first_dt,
                "last_dt": last_dt,
                "span_days": (last_dt - first_dt).days,
                "gap_days": (now - last_dt).days,
            }
        )
    return items


def monthly_counts(conn):
    rows = conn.execute(
        f"""SELECT strftime('%Y-%m', datetime(m.date/1000000000+978307200,'unixepoch','localtime')) as ym, count(*) c
            FROM message m WHERE {REACTION_EXCLUDE_SQL}
            GROUP BY ym ORDER BY ym"""
    ).fetchall()
    return [(r["ym"], r["c"]) for r in rows]


def render_stat_tile(label, value):
    return f"""<div class="stattile">
<div class="stat-value countup" data-count="{value}">0</div>
<div class="stat-label">{html.escape(label)}</div>
</div>"""


def render_leaderboard(items, max_count):
    rows = []
    for i, it in enumerate(items):
        pct = round(it["count"] / max_count * 100, 1) if max_count else 0
        clickable = ' class="lbrow clickable" onclick="location.href=\'/chat/{}\'"'.format(it["chat_id"]) if it.get("chat_id") else ' class="lbrow"'
        rows.append(
            f'<div{clickable} style="animation-delay:{i * 40}ms">'
            f'<div class="lbname">{avatar_html(it["name"], it["handle"])}{html.escape(it["name"])}</div>'
            f'<div class="lbtrack"><div class="lbbar" data-target="{pct}%"></div></div>'
            f'<div class="lbcount countup" data-count="{it["count"]}">0</div>'
            f"</div>"
        )
    return "".join(rows)


def render_contact_cards(items, sub_fn):
    out = []
    for i, it in enumerate(items):
        clickable = ' clickable" onclick="location.href=\'/chat/{}\'"'.format(it["chat_id"]) if it.get("chat_id") else '"'
        out.append(
            f'<div class="ccard{clickable} style="animation-delay:{i * 40}ms">'
            f'<div class="ccard-left">{avatar_html(it["name"], it["handle"])}'
            f'<div><div class="ccard-name">{html.escape(it["name"])}</div>'
            f'<div class="ccard-sub">{sub_fn(it)}</div></div></div>'
            f'<div class="ccard-count countup" data-count="{it["count"]}">0</div>'
            f"</div>"
        )
    return "".join(out)


def render_trend_chart(monthly):
    if not monthly:
        return "<p>No data.</p>"
    W, H = 900, 220
    pad_l, pad_r, pad_t, pad_b = 44, 10, 16, 24
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    n = len(monthly)
    col_w = plot_w / n
    max_c = max(c for _, c in monthly) or 1

    def x_at(i):
        return pad_l + col_w * (i + 0.5)

    def y_at(c):
        return pad_t + plot_h * (1 - c / max_c)

    points = [(x_at(i), y_at(c)) for i, (_, c) in enumerate(monthly)]
    line_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    y0 = pad_t + plot_h
    area_d = line_d + f" L {points[-1][0]:.1f} {y0:.1f} L {points[0][0]:.1f} {y0:.1f} Z"

    hit_cols = []
    dots = []
    x_labels = []
    seen_years = set()
    for i, (ym, c) in enumerate(monthly):
        x, y = points[i]
        year = ym[:4]
        if (ym[5:7] == "01" or i == 0) and year not in seen_years:
            seen_years.add(year)
            x_labels.append(f'<text x="{x:.1f}" y="{H - 4}" class="axislabel">{year}</text>')
        hit_cols.append(
            f'<rect class="hitcol" x="{pad_l + col_w * i:.1f}" y="{pad_t}" width="{col_w:.1f}" height="{plot_h}" '
            f'data-label="{ym}" data-count="{c}" data-cx="{x:.1f}" data-cy="{y:.1f}"></rect>'
        )
        dots.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="0"></circle>')

    svg = f"""<div class="trendwrap"><svg viewBox="0 0 {W} {H}" class="trendsvg" id="trendsvg" preserveAspectRatio="none">
<line x1="{pad_l}" y1="{y0:.1f}" x2="{W - pad_r}" y2="{y0:.1f}" class="axisline"></line>
<text x="2" y="{pad_t + 4}" class="axislabel">{max_c:,}</text>
<path d="{area_d}" class="areapath"></path>
<path d="{line_d}" class="linepath"></path>
{''.join(dots)}
{''.join(x_labels)}
{''.join(hit_cols)}
</svg><div class="trendtip" id="trendtip"></div></div>"""

    script = """
(function() {
  const svg = document.getElementById('trendsvg');
  if (!svg) return;
  const line = svg.querySelector('.linepath');
  const len = line.getTotalLength();
  line.style.strokeDasharray = len;
  line.style.strokeDashoffset = len;
  requestAnimationFrame(() => {
    line.style.transition = 'stroke-dashoffset 1.2s ease';
    line.style.strokeDashoffset = 0;
  });
  const area = svg.querySelector('.areapath');
  area.style.opacity = 0;
  requestAnimationFrame(() => {
    area.style.transition = 'opacity 1s ease 0.3s';
    area.style.opacity = 1;
  });
  const tip = document.getElementById('trendtip');
  const dots = svg.querySelectorAll('.pt');
  svg.querySelectorAll('.hitcol').forEach((col, i) => {
    col.addEventListener('mouseenter', () => {
      dots[i].setAttribute('r', 4);
      const box = svg.getBoundingClientRect();
      const scaleX = box.width / __W__, scaleY = box.height / __H__;
      tip.style.left = (parseFloat(col.dataset.cx) * scaleX) + 'px';
      tip.style.top = (parseFloat(col.dataset.cy) * scaleY) + 'px';
      tip.textContent = col.dataset.label + ': ' + parseInt(col.dataset.count, 10).toLocaleString() + ' messages';
      tip.style.opacity = 1;
    });
    col.addEventListener('mouseleave', () => {
      dots[i].setAttribute('r', 0);
      tip.style.opacity = 0;
    });
  });
})();
""".replace("__W__", str(W)).replace("__H__", str(H))

    return svg + f"<script>{script}</script>"


COUNTUP_SCRIPT = """
document.querySelectorAll('.countup').forEach(el => {
  const target = parseInt(el.dataset.count, 10) || 0;
  const dur = 900;
  const start = performance.now();
  function step(now) {
    const p = Math.min(1, (now - start) / dur);
    el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))).toLocaleString();
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
});
document.querySelectorAll('.lbbar').forEach(el => {
  requestAnimationFrame(() => requestAnimationFrame(() => { el.style.width = el.dataset.target; }));
});
"""


def render_stats():
    conn = get_conn()
    totals = conn.execute(
        f"""SELECT count(*) as msg_count, min(date) as first_d, max(date) as last_d
            FROM message m WHERE {REACTION_EXCLUDE_SQL}"""
    ).fetchone()
    chat_total = conn.execute("SELECT count(*) FROM chat").fetchone()[0]

    contacts = compute_contact_stats(conn)
    monthly = monthly_counts(conn)
    heatmap_html = build_heatmap_html(conn)
    conn.close()

    contacts.sort(key=lambda x: -x["count"])
    most_contacted = contacts[:15]
    max_count = most_contacted[0]["count"] if most_contacted else 1

    long_term = sorted(
        (c for c in contacts if c["span_days"] >= LONG_TERM_MIN_SPAN_DAYS and c["gap_days"] <= LONG_TERM_MAX_GAP_DAYS),
        key=lambda x: -x["count"],
    )[:10]

    fell_off = sorted(
        (c for c in contacts if c["count"] >= FELL_OFF_MIN_COUNT and c["gap_days"] >= FELL_OFF_MIN_GAP_DAYS),
        key=lambda x: -x["count"],
    )[:10]

    years = (totals["last_d"] - totals["first_d"]) / 1_000_000_000 / 86400 / 365 if totals["first_d"] else 0

    kpi_html = "".join(
        [
            render_stat_tile("Total messages", totals["msg_count"]),
            render_stat_tile("Conversations", chat_total),
            render_stat_tile("People texted", len(contacts)),
            render_stat_tile("Years of history", round(years, 1)),
        ]
    )

    long_term_html = render_contact_cards(
        long_term,
        lambda it: f"Friends since {it['first_dt'].year} &middot; {it['chat_count']} shared chat{'s' if it['chat_count'] != 1 else ''}",
    ) or '<p style="color:#999">No contacts cross the 1-year, active-in-90-days bar yet.</p>'

    fell_off_html = render_contact_cards(
        fell_off,
        lambda it: f"{it['gap_days']} days quiet &middot; last texted {it['last_dt'].strftime('%Y-%m-%d')}",
    ) or '<p style="color:#999">Nobody with 30+ messages has gone quiet for 180+ days. Good.</p>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Stats</title><style>{PAGE_CSS}</style></head><body>
<header><a href="/">&larr; All conversations</a> <b>Stats</b></header>
<main>
<div class="kpirow">{kpi_html}</div>

<h2 class="section-h">Activity, every day</h2>
{heatmap_html}

<h2 class="section-h">Messages per month</h2>
{render_trend_chart(monthly)}

<h2 class="section-h">Most contacted</h2>
<div class="panel">{render_leaderboard(most_contacted, max_count)}</div>

<h2 class="section-h">Long-term friends</h2>
<p class="section-sub">Talking for {LONG_TERM_MIN_SPAN_DAYS}+ days, still active in the last {LONG_TERM_MAX_GAP_DAYS} days.</p>
{long_term_html}

<h2 class="section-h">Fell out of touch</h2>
<p class="section-sub">{FELL_OFF_MIN_COUNT}+ lifetime messages, quiet for {FELL_OFF_MIN_GAP_DAYS}+ days.</p>
{fell_off_html}
</main>
<script>{COUNTUP_SCRIPT}</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        qs = parse_qs(parsed.query)

        if not parts:
            sort = qs.get("sort", ["recent"])[0]
            self._send_html(render_chat_list(sort))
        elif parts[0] == "stats" and len(parts) == 1:
            self._send_html(render_stats())
        elif parts[0] == "chat" and len(parts) == 2:
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            date_str = qs.get("date", [None])[0]
            around = qs.get("around", [None])[0]
            around_id = int(around) if around and around.isdigit() else None
            out = render_chat(chat_id, date_str, around_id)
            self._send_html(out) if out is not None else self._send_error(404)
        elif parts[0] == "chat" and len(parts) == 3 and parts[2] == "more":
            self._serve_more(parts[1], qs)
        elif parts[0] == "chat" and len(parts) == 3 and parts[2] == "media":
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            out = render_media(chat_id)
            self._send_html(out) if out is not None else self._send_error(404)
        elif parts[0] == "chat" and len(parts) == 3 and parts[2] == "search":
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            query = qs.get("q", [None])[0]
            out = render_search(chat_id, query)
            self._send_html(out) if out is not None else self._send_error(404)
        elif parts[0] == "attachment" and len(parts) == 2:
            self._serve_attachment(parts[1])
        elif parts[0] == "thumb" and len(parts) == 2:
            self._serve_thumb(parts[1])
        elif parts[0] == "avatar" and len(parts) == 2:
            self._serve_avatar(parts[1])
        else:
            self._send_error(404)

    def _serve_more(self, chat_id, qs):
        try:
            chat_id = int(chat_id)
        except ValueError:
            self._send_error(404)
            return

        def take(key):
            raw = qs.get(key, [None])[0]
            if raw in (None, ""):
                return None
            try:
                return int(raw)
            except ValueError:
                return None

        after_date, after_id = take("after_date"), take("after_id")
        before_date, before_id = take("before_date"), take("before_id")
        conn = get_conn()
        prev_day = prev_sender = next_day = next_sender = None
        strip_group_start = False

        if before_date is not None and before_id is not None:
            rows = fetch_messages(conn, chat_id, before=(before_date, before_id), limit=PAGE_SIZE)
            next_day, next_sender = sender_context(conn, before_id)
            has_more_older = len(rows) == PAGE_SIZE
            has_more_newer = True
            if rows and next_sender is not None:
                last = rows[-1]
                strip_group_start = (
                    apple_date(last["date"])[:10] == next_day
                    and (last["is_from_me"], last["handle"]) == next_sender
                )
        elif after_date is not None and after_id is not None:
            rows = fetch_messages(conn, chat_id, after=(after_date, after_id), limit=PAGE_SIZE)
            prev_day, prev_sender = sender_context(conn, after_id)
            has_more_newer = len(rows) == PAGE_SIZE
            has_more_older = True
        else:
            conn.close()
            self._send_error(404)
            return

        ids = [r["id"] for r in rows]
        att_by_msg = load_attachments(conn, ids)
        reactions_by_guid = load_reactions(conn, chat_id, [r["guid"] for r in rows])
        blocks_html = render_message_blocks(
            rows,
            att_by_msg,
            reactions_by_guid,
            prev_day=prev_day,
            prev_sender=prev_sender,
            next_day=next_day,
            next_sender=next_sender,
        )
        conn.close()
        self._send_json(
            {
                "html": blocks_html,
                "first_id": rows[0]["id"] if rows else 0,
                "first_date": str(rows[0]["date"]) if rows else "0",
                "last_id": rows[-1]["id"] if rows else 0,
                "last_date": str(rows[-1]["date"]) if rows else "0",
                "has_more_older": has_more_older,
                "has_more_newer": has_more_newer,
                "strip_group_start": strip_group_start,
            }
        )

    def _send_html(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code):
        self.send_response(code)
        self.end_headers()

    def _send_file(self, path, mime):
        try:
            size = os.path.getsize(path)
            handle = open(path, "rb")
        except OSError:
            self._send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", CACHE_CONTROL)
        self.end_headers()
        try:
            shutil.copyfileobj(handle, self.wfile, 65536)
        finally:
            handle.close()

    def _attachment_path(self, att_id):
        try:
            att_id = int(att_id)
        except ValueError:
            return None
        conn = get_conn()
        row = conn.execute(
            "SELECT filename, mime_type FROM attachment WHERE ROWID=?", (att_id,)
        ).fetchone()
        conn.close()
        if not row or not row["filename"]:
            return None
        path = os.path.expanduser(row["filename"])
        mime = row["mime_type"] or mimetypes.guess_type(path)[0] or "application/octet-stream"
        return path, mime

    def _serve_avatar(self, avatar_id):
        entry = AVATAR_INDEX.get(avatar_id)
        if not entry:
            self._send_error(404)
            return
        mime, data = entry
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", CACHE_CONTROL)
        self.end_headers()
        self.wfile.write(data)

    def _serve_thumb(self, att_id):
        info = self._attachment_path(att_id)
        if not info:
            self._send_error(404)
            return
        path, mime = info
        if not os.path.exists(path):
            self._send_error(404)
            return
        thumb = make_thumb(path) if mime.startswith("image/") or is_heic(path, mime) else None
        if thumb:
            self._send_file(thumb, "image/jpeg")
            return
        if is_heic(path, mime):
            converted = convert_heic(path)
            if converted:
                self._send_file(converted, "image/jpeg")
                return
        self._send_file(path, mime)

    def _serve_attachment(self, att_id):
        info = self._attachment_path(att_id)
        if not info:
            self._send_error(404)
            return
        path, mime = info
        if is_heic(path, mime):
            converted = convert_heic(path)
            if converted:
                path, mime = converted, "image/jpeg"
        if not os.path.exists(path):
            self._send_error(404)
            return
        self._send_file(path, mime)


if __name__ == "__main__":
    print(f"Loaded {len(CONTACTS)} contact lookup entries")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving on http://127.0.0.1:{PORT}/")
    server.serve_forever()
