"""HTML for every page. Styles and shared behavior live in static/."""

import html
import json
import math
import mimetypes
import os
from datetime import datetime, timedelta
from urllib.parse import quote

from config import DB_PATH, PAGE_SIZE, SCRIPT_DIR, SEARCH_LIMIT, START_NEWEST, load_prefs
from contacts import (
    avatar_html,
    chat_label,
    load_chat_labels,
    load_participants,
    resolve_contact,
)
from db import (
    REACTION_EXCLUDE_SQL,
    REACTION_LABELS,
    apple_date,
    chat_filter,
    date_to_apple_ns,
    fetch_messages,
    fetch_messages_around,
    get_conn,
    has_neighbor,
    live_db_error,
    merged_chat_ids,
    load_attachments,
    load_reactions,
    message_text,
)
from graph import load_circle_graph
from search import (
    index_error,
    is_index_ready,
    kick_search_index,
    matching_conversations,
    search_messages,
    snippet_html,
)
from stats import (
    DRIFT_WINDOW_MONTHS,
    ERA_MIN_MSGS,
    FELL_OFF_MIN_COUNT,
    FELL_OFF_MIN_GAP_DAYS,
    LONG_TERM_MAX_GAP_DAYS,
    LONG_TERM_MIN_SPAN_DAYS,
    SESSION_GAP_SECONDS,
    compute_contact_stats,
    dm_streams,
    effective_people,
    initiation,
    monthly_by_person,
    monthly_counts,
    people_drift,
    people_eras,
    people_over_time,
    reaction_stats,
    reply_latency,
    year_owners,
)
from voice import load_voice


def is_heic(path, mime=""):
    return mime in ("image/heic", "image/heif") or (path or "").lower().endswith((".heic", ".heif"))


def format_day_label(day_str):
    d = datetime.strptime(day_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    return d.strftime("%A, %B %-d, %Y")


def format_when(ns):
    s = apple_date(ns)
    if not s:
        return ""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
    d = dt.date()
    today = datetime.now().date()
    if d == today:
        t = dt.strftime("%-I:%M %p")
        return t[:-3] + t[-2:].lower()
    if d == today - timedelta(days=1):
        return "Yesterday"
    if 0 < (today - d).days < 7:
        return dt.strftime("%a")
    if d.year == today.year:
        return dt.strftime("%b %-d")
    return dt.strftime("%b %-d, %Y")


def format_time(ns):
    s = apple_date(ns)
    if not s:
        return ""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
    t = dt.strftime("%-I:%M %p")
    return t[:-3] + t[-2:].lower()


# iMessage puts a centered timestamp in the thread after about an hour of
# silence. Same rule here: it is display-only, not stored on the message.
TIME_BREAK_NS = 60 * 60 * 1_000_000_000


def is_time_break(prev_ns, curr_ns):
    return prev_ns is not None and curr_ns is not None and curr_ns - prev_ns >= TIME_BREAK_NS


def _time_sep(ns):
    return f'<div class="timeSep">{format_time(ns)}</div>'


def asset_url(name):
    path = os.path.join(SCRIPT_DIR, "static", name)
    try:
        v = int(os.path.getmtime(path))
    except OSError:
        v = 0
    return f"/static/{name}?v={v}"


NAV = (
    ("/", "Chats", "chats"),
    ("/media", "Photos", "photos"),
    ("/search", "Search", "search"),
    ("/stats", "Stats", "stats"),
    ("/circles", "Circles", "circles"),
    ("/twin", "Twin", "twin"),
)


def nav_html(active, twin_busy=False):
    links = []
    for href, label, key in NAV:
        classes = "nav-link"
        if key == active:
            classes += " is-active"
        if key == "twin" and twin_busy:
            classes += " is-training"
        dot = '<span class="nav-dot" aria-hidden="true"></span>' if key == "twin" else ""
        links.append(f'<a class="{classes}" href="{href}">{label}{dot}</a>')
    return f'<nav class="nav">{"".join(links)}</nav>'


# Runs in the head so the theme and the tile size are set before the first
# paint. app.js is deferred and lands too late to prevent a flash.
HEAD_BOOT = (
    "<script>(function(){var t=localStorage.getItem('theme');"
    "if(t!=='light'&&t!=='dark')t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';"
    "document.documentElement.dataset.theme=t;"
    "document.documentElement.style.colorScheme=t;"
    "var s=localStorage.getItem('mediasize');"
    "document.documentElement.dataset.mediasize=(s==='s'||s==='l')?s:'m'})();</script>"
)

THEME_TOGGLE = """<button type="button" class="theme-toggle" id="themeToggle" aria-label="Switch to dark mode" aria-pressed="false">
<span class="theme-icon theme-icon-moon" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 14.5A8.5 8.5 0 1 1 9.5 3 7 7 0 0 0 21 14.5z"/></svg></span>
<span class="theme-icon theme-icon-sun" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg></span>
</button>"""


def search_form_html(action, query="", placeholder="Search messages…", wide=False):
    cls = "search-form search-form-wide" if wide else "search-form"
    return (
        f'<form class="{cls}" method="get" action="{html.escape(action)}">'
        f'<input class="search-input" type="search" name="q" value="{html.escape(query or "")}" '
        f'placeholder="{html.escape(placeholder)}" autocomplete="off">'
        f'<button class="btn btn-primary" type="submit">Search</button></form>'
    )


def db_banner_html():
    """Red bar on every page while macOS blocks the read of chat.db. Without it
    the app looks healthy, because search and stats still serve cached data."""
    detail = live_db_error()
    if detail is None:
        return ""
    commands = (
        f"cd {SCRIPT_DIR}\n"
        "source .venv/bin/activate\n"
        "python3 app.py"
    )
    return f"""<div class="alertbar" role="alert">
<p class="alertbar-head">This Mac blocks access to your Messages database.</p>
<p class="alertbar-sub">The app reads <code>{html.escape(DB_PATH)}</code>. Until macOS gives it access, every page shows old data or no data.</p>
<details class="alertbar-help">
<summary>How to get this app to work</summary>
<ol>
<li>Open System Settings → Privacy &amp; Security → Full Disk Access.</li>
<li>Turn on the switch for the app that runs the server: Terminal, iTerm, or Cursor.</li>
<li>Quit that app with Command-Q, then open it again.</li>
<li>Run these commands in the terminal:
<pre>{html.escape(commands)}</pre></li>
<li>Reload this page.</li>
</ol>
<p class="alertbar-detail">{html.escape(detail)}</p>
</details>
</div>"""


def twin_chip_html(activity):
    if not activity.get("busy"):
        return ""
    from twin.job import activity_label

    return (
        f'<a class="twin-chip" id="twinChip" href="/twin#model">'
        f'{html.escape(activity_label(activity))}</a>'
    )


def page(title, body, *, active="chats", header_left=None, header_right="", body_class="", scripts="", chat_id=None):
    try:
        from twin.job import snapshot as twin_pulse

        activity = twin_pulse(brief=True)
    except Exception:
        activity = {"busy": False}
    left = header_left if header_left is not None else (
        f'<a class="brand" href="/">Messages</a>{nav_html(active, activity.get("busy"))}'
    )
    data_chat = f' data-chat-id="{chat_id}"' if chat_id else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{HEAD_BOOT}
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{asset_url("app.css")}">
</head>
<body class="{html.escape(body_class)}"{data_chat}>
{db_banner_html()}
<header class="topbar">
<div class="topbar-left">{left}</div>
<div class="topbar-right">{twin_chip_html(activity)}{header_right}</div>
{THEME_TOGGLE}
</header>
{body}
<script src="{asset_url("app.js")}" defer></script>
{scripts}
</body>
</html>"""


TAPBACK_BADGE_MAX = 3


def build_tapbacks_html(reactions):
    """The badge that sits on the corner of the bubble. It shows the distinct
    emoji only. app.js reads data-detail to list who reacted."""
    distinct = list(dict.fromkeys(emoji for emoji, _verb, _who in reactions))
    shown = distinct[:TAPBACK_BADGE_MAX]
    chips = "".join(f'<span class="tapback-emoji">{html.escape(emoji)}</span>' for emoji in shown)
    if len(reactions) > len(shown):
        chips += f'<span class="tapback-count">{len(reactions)}</span>'
    detail = [{"emoji": emoji, "verb": verb, "who": who} for emoji, verb, who in reactions]
    label = ", ".join(f"{verb} by {who}" for _emoji, verb, who in reactions)
    return (
        f'<button type="button" class="tapbacks" aria-haspopup="dialog" '
        f'aria-label="{html.escape(label)}" '
        f'data-detail="{html.escape(json.dumps(detail, ensure_ascii=False))}">{chips}</button>'
    )


def render_message_blocks(
    rows,
    att_by_msg,
    reactions_by_guid=None,
    highlight_id=None,
    prev_day=None,
    prev_sender=None,
    prev_date=None,
    next_day=None,
    next_sender=None,
    next_date=None,
):
    reactions_by_guid = reactions_by_guid or {}
    blocks = []
    n = len(rows)
    open_day = None

    def close_day():
        nonlocal open_day
        if open_day is not None:
            blocks.append("</div>")
            open_day = None

    def ensure_day(day):
        nonlocal open_day
        if open_day == day:
            return
        close_day()
        blocks.append(f'<div class="day" data-day="{day}">')
        open_day = day
        # Skip the pill when this day already has one in the loaded thread
        # (older page) or in the previous page (newer page).
        if day != prev_day and (next_day is None or day != next_day):
            blocks.append(f'<div class="dateSep">{format_day_label(day)}</div>')

    for idx, r in enumerate(rows):
        day = apple_date(r["date"])[:10]
        sender_key = (r["is_from_me"], r["handle"])
        ensure_day(day)

        if day != prev_day:
            prev_day = day
            prev_sender = None
        elif is_time_break(prev_date, r["date"]):
            blocks.append(_time_sep(r["date"]))
            prev_sender = None

        next_row = rows[idx + 1] if idx + 1 < n else None
        if next_row is not None:
            next_same_group = (
                (next_row["is_from_me"], next_row["handle"]) == sender_key
                and apple_date(next_row["date"])[:10] == day
                and not is_time_break(r["date"], next_row["date"])
            )
        elif next_sender is not None:
            next_same_group = (
                next_sender == sender_key
                and next_day == day
                and not is_time_break(r["date"], next_date)
            )
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
            if mime.startswith("image/") or is_heic(a["filename"] or "", mime):
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
        tapback_cls = ""
        if reactions:
            body += build_tapbacks_html(reactions)
            tapback_cls = " has-tapback"

        parts.append(f'<div class="msgbody">{body}</div>')
        if is_last_in_group:
            parts.append(f'<div class="ts">{format_time(r["date"])}</div>')
        blocks.append(
            f'<div class="row {who}{group_cls}{highlight_cls}{tapback_cls}" id="msg-{r["id"]}">{"".join(parts)}</div>'
        )
        prev_sender = sender_key
        prev_date = r["date"]
    if (
        rows
        and next_day is not None
        and apple_date(rows[-1]["date"])[:10] == next_day
        and is_time_break(rows[-1]["date"], next_date)
    ):
        blocks.append(_time_sep(next_date))
    close_day()
    return "".join(blocks)


HEAT_MODES = (("a", "All", "Every message"), ("r", "Received", "Messages you received"), ("s", "Sent", "Messages you sent"))


def heat_mode_seg_html():
    btns = "".join(
        f'<label class="seg-btn" title="{title}">'
        f'<input type="radio" name="heatmode" value="{key}"{" checked" if key == "a" else ""}> {label}</label>'
        for key, label, title in HEAT_MODES
    )
    return f'<div class="seg seg-sm" id="heatMode" role="radiogroup" aria-label="Heatmap mode">{btns}</div>'


def build_heatmap_html(conn, chat_id=None):
    cols = """date(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as day,
              count(*) c, sum(m.is_from_me = 0) recv, sum(m.is_from_me = 1) sent"""
    if chat_id is not None:
        chat_sql, chat_params = chat_filter(conn, chat_id)
        counts = conn.execute(
            f"""SELECT {cols}
               FROM message m JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
               WHERE {chat_sql} AND {REACTION_EXCLUDE_SQL}
               GROUP BY day""",
            chat_params,
        ).fetchall()
    else:
        counts = conn.execute(
            f"""SELECT {cols}
               FROM message m WHERE {REACTION_EXCLUDE_SQL}
               GROUP BY day"""
        ).fetchall()
    if not counts:
        return ""

    day_counts = {r["day"]: (r["c"], r["recv"] or 0, r["sent"] or 0) for r in counts}
    start = datetime.strptime(min(day_counts), "%Y-%m-%d").date()
    end = datetime.strptime(max(day_counts), "%Y-%m-%d").date()
    start_aligned = start - timedelta(days=(start.isoweekday() % 7))
    maxes = [max(v[i] for v in day_counts.values()) or 1 for i in range(3)]
    max_count = maxes[0]

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
            c, recv, sent = day_counts.get(day_str, (0, 0, 0))
            level = bucket(c)
            title = f"{day_str}: {c} message{'s' if c != 1 else ''}"
            clickable = c and chat_id is not None
            onclick = f' onclick="location.href=\'/chat/{chat_id}?date={day_str}\'"' if clickable else ""
            cls = "hcell clickable" if clickable else "hcell"
            cell_html_parts.append(
                f'<div class="{cls} heat-{level}" data-day="{day_str}" title="{title}"'
                f' data-a="{c}" data-r="{recv}" data-s="{sent}"{onclick}></div>'
            )

    legend = "".join(f'<div class="hcell heat-{i}"></div>' for i in range(5))
    return f"""<div class="heatmap-card" data-max="{','.join(str(m) for m in maxes)}">
<div class="heatmap-top">{heat_mode_seg_html()}</div>
<div class="heatmap">
<div class="heatmap-dow"><span></span><span></span><span>M</span><span></span><span>W</span><span></span><span>F</span><span></span></div>
<div class="heatmap-wrap"><div class="heatmap-scroll">
<div class="heatmap-months">{month_html}</div>
<div class="heatmap-grid">{''.join(cell_html_parts)}</div>
</div></div></div>
<div class="heatmap-legend"><span>Less</span>{legend}<span>More</span></div>
</div>"""


SORT_OPTIONS = {
    "recent": "Most recent",
    "count": "Most messages",
    "name": "Name A–Z",
    "oldest": "Oldest first",
}


def merge_split_chats(items, groups):
    """Collapse the rows that the chat view already reads as one thread. The
    biggest chat of a group carries the row, so its link opens the same merged
    history that the counts describe."""
    by_group = {}
    for it in items:
        by_group.setdefault(groups[it["id"]], []).append(it)
    merged = []
    for group in by_group.values():
        lead = max(group, key=lambda x: x["count"])
        if len(group) > 1:
            lead = dict(lead)
            lead["count"] = sum(x["count"] for x in group)
            lead["first"] = min(x["first"] for x in group)
            lead["last"] = max(x["last"] for x in group)
            lead["known"] = any(x["known"] for x in group)
        merged.append(lead)
    return merged


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
    groups = {r["id"]: merged_chat_ids(conn, r["id"]) for r in rows}
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
    items = merge_split_chats(items, groups)

    if sort == "name":
        items.sort(key=lambda x: x["name"].lower())
    elif sort == "count":
        items.sort(key=lambda x: -x["count"])
    elif sort == "oldest":
        items.sort(key=lambda x: x["first"])
    else:
        sort = "recent"
        items.sort(key=lambda x: -x["last"])

    row_html = []
    for it in items:
        row_html.append(
            f'<a class="chat-row" href="/chat/{it["id"]}" '
            f'data-search="{html.escape(it["name"].lower())}" data-known="{"1" if it["known"] else "0"}">'
            f'{avatar_html(it["name"], it["identifier"])}'
            f'<div class="chat-body"><div class="chat-top">'
            f'<span class="chat-name">{html.escape(it["name"])}</span>'
            f'<span class="chat-when">{format_when(it["last"])}</span></div>'
            f'<div class="chat-sub">{it["count"]:,} messages</div></div></a>'
        )

    opts_html = "".join(
        f'<option value="{k}"{" selected" if k == sort else ""}>{v}</option>'
        for k, v in SORT_OPTIONS.items()
    )
    start = load_prefs()["start"]
    oldest_on = " checked" if start != START_NEWEST else ""
    newest_on = " checked" if start == START_NEWEST else ""

    body = f"""<main class="page">
<div class="toolbar">
<div class="toolbar-grow"><input class="field" id="filter" placeholder="Find a conversation" autocomplete="off"></div>
<label class="check"><input type="checkbox" id="knownOnly"> Known contacts</label>
<select class="select" id="sortSelect">{opts_html}</select>
</div>
<div class="toolbar">
<span class="prefs-label">Open at</span>
<div class="seg" role="radiogroup" aria-label="Open chats and photos">
<label class="seg-btn"><input type="radio" name="start" value="oldest"{oldest_on}> Oldest</label>
<label class="seg-btn"><input type="radio" name="start" value="newest"{newest_on}> Newest</label>
</div>
</div>
<div class="chat-list" id="rows">{''.join(row_html)}</div>
</main>"""
    return page(
        "Messages",
        body,
        active="chats",
        header_right=search_form_html("/search", placeholder="Search all messages…", wide=True),
    )


def render_chat(chat_id, date_str=None, around_id=None):
    conn = get_conn()
    chat = conn.execute(
        "SELECT ROWID as id, chat_identifier, display_name FROM chat WHERE ROWID=?", (chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        return None

    chat_sql, chat_params = chat_filter(conn, chat_id)
    bounds = conn.execute(
        f"""SELECT min(m.date) as lo, max(m.date) as hi FROM message m
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID WHERE {chat_sql}""",
        chat_params,
    ).fetchone()
    media_count = conn.execute(
        f"""SELECT count(*) FROM message_attachment_join maj
           JOIN chat_message_join cmj ON cmj.message_id = maj.message_id
           JOIN attachment att ON att.ROWID = maj.attachment_id
           WHERE {chat_sql} AND att.filename NOT LIKE '%.pluginPayloadAttachment'""",
        chat_params,
    ).fetchone()[0]

    start_newest = load_prefs()["start"] == START_NEWEST
    jump_to_end = False
    if around_id:
        rows = fetch_messages_around(conn, chat_id, around_id)
    else:
        start_ns = None
        if date_str:
            try:
                start_ns = date_to_apple_ns(date_str)
            except ValueError:
                start_ns = None
        if start_ns is None and start_newest:
            rows = fetch_messages(conn, chat_id, from_end=True, limit=PAGE_SIZE)
            jump_to_end = True
        else:
            rows = fetch_messages(conn, chat_id, start_ns=start_ns, limit=PAGE_SIZE)

    ids = [r["id"] for r in rows]
    att_by_msg = load_attachments(conn, ids)
    reactions_by_guid = load_reactions(conn, chat_id, [r["guid"] for r in rows])
    blocks_html = render_message_blocks(rows, att_by_msg, reactions_by_guid, highlight_id=around_id)
    heatmap_html = build_heatmap_html(conn, chat_id)

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
    elif jump_to_end:
        cur_date = max_date
    else:
        cur_date = min_date

    around_js = json.dumps(around_id)
    top_label = "" if has_older else "Beginning of conversation"
    bottom_label = "Loading more…" if has_newer else "End of conversation"
    rail_html = ""
    if min_date and max_date:
        rail_html = f"""<div class="media-rail" id="chatRail" data-chat-id="{chat_id}" data-min="{min_date}" data-max="{max_date}" data-current="{cur_date}">
<div class="media-rail-track" id="chatRailTrack"><div class="media-rail-dot" id="chatRailDot"></div></div>
<div class="media-rail-label" id="chatRailLabel"></div>
</div>"""

    body = f"""<main class="page">
{heatmap_html}
<div id="sentinel-top">{top_label}</div>
<div class="bubblewrap" id="messages">{blocks_html}</div>
<div id="sentinel">{bottom_label}</div>
</main>
{rail_html}"""

    header_left = (
        f'<a class="back" href="/">← Chats</a>'
        f'<div class="chat-title">{avatar_html(title, chat["chat_identifier"])}'
        f'<b>{html.escape(title)}</b></div>'
    )
    header_right = (
        f'<a class="btn btn-ghost" href="/chat/{chat_id}/media">Photos · {media_count:,}</a>'
        f'{search_form_html(f"/chat/{chat_id}/search", placeholder="Search this chat…")}'
        f'<input type="date" id="datepicker" value="{cur_date}" min="{min_date}" max="{max_date}" data-chat-id="{chat_id}">'
    )
    scripts = f"""<script>
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

function daySep(el) {{
  const first = el.firstElementChild;
  return first && first.classList.contains('dateSep') ? first : null;
}}

function mergeAdjacentDays() {{
  const days = Array.from(wrap.children).filter((el) => el.classList.contains('day'));
  for (let i = 1; i < days.length; i++) {{
    const prev = days[i - 1];
    const cur = days[i];
    if (prev.dataset.day !== cur.dataset.day) continue;
    const sep = daySep(cur) || daySep(prev);
    const frag = document.createDocumentFragment();
    Array.from(prev.childNodes).forEach((n) => {{
      if (n.nodeType === 1 && n.classList.contains('dateSep')) return;
      frag.appendChild(n);
    }});
    if (sep) {{
      if (sep.parentNode !== cur) cur.prepend(sep);
      sep.after(frag);
    }} else {{
      cur.prepend(frag);
    }}
    prev.remove();
  }}
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
    mergeAdjacentDays();
    lastDate = data.last_date;
    lastId = data.last_id;
  }}
  hasNewer = data.has_more_newer;
  loading = false;
  sentinel.textContent = hasNewer ? 'Loading more…' : 'End of conversation';
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
    mergeAdjacentDays();
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

function jumpToEnd() {{
  window.scrollTo(0, document.documentElement.scrollHeight);
}}

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
  if ({json.dumps(jump_to_end)}) {{
    jumpToEnd();
    window.addEventListener('load', jumpToEnd);
  }}
  armObservers();
}}
</script>"""
    return page(
        title,
        body,
        header_left=header_left,
        header_right=header_right,
        body_class="chatpage",
        scripts=scripts,
        chat_id=chat_id,
    )


def month_label(ym):
    return datetime.strptime(ym, "%Y-%m").strftime("%B %Y")


def short_month(ym):
    return datetime.strptime(ym, "%Y-%m").strftime("%b %Y")


PLAY_ICON = (
    '<svg width="8" height="9" viewBox="0 0 8 9" fill="currentColor" aria-hidden="true">'
    '<path d="M0 0l8 4.5L0 9z"/></svg>'
)


def media_tile_html(a, chat_id, chat_name=None, visual_only=False):
    mime = a["mime_type"] or mimetypes.guess_type(a["filename"] or "")[0] or ""
    is_image = mime.startswith("image/") or is_heic(a["filename"] or "", mime)
    is_video = mime.startswith("video/")
    if visual_only and not is_image and not is_video:
        return None, None
    link = f'/chat/{chat_id}?around={a["msg_id"]}'
    when = apple_date(a["date"])
    title = f"{chat_name} · {when}" if chat_name else when
    if is_image:
        inner = f'<img src="/thumb/{a["att_id"]}" data-full-src="/attachment/{a["att_id"]}" loading="lazy">'
    elif is_video:
        inner = (
            f'<video src="/attachment/{a["att_id"]}" preload="metadata" muted></video>'
            f'<span class="tile-badge">{PLAY_ICON}<b class="tile-dur"></b></span>'
        )
    else:
        fname = html.escape(os.path.basename(a["filename"] or "file"))
        inner = f'<div class="filetile">{fname}</div>'
    caption = (
        f'<span class="tile-chat">{html.escape(chat_name)}</span>' if chat_name else ""
    )
    tile = (
        f'<a class="tile" href="{link}" title="{html.escape(title)}" '
        f'data-msg-id="{a["msg_id"]}" data-chat-id="{chat_id}">{inner}{caption}</a>'
    )
    return apple_date(a["date"])[:7], tile


MEDIA_SIZES = (("s", "S", "Small tiles"), ("m", "M", "Medium tiles"), ("l", "L", "Large tiles"))


def media_size_seg_html():
    """The checked radio is set by app.js from localStorage."""
    btns = "".join(
        f'<label class="seg-btn" title="{title}">'
        f'<input type="radio" name="mediasize" value="{key}"> {label}</label>'
        for key, label, title in MEDIA_SIZES
    )
    return f'<div class="seg seg-sm" id="mediaSize" role="radiogroup" aria-label="Tile size">{btns}</div>'


def media_start_script():
    if load_prefs()["start"] != START_NEWEST:
        return ""
    return (
        "<script>function goEnd(){window.scrollTo(0,document.documentElement.scrollHeight)}"
        "goEnd();window.addEventListener('load',goEnd);</script>"
    )


def media_sections_html(items, empty_msg):
    """items is a list of (ym, tile_html) in chronological order."""
    sections = []
    cur_ym = None
    tiles = []
    for ym, tile in items:
        if ym != cur_ym:
            if tiles:
                sections.append((cur_ym, tiles))
            cur_ym = ym
            tiles = []
        tiles.append(tile)
    if tiles:
        sections.append((cur_ym, tiles))
    if not sections:
        return f'<p class="empty">{empty_msg}</p>', ""
    section_html = "".join(
        f'<section class="media-month" data-year="{ym[:4]}" data-label="{month_label(ym)}">'
        f'<h3 class="media-month-h">{month_label(ym)}</h3>'
        f'<div class="mediagrid">{"".join(tile_htmls)}</div></section>'
        for ym, tile_htmls in sections
    )
    rail_html = """<div class="media-rail" id="mediaRail">
<div class="media-rail-track" id="mediaRailTrack"><div class="media-rail-dot" id="mediaRailDot"></div></div>
<div class="media-rail-label" id="mediaRailLabel"></div>
</div>"""
    return section_html, rail_html


def render_media(chat_id):
    conn = get_conn()
    chat = conn.execute(
        "SELECT ROWID as id, chat_identifier, display_name FROM chat WHERE ROWID=?", (chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        return None

    chat_sql, chat_params = chat_filter(conn, chat_id)
    media = conn.execute(
        f"""SELECT att.ROWID as att_id, att.mime_type, att.filename, m.ROWID as msg_id, m.date
           FROM message_attachment_join maj
           JOIN attachment att ON att.ROWID = maj.attachment_id
           JOIN message m ON m.ROWID = maj.message_id
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           WHERE {chat_sql} AND att.filename NOT LIKE '%.pluginPayloadAttachment'
           ORDER BY m.date ASC, m.ROWID ASC""",
        chat_params,
    ).fetchall()

    participants = None
    if not chat["display_name"] and not resolve_contact(chat["chat_identifier"]):
        participants = load_participants(conn, [chat_id]).get(chat_id)
    title = chat_label(chat["display_name"], chat["chat_identifier"], participants)
    conn.close()

    items = []
    for a in media:
        ym, tile = media_tile_html(a, chat_id)
        if tile:
            items.append((ym, tile))
    section_html, rail_html = media_sections_html(items, "No photos or files in this conversation.")

    header_left = (
        f'<a class="back" href="/chat/{chat_id}">← {html.escape(title)}</a>'
        f'<b>Photos</b>'
    )
    return page(
        f"Photos: {title}",
        f'<main class="page">{section_html}</main>{rail_html}',
        header_left=header_left,
        header_right=(
            f'{media_size_seg_html()}<a class="btn btn-ghost" href="/media">All photos</a>'
            f'<span class="muted">{len(items):,} items</span>'
        ),
        body_class="mediapage",
        scripts=media_start_script(),
        chat_id=chat_id,
    )


def render_all_media():
    conn = get_conn()
    media = conn.execute(
        """SELECT att.ROWID as att_id, att.mime_type, att.filename, m.ROWID as msg_id, m.date,
                  cmj.chat_id as chat_id
           FROM message_attachment_join maj
           JOIN attachment att ON att.ROWID = maj.attachment_id
           JOIN message m ON m.ROWID = maj.message_id
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           WHERE att.filename NOT LIKE '%.pluginPayloadAttachment'
             AND (
               ifnull(att.mime_type, '') LIKE 'image/%'
               OR ifnull(att.mime_type, '') LIKE 'video/%'
               OR lower(ifnull(att.filename, '')) LIKE '%.heic'
               OR lower(ifnull(att.filename, '')) LIKE '%.heif'
               OR lower(ifnull(att.filename, '')) LIKE '%.mov'
               OR lower(ifnull(att.filename, '')) LIKE '%.mp4'
             )
           ORDER BY m.date ASC, m.ROWID ASC"""
    ).fetchall()
    labels = load_chat_labels(conn, [a["chat_id"] for a in media])
    conn.close()

    seen = set()
    items = []
    for a in media:
        if a["att_id"] in seen:
            continue
        seen.add(a["att_id"])
        ym, tile = media_tile_html(
            a, a["chat_id"], chat_name=labels.get(a["chat_id"]), visual_only=True
        )
        if tile:
            items.append((ym, tile))

    section_html, rail_html = media_sections_html(items, "No photos or videos yet.")
    return page(
        "Photos",
        f'<main class="page">{section_html}</main>{rail_html}',
        active="photos",
        header_right=f'{media_size_seg_html()}<span class="muted">{len(items):,} items</span>',
        body_class="mediapage",
        scripts=media_start_script(),
    )


def render_indexing_page(query, chat_id=None):
    action = f"/chat/{chat_id}/search" if chat_id else "/search"
    err = f'<p class="empty">{html.escape(index_error())}</p>' if index_error() else ""
    body = f"""<main class="page">
<h1 class="section-h">Building search index</h1>
<p class="section-sub">One-time pass over your messages. This page refreshes on its own.</p>
{err}
</main>"""
    extra_head = '<meta http-equiv="refresh" content="1">'
    # page() does not expose extra head tags; inject via a leading comment-free prefix.
    html_out = page(
        "Search",
        body,
        active="search",
        header_left=(
            f'<a class="back" href="/chat/{chat_id}">← Conversation</a>' if chat_id else None
        ),
        header_right=search_form_html(action, query, "Search messages…", wide=True),
    )
    return html_out.replace("<head>", "<head>\n" + extra_head, 1)


def render_search(query, chat_id=None):
    chat_title = None
    chat_ids = None
    if chat_id is not None:
        conn = get_conn()
        chat = conn.execute(
            "SELECT ROWID as id, chat_identifier, display_name FROM chat WHERE ROWID=?", (chat_id,)
        ).fetchone()
        if not chat:
            conn.close()
            return None
        participants = None
        if not chat["display_name"] and not resolve_contact(chat["chat_identifier"]):
            participants = load_participants(conn, [chat_id]).get(chat_id)
        chat_title = chat_label(chat["display_name"], chat["chat_identifier"], participants)
        chat_ids = merged_chat_ids(conn, chat_id)
        conn.close()

    q = (query or "").strip()
    if q and not is_index_ready():
        kick_search_index()
        if not is_index_ready():
            return render_indexing_page(query, chat_id)

    rows, phrases, terms, broadened = search_messages(q, chat_ids=chat_ids) if q else ([], [], [], False)
    truncated = len(rows) > SEARCH_LIMIT
    rows = rows[:SEARCH_LIMIT]

    labels = {}
    if rows:
        conn = get_conn()
        labels = load_chat_labels(conn, [r["chat_id"] for r in rows])
        conn.close()

    conv_html = ""
    if q and chat_id is None:
        convs = matching_conversations(q)
        if convs:
            chips = "".join(
                f'<a class="chat-hit" href="/chat/{cid}">{avatar_html(name, ident)}{html.escape(name)}</a>'
                for cid, name, ident in convs
            )
            conv_html = f'<p class="section-sub">Conversations</p><div class="chat-hits">{chips}</div>'

    items = []
    for r in rows:
        who = "Me" if r["is_from_me"] else (resolve_contact(r["handle"]) or r["handle"] or "Unknown")
        chat_name = labels.get(r["chat_id"], "")
        meta_bits = []
        if chat_id is None and chat_name:
            meta_bits.append(html.escape(chat_name))
        meta_bits.append(html.escape(who))
        meta_bits.append(apple_date(r["date"]))
        items.append(
            f'<a class="search-result" href="/chat/{r["chat_id"]}?around={r["msg_id"]}">'
            f'<div class="sr-meta">{" · ".join(meta_bits)}</div>'
            f'<div class="sr-text">{snippet_html(r["body"], phrases, terms)}</div></a>'
        )

    if q:
        shown = len(rows)
        total_note = f"{shown}+" if truncated else str(shown)
        word = "match" if shown == 1 and not truncated else "matches"
        status = f'{total_note} {word} for “{html.escape(q)}”'
        if broadened:
            status += " — no phrase-and-all-words hit, so this is anything that matched a word."
        status_html = f'<p class="section-sub">{status}</p>'
        if truncated:
            status_html += f'<p class="sr-note">Showing the first {SEARCH_LIMIT}.</p>'
        results = "".join(items) if items else '<p class="empty">No matches.</p>'
    else:
        status_html = (
            '<p class="sr-note">Quote a phrase for an exact run of words, like '
            "“see you tomorrow”. Other words match related forms "
            "(run / running) and prefixes, ranked by relevance.</p>"
        )
        results = ""

    if chat_id is not None:
        header_left = f'<a class="back" href="/chat/{chat_id}">← {html.escape(chat_title)}</a>'
        action = f"/chat/{chat_id}/search"
        placeholder = "Search this conversation…"
        extra = f'<a class="btn btn-ghost" href="/search?q={quote(q)}">Search all</a>' if q else '<a class="btn btn-ghost" href="/search">Search all</a>'
        page_title = f"Search: {chat_title}"
        active = "search"
    else:
        header_left = None
        action = "/search"
        placeholder = "Search all messages…"
        extra = ""
        page_title = "Search"
        active = "search"

    return page(
        page_title,
        f"<main class=\"page\">{status_html}{conv_html}{results}</main>",
        active=active,
        header_left=header_left,
        header_right=extra + search_form_html(action, q, placeholder, wide=True),
    )


def render_stat_tile(label, value):
    return f"""<div class="kpi">
<div class="kpi-value countup" data-count="{value}">0</div>
<div class="kpi-label">{html.escape(label)}</div>
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


def smooth_line(points):
    if len(points) < 3:
        return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else p2
        c1x = p1[0] + (p2[0] - p0[0]) / 6
        c1y = p1[1] + (p2[1] - p0[1]) / 6
        c2x = p2[0] - (p3[0] - p1[0]) / 6
        c2y = p2[1] - (p3[1] - p1[1]) / 6
        parts.append(f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {p2[0]:.1f} {p2[1]:.1f}")
    return " ".join(parts)


def fmt_value(v, decimals=0):
    return f"{v:,.{decimals}f}"


def render_trend_chart(months, series, unit="messages", decimals=0, chart_id="trend"):
    """A line for each series over a shared month axis. One series draws a
    filled area, two or more draw plain lines with a key above them."""
    if not months or not series:
        return '<p class="empty">No data.</p>'
    W, H = 900, 240
    pad_l, pad_r, pad_t, pad_b = 48, 12, 16, 28
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    n = len(months)
    col_w = plot_w / n
    max_c = max(max(s["values"]) for s in series) or 1

    def x_at(i):
        return pad_l + col_w * (i + 0.5)

    def y_at(c):
        return pad_t + plot_h * (1 - c / max_c)

    y0 = pad_t + plot_h
    paths = []
    dot_groups = []
    for si, s in enumerate(series):
        points = [(x_at(i), y_at(c)) for i, c in enumerate(s["values"])]
        line_d = smooth_line(points)
        if len(series) == 1:
            area_d = line_d + f" L {points[-1][0]:.1f} {y0:.1f} L {points[0][0]:.1f} {y0:.1f} Z"
            paths.append(f'<path d="{area_d}" class="areapath" fill="url(#fill{chart_id})"></path>')
        paths.append(f'<path d="{line_d}" class="linepath {s["cls"]}"></path>')
        dot_groups.append(
            '<g class="ptgroup">'
            + "".join(f'<circle class="pt {s["cls"]}" cx="{x:.1f}" cy="{y:.1f}" r="0"></circle>' for x, y in points)
            + "</g>"
        )

    grid = [
        f'<line x1="{pad_l}" y1="{y_at(max_c * frac):.1f}" x2="{W - pad_r}" y2="{y_at(max_c * frac):.1f}" class="gridline"></line>'
        for frac in (0.25, 0.5, 0.75)
    ]

    hit_cols = []
    x_labels = []
    seen_years = set()
    for i, ym in enumerate(months):
        year = ym[:4]
        if (ym[5:7] == "01" or i == 0) and year not in seen_years:
            seen_years.add(year)
            x_labels.append(f'<text x="{x_at(i):.1f}" y="{H - 6}" class="axislabel">{year}</text>')
        if len(series) == 1:
            vals = f'{fmt_value(series[0]["values"][i], decimals)} {unit}'
        else:
            vals = " · ".join(f'{s["label"]} {fmt_value(s["values"][i], decimals)}' for s in series)
        top = min(y_at(s["values"][i]) for s in series)
        hit_cols.append(
            f'<rect class="hitcol" x="{pad_l + col_w * i:.1f}" y="{pad_t}" width="{col_w:.1f}" height="{plot_h}" '
            f'data-label="{month_label(ym)}" data-vals="{html.escape(vals, quote=True)}" '
            f'data-cx="{x_at(i):.1f}" data-cy="{top:.1f}"></rect>'
        )

    key = ""
    if len(series) > 1:
        key = '<div class="trendkey">' + "".join(
            f'<span class="tk {s["cls"]}">{html.escape(s["label"])}</span>' for s in series
        ) + "</div>"

    svg = f"""{key}<div class="trendwrap"><svg viewBox="0 0 {W} {H}" class="trendsvg" id="svg{chart_id}" preserveAspectRatio="none">
<defs><linearGradient id="fill{chart_id}" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="var(--signal)" stop-opacity="0.28"/>
<stop offset="100%" stop-color="var(--signal)" stop-opacity="0.02"/>
</linearGradient></defs>
{''.join(grid)}
<line x1="{pad_l}" y1="{y0:.1f}" x2="{W - pad_r}" y2="{y0:.1f}" class="axisline"></line>
<text x="4" y="{pad_t + 4}" class="axislabel">{fmt_value(max_c, decimals)}</text>
{''.join(paths)}
{''.join(dot_groups)}
{''.join(x_labels)}
{''.join(hit_cols)}
</svg><div class="trendtip" id="tip{chart_id}"></div></div>"""

    script = """
(function() {
  const svg = document.getElementById('svg__ID__');
  if (!svg) return;
  svg.querySelectorAll('.linepath').forEach((line, i) => {
    const len = line.getTotalLength();
    line.style.strokeDasharray = len;
    line.style.strokeDashoffset = len;
    requestAnimationFrame(() => {
      line.style.transition = 'stroke-dashoffset 1.2s ease ' + (i * 0.15) + 's';
      line.style.strokeDashoffset = 0;
    });
  });
  const area = svg.querySelector('.areapath');
  if (area) {
    area.style.opacity = 0;
    requestAnimationFrame(() => {
      area.style.transition = 'opacity 1s ease 0.3s';
      area.style.opacity = 1;
    });
  }
  const tip = document.getElementById('tip__ID__');
  const groups = svg.querySelectorAll('.ptgroup');
  svg.querySelectorAll('.hitcol').forEach((col, i) => {
    col.addEventListener('mouseenter', () => {
      groups.forEach(g => g.children[i].setAttribute('r', 4.5));
      const box = svg.getBoundingClientRect();
      tip.style.left = (parseFloat(col.dataset.cx) * box.width / __W__) + 'px';
      tip.style.top = (parseFloat(col.dataset.cy) * box.height / __H__) + 'px';
      tip.textContent = col.dataset.label + ' · ' + col.dataset.vals;
      tip.style.opacity = 1;
    });
    col.addEventListener('mouseleave', () => {
      groups.forEach(g => g.children[i].setAttribute('r', 0));
      tip.style.opacity = 0;
    });
  });
})();
""".replace("__W__", str(W)).replace("__H__", str(H)).replace("__ID__", chart_id)

    return svg + f"<script>{script}</script>"


DRIFT_W, DRIFT_H = 240, 34


def drift_spark(person, months, window, tone):
    """One row of the small-multiple chart, scaled to this person's own peak."""
    values = person["values"]
    n = len(values)
    top = max(values) or 1
    step = DRIFT_W / (n - 1) if n > 1 else DRIFT_W
    xs = [i * step for i in range(n)]
    ys = [DRIFT_H - 1.5 - (v / top) * (DRIFT_H - 3) for v in values]
    line = " ".join(f"{'M' if i == 0 else 'L'} {xs[i]:.1f} {ys[i]:.1f}" for i in range(n))
    area = f"{line} L {xs[-1]:.1f} {DRIFT_H} L 0 {DRIFT_H} Z"
    cut = window * step
    hits = "".join(
        f'<rect class="hitcol" x="{xs[i] - step / 2:.1f}" y="0" width="{step:.1f}" height="{DRIFT_H}" '
        f'data-label="{month_label(months[i])}" data-c="{values[i]}" data-cx="{xs[i]:.1f}"></rect>'
        for i in range(n)
    )
    return (
        f'<svg class="dr-spark {tone}" viewBox="0 0 {DRIFT_W} {DRIFT_H}" preserveAspectRatio="none" '
        f'role="img" aria-label="{html.escape(person["name"])} monthly messages">'
        f'<path class="dr-area" d="{area}"></path>'
        f'<path class="dr-line" d="{line}" vector-effect="non-scaling-stroke"></path>'
        f'<line class="dr-cut" x1="{cut:.1f}" y1="0" x2="{cut:.1f}" y2="{DRIFT_H}"></line>'
        f'{hits}</svg>'
    )


def drift_rows(group, months, window, tone):
    rows = []
    for person in group:
        name = html.escape(person["name"])
        inner = (
            f'<span class="dr-name">{name}</span>'
            f'<span class="dr-plot">{drift_spark(person, months, window, tone)}</span>'
            f'<span class="dr-num {tone}">{person["drift"] * 100:+.1f}<span class="dr-unit">pts</span></span>'
        )
        if person["chat_id"]:
            rows.append(f'<a class="dr-row" href="/chat/{person["chat_id"]}">{inner}</a>')
        else:
            rows.append(f'<div class="dr-row">{inner}</div>')
    return "".join(rows)


def drift_table(rising, faded, window):
    body = "".join(
        f'<tr><td>{html.escape(p["name"])}</td><td>{p["prev"]:,}</td><td>{p["recent"]:,}</td>'
        f'<td>{p["share_prev"] * 100:.1f}%</td><td>{p["share_now"] * 100:.1f}%</td>'
        f'<td>{p["drift"] * 100:+.1f}</td></tr>'
        for p in list(rising) + list(faded)
    )
    return (
        f'<details class="dr-table"><summary>Table view</summary><table>'
        f"<thead><tr><th>Person</th><th>Prior {window} mo</th><th>Last {window} mo</th>"
        f"<th>Share before</th><th>Share now</th><th>Shift</th></tr></thead>"
        f"<tbody>{body}</tbody></table></details>"
    )


def render_people_drift(rising, faded, months, window):
    if not rising and not faded:
        return '<p class="empty">Not enough history yet to compare two years.</p>'
    cut_pct = 100 * window / (len(months) - 1)
    groups = ""
    if rising:
        groups += (
            '<div class="dr-group"><span class="dr-tag up">Drifting toward</span></div>'
            + drift_rows(rising, months, window, "up")
        )
    if faded:
        groups += (
            '<div class="dr-group"><span class="dr-tag down">Drifting away</span></div>'
            + drift_rows(faded, months, window, "down")
        )
    return f"""<div class="trendwrap drift" id="driftwrap">
{groups}
<div class="dr-row dr-axis"><span class="dr-name"></span><span class="dr-plot">
<span class="dr-ax-l">{month_label(months[0])}</span>
<span class="dr-ax-c" style="left:{cut_pct:.2f}%">{month_label(months[window])}</span>
<span class="dr-ax-r">{month_label(months[-1])}</span>
</span><span class="dr-num"></span></div>
{drift_table(rising, faded, window)}
<div class="trendtip dr-tip" id="drifttip"></div>
</div>
{spark_tip_script("driftwrap", "drifttip", DRIFT_W)}"""


def spark_tip_script(wrap_id, tip_id, view_w):
    """Hover tooltip for a stack of sparklines. Scoped to one wrapper, so two
    charts on the same page do not steal each other's columns."""
    return f"""<script>
(function() {{
  const wrap = document.getElementById('{wrap_id}');
  const tip = document.getElementById('{tip_id}');
  if (!wrap || !tip) return;
  wrap.querySelectorAll('.dr-spark .hitcol').forEach(col => {{
    col.addEventListener('mouseenter', () => {{
      const svg = col.ownerSVGElement, box = svg.getBoundingClientRect();
      const wbox = wrap.getBoundingClientRect();
      const sub = col.dataset.sub ? ' · ' + col.dataset.sub : '';
      tip.textContent = col.dataset.text
        || (col.dataset.label + ' · ' + (+col.dataset.c).toLocaleString() + ' messages' + sub);
      tip.style.left = (box.left - wbox.left + col.dataset.cx * box.width / {view_w}) + 'px';
      tip.style.top = (box.top - wbox.top) + 'px';
      tip.style.opacity = 1;
    }});
    col.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; }});
  }});
}})();
</script>"""


ERA_W, ERA_H = 240, 30


def era_spark(person, months, january_x):
    """One person's whole history, as a share of the month, scaled to their own
    peak. A tick marks the peak month."""
    shares = person["shares"]
    n = len(shares)
    top = max(shares) or 1
    step = ERA_W / (n - 1) if n > 1 else ERA_W
    xs = [i * step for i in range(n)]
    ys = [ERA_H - 1.5 - (s / top) * (ERA_H - 3) for s in shares]
    line = " ".join(f"{'M' if i == 0 else 'L'} {xs[i]:.1f} {ys[i]:.1f}" for i in range(n))
    area = f"{line} L {xs[-1]:.1f} {ERA_H} L 0 {ERA_H} Z"
    grid = "".join(
        f'<line class="er-grid" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{ERA_H}"></line>'
        for x in january_x
    )
    peak = person["peak_i"]
    hits = "".join(
        f'<rect class="hitcol" x="{xs[i] - step / 2:.1f}" y="0" width="{step:.1f}" height="{ERA_H}" '
        f'data-label="{month_label(months[i])}" data-c="{person["values"][i]}" '
        f'data-sub="{shares[i] * 100:.0f}% of that month" data-cx="{xs[i]:.1f}"></rect>'
        for i in range(n)
    )
    return (
        f'<svg class="dr-spark era" viewBox="0 0 {ERA_W} {ERA_H}" preserveAspectRatio="none" '
        f'role="img" aria-label="{html.escape(person["name"])} share of each month">'
        f"{grid}"
        f'<path class="dr-area" d="{area}"></path>'
        f'<path class="dr-line" d="{line}" vector-effect="non-scaling-stroke"></path>'
        f'<line class="er-peak" x1="{xs[peak]:.1f}" y1="{ys[peak]:.1f}" x2="{xs[peak]:.1f}" y2="{ERA_H}" '
        f'vector-effect="non-scaling-stroke"></line>'
        f"{hits}</svg>"
    )


def render_people_eras(eras, months):
    if not eras:
        return '<p class="empty">Not enough history yet to show an era.</p>'
    january_x = [
        i * ERA_W / (len(months) - 1)
        for i, ym in enumerate(months)
        if ym.endswith("-01") and i
    ]
    rows = []
    for person in eras:
        # Warm for an old peak, cool for a recent one, so the ladder reads as time.
        t = person["peak_i"] / max(len(months) - 1, 1)
        inner = (
            f'<span class="dr-name">{html.escape(person["name"])}</span>'
            f'<span class="dr-plot">{era_spark(person, months, january_x)}</span>'
            f'<span class="er-when"><b>{short_month(person["peak_ym"])}</b>'
            f'<span>{person["peak_share"] * 100:.0f}% of that month</span></span>'
        )
        style = f'style="--tone: color-mix(in oklab, var(--rise) {t * 100:.0f}%, var(--fade))"'
        tag = "a" if person["chat_id"] else "div"
        href = f' href="/chat/{person["chat_id"]}"' if person["chat_id"] else ""
        rows.append(f'<{tag} class="dr-row er-row"{href} {style}>{inner}</{tag}>')

    labels = "".join(
        f'<span class="er-ax" style="left:{i * 100 / (len(months) - 1):.2f}%">{ym[:4]}</span>'
        for i, ym in enumerate(months)
        if ym.endswith("-01") and i
    )
    return f"""<div class="trendwrap drift eras" id="eraswrap">
{"".join(rows)}
<div class="dr-row dr-axis"><span class="dr-name"></span>
<span class="dr-plot"><span class="dr-ax-l">{short_month(months[0])}</span>{labels}
<span class="dr-ax-r">{short_month(months[-1])}</span></span><span class="er-when"></span></div>
<div class="trendtip dr-tip" id="erastip"></div>
</div>
{spark_tip_script("eraswrap", "erastip", ERA_W)}"""


LAT_W, LAT_H = 240, 34


def format_gap(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


LAT_MAX_GAP_MONTHS = 2


def latency_spark(row, months):
    """Two lines over the shared month axis: how fast you answer them, and how
    fast they answer you.

    The scale inside a row is logarithmic. Reply times run from seconds to
    hours, so one slow month flattens everything else on a linear scale. A
    month without enough replies to trust is skipped, and a run of skipped
    months breaks the line rather than drawing across the hole.
    """
    n = len(months)
    step = LAT_W / (n - 1) if n > 1 else LAT_W
    values = [v for _, v in row["my_points"] + row["their_points"]] or [1]
    lo, hi = max(min(values), 1), max(values)
    span = math.log10(hi / lo) or 1

    def y_at(v):
        return LAT_H - 1.5 - (math.log10(max(v, lo) / lo) / span) * (LAT_H - 3)

    def path(points):
        segments = []
        run = []
        for i, v in points:
            if run and i - run[-1][0] > LAT_MAX_GAP_MONTHS:
                segments.append(run)
                run = []
            run.append((i, v))
        if run:
            segments.append(run)
        out = []
        for segment in segments:
            steps = [f"{'M' if k == 0 else 'L'} {i * step:.1f} {y_at(v):.1f}" for k, (i, v) in enumerate(segment)]
            if len(segment) == 1:
                # A round cap turns a zero-length line into a dot, so a lone
                # month still shows instead of vanishing.
                steps.append(steps[0].replace("M", "L"))
            out.append(" ".join(steps))
        return " ".join(out)

    mine = dict(row["my_points"])
    theirs = dict(row["their_points"])
    hits = []
    for i in sorted(set(mine) | set(theirs)):
        parts = [month_label(months[i])]
        if i in mine:
            parts.append(f"you {format_gap(mine[i])}")
        if i in theirs:
            parts.append(f"them {format_gap(theirs[i])}")
        hits.append(
            f'<rect class="hitcol" x="{i * step - step / 2:.1f}" y="0" width="{step:.1f}" height="{LAT_H}" '
            f'data-text="{html.escape(" · ".join(parts), quote=True)}" data-cx="{i * step:.1f}"></rect>'
        )
    return (
        f'<svg class="dr-spark lat" viewBox="0 0 {LAT_W} {LAT_H}" preserveAspectRatio="none" '
        f'role="img" aria-label="{html.escape(row["name"])} median reply time by month">'
        f'<path class="dr-line lat-them" d="{path(row["their_points"])}" vector-effect="non-scaling-stroke"></path>'
        f'<path class="dr-line lat-me" d="{path(row["my_points"])}" vector-effect="non-scaling-stroke"></path>'
        f'{"".join(hits)}</svg>'
    )


def render_reply_latency(rows, months):
    if not rows:
        return '<p class="empty">Not enough back-and-forth yet to time a reply.</p>'
    out = []
    for row in rows:
        inner = (
            f'<span class="dr-name">{html.escape(row["name"])}</span>'
            f'<span class="dr-plot">{latency_spark(row, months)}</span>'
            f'<span class="lat-nums"><b class="lat-me">{format_gap(row["my_median"])}</b>'
            f'<b class="lat-them">{format_gap(row["their_median"])}</b></span>'
        )
        tag = "a" if row["chat_id"] else "div"
        href = f' href="/chat/{row["chat_id"]}"' if row["chat_id"] else ""
        out.append(f'<{tag} class="dr-row lat-row"{href}>{inner}</{tag}>')
    labels = "".join(
        f'<span class="er-ax" style="left:{i * 100 / (len(months) - 1):.2f}%">{ym[:4]}</span>'
        for i, ym in enumerate(months)
        if ym.endswith("-01") and i
    )
    return f"""<div class="trendkey"><span class="tk lat-me">You answer them</span><span class="tk lat-them">They answer you</span></div>
<div class="trendwrap drift" id="latwrap">
<div class="dr-row lat-row lat-head"><span class="dr-name"></span><span class="dr-plot"></span>
<span class="lat-nums"><b>you</b><b>them</b></span></div>
{"".join(out)}
<div class="dr-row dr-axis"><span class="dr-name"></span>
<span class="dr-plot"><span class="dr-ax-l">{short_month(months[0])}</span>{labels}
<span class="dr-ax-r">{short_month(months[-1])}</span></span><span class="lat-nums"></span></div>
<div class="trendtip dr-tip" id="lattip"></div>
</div>
{spark_tip_script("latwrap", "lattip", LAT_W)}"""


def render_initiation(people, years):
    if not people:
        return '<p class="empty">Not enough separate conversations yet to see who opens them.</p>'
    head = "".join(f'<span class="in-cell in-head">{y}</span>' for y in years)
    rows = []
    for person in people:
        cells = []
        for year in years:
            slot = person["years"].get(year)
            if not slot:
                cells.append('<span class="in-cell in-empty"></span>')
                continue
            mine, total = slot
            share = mine / total
            tone = f"color-mix(in oklab, var(--rise) {share * 100:.0f}%, var(--fade))"
            title = f"{year}: you opened {mine} of {total} conversations"
            cells.append(
                f'<span class="in-cell" title="{html.escape(title, quote=True)}" '
                f'style="background: color-mix(in oklab, {tone} 42%, var(--surface))">'
                f'{share * 100:.0f}%</span>'
            )
        inner = (
            f'<span class="dr-name">{html.escape(person["name"])}</span>'
            f'<span class="in-strip">{"".join(cells)}</span>'
            f'<span class="in-total">{person["share"] * 100:.0f}%'
            f'<span class="dr-unit">of {person["total"]}</span></span>'
        )
        tag = "a" if person["chat_id"] else "div"
        href = f' href="/chat/{person["chat_id"]}"' if person["chat_id"] else ""
        rows.append(f'<{tag} class="dr-row in-row"{href}>{inner}</{tag}>')
    return f"""<div class="trendkey"><span class="tk lat-them">They open</span><span class="tk lat-me">You open</span></div>
<div class="trendwrap drift">
<div class="dr-row in-row in-head-row"><span class="dr-name"></span>
<span class="in-strip">{head}</span><span class="in-total">All time</span></div>
{"".join(rows)}
</div>"""


def reaction_emoji(token):
    if isinstance(token, str):
        return token
    kind = REACTION_LABELS.get(token)
    return kind[0] if kind else "•"


def render_reactions(rows):
    if not rows:
        return '<p class="empty">No tapbacks on your one-to-one threads yet.</p>'
    out = []
    for row in rows:
        chips = "".join(
            f'<span class="rx-chip"><span class="rx-e">{reaction_emoji(token)}</span>{n:,}</span>'
            for token, n in row["got"].most_common(4)
        )
        inner = (
            f'<span class="dr-name">{html.escape(row["name"])}</span>'
            f'<span class="rx-chips">{chips}</span>'
            f'<span class="rx-rate">{row["rate"] * 100:.0f}%'
            f'<span class="dr-unit">of {row["my_msgs"]:,}</span></span>'
        )
        tag = "a" if row["chat_id"] else "div"
        href = f' href="/chat/{row["chat_id"]}"' if row["chat_id"] else ""
        out.append(f'<{tag} class="dr-row rx-row"{href}>{inner}</{tag}>')
    return f'<div class="trendwrap drift">{"".join(out)}</div>'


def _exact_search_href(text):
    """Search URL that quotes the term so it matches as a literal phrase."""
    return "/search?q=" + html.escape(quote(f'"{text}"'), quote=True)


def render_word_cloud(words):
    if not words:
        return ""
    max_n = words[0]["n"] or 1
    parts = []
    for item in words:
        t = (item["n"] / max_n) ** 0.45
        size = 13 + t * 22
        parts.append(
            f'<a class="wc-word" href="{_exact_search_href(item["word"])}" '
            f'style="font-size:{size:.1f}px" title="{item["n"]:,} times">'
            f'{html.escape(item["word"])}</a>'
        )
    return f'<div class="wordcloud">{"".join(parts)}</div>'


def render_voice_section(chat_by_handle):
    voice = load_voice()
    if not voice:
        kick_search_index()
        return (
            '<h2 class="section-h">How you talk</h2>'
            '<p class="section-sub">Word stats are still building from your sent messages. Refresh this page in a moment.</p>'
        )
    phrase_html = "".join(
        f'<a class="vchip" href="{_exact_search_href(p["phrase"])}">'
        f'{html.escape(p["phrase"])}<span class="n">{p["n"]:,}</span></a>'
        for p in voice.get("phrases") or []
    )
    people_html = []
    for p in voice.get("people") or []:
        chips = "".join(
            f'<a class="vchip" href="{_exact_search_href(w["word"])}">'
            f'{html.escape(w["word"])}<span class="lift">{w["lift"]}×</span>'
            f'<span class="n">{w["n"]:,}</span></a>'
            for w in p.get("words") or []
        )
        if not chips:
            continue
        chat_id = chat_by_handle.get(p["handle"])
        name = html.escape(p["name"])
        title = f'<a href="/chat/{chat_id}">{name}</a>' if chat_id else f"<span>{name}</span>"
        people_html.append(
            f'<div class="voice-person"><div class="voice-person-h">{title}'
            f'<span class="n">{p["msgs"]:,} texts you sent</span></div>'
            f'<div class="vchips">{chips}</div></div>'
        )
    years_html = "".join(
        f'<div class="voice-person"><div class="voice-person-h"><span>{y["year"]}</span>'
        f'<span class="n">{y["msgs"]:,} texts you sent</span></div>'
        f'<div class="vchips">'
        + "".join(
            f'<a class="vchip" href="{_exact_search_href(w["word"])}">'
            f'{html.escape(w["word"])}<span class="lift">{w["lift"]}×</span>'
            f'<span class="n">{w["n"]:,}</span></a>'
            for w in y.get("words") or []
        )
        + "</div></div>"
        for y in voice.get("years") or []
        if y.get("words")
    )
    years_block = (
        '<h2 class="section-h">Words that belong to a year</h2>'
        '<p class="section-sub">You said these more that year than you do across all your texts. '
        "Lift is how many times more. The list reads as what you were doing.</p>" + years_html
        if years_html
        else ""
    )
    people_block = (
        "<h2 class=\"section-h\">Words that belong to a person</h2>"
        "<p class=\"section-sub\">You say these more with them than you do overall. "
        "Lift is how many times more.</p>"
        + "".join(people_html)
        if people_html
        else ""
    )
    return f"""<h2 class="section-h">How you talk</h2>
<p class="section-sub">{voice["msgs"]:,} texts you sent · {voice["avg_len"]} characters average. Click a word to search.</p>
<div class="panel">{render_word_cloud(voice.get("words") or [])}</div>
<h2 class="section-h">Phrases you reach for</h2>
<p class="section-sub">Two or more words, when you say them a lot.</p>
<div class="vchips">{phrase_html}</div>
{years_block}
{people_block}"""


def render_stats():
    conn = get_conn()
    totals = conn.execute(
        f"""SELECT count(*) as msg_count, min(date) as first_d, max(date) as last_d
            FROM message m WHERE {REACTION_EXCLUDE_SQL}"""
    ).fetchone()
    chat_total = conn.execute("SELECT count(*) FROM chat").fetchone()[0]

    contacts = compute_contact_stats(conn)
    monthly = monthly_counts(conn)
    people, all_months = monthly_by_person(
        conn, chat_per={c["handle"]: c["chat_id"] for c in contacts}
    )
    stream_series = people_over_time(people, all_months)
    rising, faded, drift_months = people_drift(people, all_months)
    eras, era_months = people_eras(people, all_months)
    owners = year_owners(people, all_months)
    heatmap_html = build_heatmap_html(conn)
    chat_per = {c["handle"]: c["chat_id"] for c in contacts}
    streams = dm_streams(conn, chat_per=chat_per)
    latency = reply_latency(streams, all_months)
    openers, opener_years = initiation(streams)
    reactions = reaction_stats(conn, streams)
    effective = effective_people(people, all_months)
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
        lambda it: f"Friends since {it['first_dt'].year} · {it['chat_count']} shared chat{'s' if it['chat_count'] != 1 else ''}",
    ) or '<p class="empty">No contacts cross the 1-year, active-in-90-days bar yet.</p>'

    fell_off_html = render_contact_cards(
        fell_off,
        lambda it: f"{it['gap_days']} days quiet · last texted {it['last_dt'].strftime('%Y-%m-%d')}",
    ) or '<p class="empty">Nobody with 30+ messages has gone quiet for 180+ days.</p>'

    peak_items = [
        {
            "name": s["name"],
            "handle": s["handle"],
            "chat_id": s["chat_id"],
            "count": s["peak_c"],
            "peak_ym": s["peak_ym"],
        }
        for s in sorted(stream_series, key=lambda s: -s["peak_c"])
    ]
    peak_html = render_contact_cards(
        peak_items,
        lambda it: f"Most messages in {month_label(it['peak_ym'])}",
    ) or '<p class="empty">Not enough history to find a peak month yet.</p>'

    owners_html = render_contact_cards(
        owners,
        lambda it: f"{it['year']} · {it['share'] * 100:.0f}% of everything you received",
    ) or '<p class="empty">Not enough history to name a year yet.</p>'

    body = f"""<main class="page">
<div class="kpirow">{kpi_html}</div>

<h2 class="section-h">Activity, every day</h2>
{heatmap_html}

<h2 class="section-h">How fast you answer</h2>
<p class="section-sub">Median time to reply, for your {len(latency)} busiest one-to-one threads, month by month.
The clock starts on the first message nobody has answered yet. A turn that takes more than a day is a new
conversation, not a reply, so it does not count. Each row has its own log scale, so up is slower.</p>
{render_reply_latency(latency, all_months)}

<h2 class="section-h">Who reaches first</h2>
<p class="section-sub">Share of conversations that you opened, year by year. A new conversation starts after
{SESSION_GAP_SECONDS // 3600} hours of silence. Near 50% is a thread you both open. Near 0 or 100 is one person carrying it.</p>
{render_initiation(openers, opener_years)}

<h2 class="section-h">How wide your attention is</h2>
<p class="section-sub">The effective number of people you texted each month. One person holding everything reads as 1.
Ten people at an even tenth reads as 10. It moves on its own: your total volume can climb while the circle narrows.</p>
{render_trend_chart(
    [ym for ym, _ in effective],
    [{"label": "People", "cls": "s-solo", "values": [v for _, v in effective]}],
    unit="people", decimals=1, chart_id="eff",
)}

<h2 class="section-h">Who reacts to you</h2>
<p class="section-sub">Tapbacks on the messages you sent, as a share of everything you sent them.
One-to-one threads only, so the reactor is never in doubt.</p>
{render_reactions(reactions)}

{render_voice_section({c["handle"]: c["chat_id"] for c in contacts})}

<h2 class="section-h">People fading in and out</h2>
<p class="section-sub">Share of everything you received, last {DRIFT_WINDOW_MONTHS} months against the {DRIFT_WINDOW_MONTHS} before.
Share, not raw count, because your total volume kept growing. Each line is scaled to its own peak.</p>
{render_people_drift(rising, faded, drift_months, DRIFT_WINDOW_MONTHS)}

<h2 class="section-h">Eras</h2>
<p class="section-sub">Everyone with {ERA_MIN_MSGS}+ messages, all time, as a share of everything you received that month.
Sorted by the month they held the largest share, so the list reads top to bottom as time.</p>
{render_people_eras(eras, era_months)}

<h2 class="section-h">Who held each year</h2>
<p class="section-sub">The person with the largest share of everything you received, year by year.</p>
{owners_html}

<h2 class="section-h">When you were closest</h2>
<p class="section-sub">The single busiest month with each of your {len(stream_series)} most-texted people.</p>
{peak_html}

<h2 class="section-h">Messages per month</h2>
{render_trend_chart(
    [ym for ym, _, _ in monthly],
    [
        {"label": "Received", "cls": "s-recv", "values": [r for _, r, _ in monthly]},
        {"label": "Sent", "cls": "s-sent", "values": [s for _, _, s in monthly]},
    ],
)}

<h2 class="section-h">Most contacted</h2>
<div class="panel">{render_leaderboard(most_contacted, max_count)}</div>

<h2 class="section-h">Long-term friends</h2>
<p class="section-sub">Talking for {LONG_TERM_MIN_SPAN_DAYS}+ days, still active in the last {LONG_TERM_MAX_GAP_DAYS} days.</p>
{long_term_html}

<h2 class="section-h">Fell out of touch</h2>
<p class="section-sub">{FELL_OFF_MIN_COUNT}+ lifetime messages, quiet for {FELL_OFF_MIN_GAP_DAYS}+ days.</p>
{fell_off_html}
</main>"""
    return page("Stats", body, active="stats")


def json_script(element_id, obj):
    payload = json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return f'<script type="application/json" id="{html.escape(element_id)}">{payload}</script>'


def render_circles():
    conn = get_conn()
    data = load_circle_graph(conn)
    conn.close()
    n_people = len(data["people"])
    n_groups = len(data["groups"])
    if not n_groups:
        body = """<main class="circles-page" id="circlesPage">
<div class="circles-empty">
<h1>Circles</h1>
<p>Group chats will show up here. Right now every conversation is one-to-one, so there is no overlap to draw.</p>
</div>
</main>"""
        return page("Circles", body, active="circles", body_class="circlespage")

    people_word = "person" if n_people == 1 else "people"
    group_word = "group chat" if n_groups == 1 else "group chats"
    solo = data["solo"]
    solo_bit = f" · {solo:,} more only in one-to-one threads" if solo else ""
    body = f"""<main class="circles-page" id="circlesPage">
{json_script("circlesData", data)}
<div class="circles-hud">
<input class="search-input" id="circlesFind" type="search" placeholder="Find someone…" autocomplete="off" aria-label="Find someone">
<p class="circles-meta" id="circlesMeta">{n_groups:,} {group_word} · {n_people:,} {people_word}{html.escape(solo_bit)}</p>
</div>
<div class="circles-viewport" id="circlesViewport">
<div class="circles-world" id="circlesWorld">
<svg class="circles-edges" id="circlesEdges" viewBox="0 0 4000 4000" width="4000" height="4000" aria-hidden="true"></svg>
<div class="circles-nodes" id="circlesNodes"></div>
</div>
</div>
<aside class="circles-sheet" id="circlesSheet" aria-live="polite"></aside>
</main>"""
    return page("Circles", body, active="circles", body_class="circlespage")


def render_twin():
    from twin.job import snapshot

    s = snapshot()
    train_dis = " disabled" if (not s["mlx"] or s["busy"]) else ""
    select_dis = " disabled" if s["busy"] else ""
    selected_model = s.get("model") or "qwen3-capable"
    selected_info = next(
        (m for m in s["models"] if m["key"] == selected_model),
        s["models"][0],
    )
    adapter_runs = s.get("adapter_runs") or []
    chat_ready = bool(adapter_runs)
    send_dis = " disabled" if (not chat_ready or s["busy"]) else ""
    chat_select_dis = " disabled" if (s["busy"] or not chat_ready) else ""
    chat_picker_hidden = "" if chat_ready else " hidden"

    def run_label(run):
        bits = [run.get("name") or run.get("model") or "Model"]
        if run.get("params"):
            bits.append(run["params"])
        created = run.get("created_at")
        if isinstance(created, (int, float)) and created > 0:
            bits.append(datetime.fromtimestamp(created).strftime("%b %-d, %-I:%M %p"))
        if run.get("iters"):
            bits.append(f'{int(run["iters"]):,} steps')
        if run.get("data_hash"):
            bits.append(str(run["data_hash"])[:8])
        return " · ".join(bits)

    def checkpoint_label(ckpt):
        if ckpt.get("step") == "latest":
            n = ckpt.get("step_n") or 0
            return f"Latest · {int(n):,} steps" if n else "Latest"
        return f'Step {int(ckpt["step_n"] or ckpt["step"]):,}'

    def options_for_runs(runs, selected_id=""):
        chunks = []
        for run in runs:
            ckpts = run.get("checkpoints") or []
            if not ckpts:
                continue
            label = html.escape(run_label(run))
            if len(ckpts) == 1:
                ckpt = ckpts[0]
                selected = " selected" if ckpt["id"] == selected_id else ""
                chunks.append(
                    f'<option value="{html.escape(ckpt["id"])}"{selected}>{label}</option>'
                )
                continue
            chunks.append(f'<optgroup label="{label}">')
            for ckpt in ckpts:
                selected = " selected" if ckpt["id"] == selected_id else ""
                chunks.append(
                    f'<option value="{html.escape(ckpt["id"])}"{selected}>'
                    f"{html.escape(checkpoint_label(ckpt))}</option>"
                )
            chunks.append("</optgroup>")
        return "".join(chunks)

    model_groups = {}
    for model in s["models"]:
        suffix = " · trained" if model["has_adapter"] else ""
        selected = " selected" if model["key"] == selected_model else ""
        option = (
            f'<option value="{html.escape(model["key"])}"{selected}>'
            f'{html.escape(model["name"])} — {html.escape(model["params"])} · '
            f'{html.escape(model["download"])}{suffix}</option>'
        )
        model_groups.setdefault(model["category"], []).append(option)
    model_options = "".join(
        f'<optgroup label="{html.escape(category)}">{"".join(options)}</optgroup>'
        for category, options in model_groups.items()
    )
    first_chat = ""
    if adapter_runs and adapter_runs[0].get("checkpoints"):
        first_chat = adapter_runs[0]["checkpoints"][0]["id"]
    chat_options = options_for_runs(adapter_runs, first_chat)
    resume_runs = [run for run in adapter_runs if run.get("model") == selected_model]
    resume_options = '<option value="" selected>Fresh weights</option>' + options_for_runs(
        resume_runs
    )
    model_badges = "".join(
        (
            (
            '<span class="twin-recommended" id="twinModelRecommended">Recommended</span>'
            if selected_info.get("recommended")
            else '<span class="twin-recommended" id="twinModelRecommended" hidden>Recommended</span>'
            ),
            (
            '<span class="twin-downloaded" id="twinModelDownloaded">Downloaded</span>'
            if selected_info["cached"]
            else '<span class="twin-downloaded" id="twinModelDownloaded" hidden>Downloaded</span>'
            ),
            (
            '<span class="twin-trained" id="twinModelTrained">Trained</span>'
            if selected_info["has_adapter"]
            else '<span class="twin-trained" id="twinModelTrained" hidden>Trained</span>'
            ),
        )
    )
    if not s["mlx"]:
        hint = (
            "Install MLX training support in this app's virtualenv, then restart the server:"
            "<pre>./.venv/bin/python -m pip install -r twin/requirements.txt</pre>"
        )
    else:
        hint = (
            "Quick is a 30-step smoke test. Complete uses every chat. "
            "Leave steps blank for the default, or continue from a saved checkpoint. "
            "Each train writes a new adapter and leaves earlier ones alone."
        )
    if s["busy"] or s["phase"] in ("error", "cancelled"):
        default_tab = "model"
    elif adapter_runs:
        default_tab = "chat"
    else:
        default_tab = "audit"
    busy_cls = " is-busy" if s["busy"] else ""
    metrics_hidden = "" if (s["busy"] or s["metrics"] or s["phase"] in ("error", "cancelled")) else " hidden"

    def tab_state(name):
        on = name == default_tab
        return f'aria-selected="{"true" if on else "false"}"'

    def panel_hidden(name):
        return "" if name == default_tab else " hidden"

    return page(
        "Twin",
        f"""<main class="page twin-page{busy_cls}" id="twinPage" data-tab="{default_tab}" data-person="me">
<header class="twin-hero">
<div class="twin-hero-copy">
<div class="twin-hero-who">
<h1 class="twin-title">Twin</h1>
<div class="twin-who" id="twinWho">
<p class="twin-eyebrow" id="twinWhoLabel">Train as</p>
<button type="button" class="twin-who-btn" id="twinWhoBtn" aria-haspopup="listbox" aria-expanded="false" aria-controls="twinWhoMenu">
<span class="twin-who-avatar" id="twinWhoAvatar">{avatar_html("You")}</span>
<span class="twin-who-name" id="twinWhoName">You</span>
<span class="twin-who-caret" aria-hidden="true">⌄</span>
</button>
<div class="twin-who-menu" id="twinWhoMenu" hidden>
<input class="twin-who-search field" id="twinWhoSearch" type="search" placeholder="Search contacts" autocomplete="off" aria-label="Search contacts">
<ul class="twin-who-list" id="twinWhoList" role="listbox" aria-labelledby="twinWhoLabel"></ul>
</div>
</div>
</div>
<p class="twin-lede" id="twinLede">Fine-tune a private local model on the way you actually text. Messages and adapters never leave this Mac.</p>
</div>
<div class="twin-tabs" id="twinTabs" role="tablist" aria-label="Twin">
<a class="twin-tab" role="tab" id="twinTabAudit" href="#audit" data-tab="audit" aria-controls="twinPanelAudit" {tab_state("audit")}>Audit</a>
<a class="twin-tab" role="tab" id="twinTabModel" href="#model" data-tab="model" aria-controls="twinPanelModel" {tab_state("model")}>Model<span class="twin-tab-dot" aria-hidden="true"></span></a>
<a class="twin-tab" role="tab" id="twinTabChat" href="#chat" data-tab="chat" aria-controls="twinPanelChat" {tab_state("chat")}>Chat</a>
</div>
</header>

<div class="twin-panels">
<section class="twin-panel twin-panel-audit" id="twinPanelAudit" role="tabpanel" data-tab="audit" aria-labelledby="twinDataTitle"{panel_hidden("audit")}>
<div class="twin-data card card-pad">
<div class="twin-section-head">
<div><h2 id="twinDataTitle">Your training material</h2></div>
<p class="twin-section-note" id="twinDataNote">Inspecting the archive…</p>
</div>
<div class="twin-data-grid" id="twinDataGrid">
<div><strong data-stat="sent_texts">—</strong><span>sent texts</span></div>
<div><strong data-stat="direct_chats">—</strong><span>direct chats</span></div>
<div><strong data-stat="group_chats">—</strong><span>group chats</span></div>
<div><strong data-stat="attachments_only">—</strong><span>media-only, skipped</span></div>
</div>
<p class="twin-data-copy" id="twinDataCopy">Direct 1:1 chats are used for training. Each reply is one example. Later sessions are held out so validation is real. Group chats are counted here and left out of the adapter. Short acknowledgments are capped. Tapbacks, unsent notices, and media without text are skipped. Secrets are replaced before training.</p>
</div>
</section>

<section class="twin-panel twin-panel-model" id="twinPanelModel" role="tabpanel" data-tab="model" aria-labelledby="twinModelTitle"{panel_hidden("model")}>
<div class="twin-train card card-pad">
<div class="twin-section-head">
<div><h2 id="twinModelTitle">Choose the brain</h2></div>
<p class="twin-section-note">Vetted 4-bit MLX chat models up to 8B. Weights download only when you train or chat.</p>
</div>
<div class="twin-model-picker" id="twinModels">
<label class="twin-model-label" for="twinModelSelect">Model</label>
<div class="twin-select-wrap">
<select class="twin-model-select" id="twinModelSelect" name="twinmodel"{select_dis}>{model_options}</select>
<span class="twin-select-arrow" aria-hidden="true">⌄</span>
</div>
<article class="twin-model-summary" id="twinModelSummary">
<div class="twin-model-head"><strong id="twinModelName">{html.escape(selected_info['name'])}</strong><span>{model_badges}</span></div>
<p class="twin-model-size" id="twinModelMeta">{html.escape(selected_info['publisher'])} · {html.escape(selected_info['params'])} parameters · {html.escape(selected_info['download'])} download</p>
<p class="twin-model-copy" id="twinModelCopy">{html.escape(selected_info['description'])}</p>
<p class="twin-model-memory" id="twinModelMemory">{html.escape(selected_info['memory'])}</p>
</article>
</div>
<div class="twin-launch">
<div class="twin-launch-main">
<p class="twin-eyebrow">Train</p>
<div class="seg" id="twinRun">
<label class="seg-btn"><input type="radio" name="twinrun" value="quick"> Quick</label>
<label class="seg-btn"><input type="radio" name="twinrun" value="complete" checked> Complete</label>
</div>
</div>
<div class="twin-launch-opts">
<label class="twin-launch-field" for="twinIters">
<span>Steps</span>
<input class="field twin-iters" id="twinIters" name="twiniters" type="number" min="1" max="1000000" inputmode="numeric" placeholder="Auto"{select_dis}>
</label>
<label class="twin-launch-field twin-launch-resume" for="twinResume">
<span>Start from</span>
<div class="twin-select-wrap">
<select class="twin-model-select twin-resume-select" id="twinResume" name="twinresume"{select_dis}>{resume_options}</select>
<span class="twin-select-arrow" aria-hidden="true">⌄</span>
</div>
</label>
</div>
<button type="button" class="btn btn-primary twin-train-btn" id="twinTrain"{train_dis}>Train locally</button>
</div>
<div class="twin-hint muted">{hint}</div>
</div>
<div class="twin-metrics" id="twinMetrics"{metrics_hidden}>
<div class="twin-section-head">
<div><h2 id="twinMetricsTitle">How the model is learning</h2></div>
<p class="twin-section-note">Train and validation loss from the MLX loop. Validation uses later sessions that are not in training. Throughput, learning rate, and memory are the same step reports.</p>
</div>
<dl class="twin-metric-readout" id="twinMetricReadout" hidden></dl>
<div class="twin-chart-grid">
<figure class="card card-pad twin-chart"><figcaption><strong>Loss</strong><span>Lower is better</span></figcaption><div class="twin-chart-plot"><svg id="twinLossChart" viewBox="0 0 520 190" role="img" aria-label="Training and validation loss"><g class="chart-grid"></g><path class="chart-line chart-train"></path><path class="chart-line chart-reference"></path><g class="chart-dots"></g><g class="chart-hover"></g><g class="chart-labels"></g><rect class="chart-hitbox" x="34" y="10" width="474" height="154"></rect></svg><div class="trendtip twin-chart-tip"></div></div><div class="twin-legend"><span class="is-train">Train</span><span class="is-reference">Val</span></div></figure>
<figure class="card card-pad twin-chart"><figcaption><strong>Throughput</strong><span>Tokens per second</span></figcaption><div class="twin-chart-plot"><svg id="twinSpeedChart" viewBox="0 0 520 190" role="img" aria-label="Training throughput"><g class="chart-grid"></g><path class="chart-line chart-speed"></path><g class="chart-hover"></g><g class="chart-labels"></g><rect class="chart-hitbox" x="34" y="10" width="474" height="154"></rect></svg><div class="trendtip twin-chart-tip"></div></div><div class="twin-chart-summary"><span id="twinPeakSpeed">Waiting for training</span><span id="twinPeakMemory"></span></div></figure>
</div>
</div>
<div class="twin-live" id="twinLive">
<div class="twin-live-head">
<p class="twin-status" id="twinStatus" aria-live="polite"></p>
<p class="twin-progress-meta" id="twinProgressMeta"></p>
<button type="button" class="btn btn-ghost twin-stop" id="twinStop" hidden>Stop</button>
<button type="button" class="btn btn-primary twin-go-chat" id="twinGoChat" hidden>Go to chat</button>
</div>
<div class="twin-progress" id="twinProgress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span></span></div>
<ol class="twin-steps" id="twinSteps">
<li data-phase="inspect"><a href="#audit"><span>1</span><div><strong>Audit archive</strong><small>Count usable text without exposing it.</small><time class="twin-step-time"></time></div></a></li>
<li data-phase="export"><a href="#model"><span>2</span><div><strong>Build pairs</strong><small>Sessionize 1:1 chats and hold out later sessions.</small><time class="twin-step-time"></time></div></a></li>
<li data-phase="train"><a href="#model"><span>3</span><div><strong>Fit adapter</strong><small>Download weights if needed, then train on your JSONL.</small><time class="twin-step-time"></time></div></a></li>
<li data-phase="chat"><a href="#chat"><span>4</span><div><strong>Text the twin</strong><small>Open chat when you want to try the adapter.</small><time class="twin-step-time"></time></div></a></li>
</ol>
</div>
<div class="twin-runs card card-pad" id="twinRuns" hidden>
<div class="twin-section-head">
<div>
<p class="twin-eyebrow">History</p>
<h2 id="twinRunsTitle">Training attempts</h2>
</div>
<p class="twin-section-note">Latest first. Compare duration, data size, and loss across attempts.</p>
</div>
<ol id="twinRunList"></ol>
</div>
</section>

<section class="twin-panel twin-panel-chat" id="twinPanelChat" role="tabpanel" data-tab="chat" aria-labelledby="twinChatTitle"{panel_hidden("chat")}>
<div class="twin-chat-head">
<div><h2 id="twinChatTitle">Text your twin</h2></div>
<div class="twin-chat-tools">
<div class="twin-select-wrap twin-chat-select-wrap" id="twinChatPicker"{chat_picker_hidden}>
<select class="twin-model-select twin-chat-select" id="twinChatSelect" name="twinchat" aria-label="Trained checkpoint"{chat_select_dis}>{chat_options}</select>
<span class="twin-select-arrow" aria-hidden="true">⌄</span>
</div>
<button type="button" class="btn btn-ghost twin-new-chat" id="twinNewChat" disabled>New chat</button>
</div>
</div>
<div class="twin-thread" id="twinThread"></div>
<form class="twin-compose" id="twinCompose">
<input class="field" id="twinInput" type="text" maxlength="500" autocomplete="off" placeholder="Text the twin…"{send_dis}>
<button class="btn btn-primary" type="submit" id="twinSend"{send_dis}>Send</button>
</form>
</section>
</div>
</main>""",
        active="twin",
        body_class="twinpage",
    )


def render_db_error(detail):
    if live_db_error():
        lead = "Open the steps in the red bar at the top of this page."
    else:
        lead = (
            f"The file <code>{html.escape(DB_PATH)}</code> is readable, but SQLite cannot open it. "
            "Quit Messages.app, then reload this page."
        )
    body = f"""<main class="page">
<h1 class="section-h">Cannot open the local Messages database</h1>
<p>{lead}</p>
<p class="muted">{html.escape(detail)}</p>
</main>"""
    return page("Messages", body)
