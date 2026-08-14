"""HTML for every page. Styles and shared behavior live in static/."""

import html
import json
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
    apple_date,
    date_to_apple_ns,
    fetch_messages,
    fetch_messages_around,
    get_conn,
    has_neighbor,
    load_attachments,
    load_reactions,
    message_text,
    reaction_label,
    strip_guid_prefix,
)
from search import (
    index_error,
    is_index_ready,
    kick_search_index,
    matching_conversations,
    search_messages,
    snippet_html,
)
from stats import (
    FELL_OFF_MIN_COUNT,
    FELL_OFF_MIN_GAP_DAYS,
    LONG_TERM_MAX_GAP_DAYS,
    LONG_TERM_MIN_SPAN_DAYS,
    STREAM_COLORS,
    compute_contact_stats,
    monthly_counts,
    people_over_time,
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
)


def nav_html(active):
    links = "".join(
        f'<a class="nav-link{" is-active" if key == active else ""}" href="{href}">{label}</a>'
        for href, label, key in NAV
    )
    return f'<nav class="nav">{links}</nav>'


THEME_BOOT = (
    "<script>(function(){var t=localStorage.getItem('theme');"
    "if(t!=='light'&&t!=='dark')t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';"
    "document.documentElement.dataset.theme=t;"
    "document.documentElement.style.colorScheme=t})();</script>"
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


def page(title, body, *, active="chats", header_left=None, header_right="", body_class="", scripts="", chat_id=None):
    left = header_left if header_left is not None else (
        f'<a class="brand" href="/">Messages</a>{nav_html(active)}'
    )
    data_chat = f' data-chat-id="{chat_id}"' if chat_id else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{THEME_BOOT}
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{asset_url("app.css")}">
</head>
<body class="{html.escape(body_class)}"{data_chat}>
<header class="topbar">
<div class="topbar-left">{left}</div>
<div class="topbar-right">{header_right}{THEME_TOGGLE}</div>
</header>
{body}
<script src="{asset_url("app.js")}" defer></script>
{scripts}
</body>
</html>"""


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
        if reactions:
            grouped = {}
            for label, who_reacted in reactions:
                grouped.setdefault(label, []).append(who_reacted)
            for label, names in grouped.items():
                body += f'<div class="reaction-pill">{html.escape(label)} &middot; {html.escape(", ".join(names))}</div>'

        parts.append(body)
        if is_last_in_group:
            parts.append(f'<div class="ts">{format_time(r["date"])}</div>')
        blocks.append(f'<div class="row {who}{group_cls}{highlight_cls}" id="msg-{r["id"]}">{"".join(parts)}</div>')
        prev_sender = sender_key
    return "".join(blocks)


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
    start_aligned = start - timedelta(days=(start.isoweekday() % 7))
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
            level = bucket(c)
            title = f"{day_str}: {c} message{'s' if c != 1 else ''}"
            clickable = c and chat_id is not None
            onclick = f' onclick="location.href=\'/chat/{chat_id}?date={day_str}\'"' if clickable else ""
            cls = "hcell clickable" if clickable else "hcell"
            cell_html_parts.append(
                f'<div class="{cls} heat-{level}" data-day="{day_str}" title="{title}"{onclick}></div>'
            )

    legend = "".join(f'<div class="hcell heat-{i}"></div>' for i in range(5))
    return f"""<div class="heatmap-card"><div class="heatmap">
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
        inner = f'<video src="/attachment/{a["att_id"]}" preload="metadata" muted></video>'
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

    media = conn.execute(
        """SELECT att.ROWID as att_id, att.mime_type, att.filename, m.ROWID as msg_id, m.date
           FROM message_attachment_join maj
           JOIN attachment att ON att.ROWID = maj.attachment_id
           JOIN message m ON m.ROWID = maj.message_id
           JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
           WHERE cmj.chat_id = ? AND att.filename NOT LIKE '%.pluginPayloadAttachment'
           ORDER BY m.date ASC, m.ROWID ASC""",
        (chat_id,),
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
        header_right=f'<a class="btn btn-ghost" href="/media">All photos</a><span class="muted">{len(items):,} items</span>',
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
        header_right=f'<span class="muted">{len(items):,} items</span>',
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
        conn.close()

    q = (query or "").strip()
    if q and not is_index_ready():
        kick_search_index()
        if not is_index_ready():
            return render_indexing_page(query, chat_id)

    rows, phrases, terms, broadened = search_messages(q, chat_id=chat_id) if q else ([], [], [], False)
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


def render_trend_chart(monthly):
    if not monthly:
        return '<p class="empty">No data.</p>'
    W, H = 900, 240
    pad_l, pad_r, pad_t, pad_b = 48, 12, 16, 28
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
    line_d = smooth_line(points)
    y0 = pad_t + plot_h
    area_d = line_d + f" L {points[-1][0]:.1f} {y0:.1f} L {points[0][0]:.1f} {y0:.1f} Z"

    grid = []
    for frac in (0.25, 0.5, 0.75):
        y = y_at(max_c * frac)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" class="gridline"></line>')

    hit_cols = []
    dots = []
    x_labels = []
    seen_years = set()
    for i, (ym, c) in enumerate(monthly):
        x, y = points[i]
        year = ym[:4]
        if (ym[5:7] == "01" or i == 0) and year not in seen_years:
            seen_years.add(year)
            x_labels.append(f'<text x="{x:.1f}" y="{H - 6}" class="axislabel">{year}</text>')
        hit_cols.append(
            f'<rect class="hitcol" x="{pad_l + col_w * i:.1f}" y="{pad_t}" width="{col_w:.1f}" height="{plot_h}" '
            f'data-label="{ym}" data-count="{c}" data-cx="{x:.1f}" data-cy="{y:.1f}"></rect>'
        )
        dots.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="0"></circle>')

    svg = f"""<div class="trendwrap"><svg viewBox="0 0 {W} {H}" class="trendsvg" id="trendsvg" preserveAspectRatio="none">
<defs><linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="var(--signal)" stop-opacity="0.28"/>
<stop offset="100%" stop-color="var(--signal)" stop-opacity="0.02"/>
</linearGradient></defs>
{''.join(grid)}
<line x1="{pad_l}" y1="{y0:.1f}" x2="{W - pad_r}" y2="{y0:.1f}" class="axisline"></line>
<text x="4" y="{pad_t + 4}" class="axislabel">{max_c:,}</text>
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
      dots[i].setAttribute('r', 4.5);
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


def render_people_stream(series, months):
    if not series or not months:
        return '<p class="empty">No data.</p>'
    W, H = 900, 280
    pad_l, pad_r, pad_t, pad_b = 12, 12, 12, 28
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    n = len(months)
    col_w = plot_w / n
    xs = [pad_l + col_w * (i + 0.5) for i in range(n)]
    totals = [sum(s["values"][i] for s in series) for i in range(n)]
    max_total = max(totals) or 1
    scale = plot_h / max_total
    y_mid = pad_t + plot_h / 2

    bands = []
    baselines = [y_mid - t * scale / 2 for t in totals]
    for si, s in enumerate(series):
        y_top = []
        y_bot = []
        for i, v in enumerate(s["values"]):
            y_bot.append(baselines[i])
            y_top.append(baselines[i] + v * scale)
            baselines[i] += v * scale
        top_d = " ".join(f"{'M' if i == 0 else 'L'} {xs[i]:.1f} {y_top[i]:.1f}" for i in range(n))
        bot_d = " ".join(f"L {xs[i]:.1f} {y_bot[i]:.1f}" for i in range(n - 1, -1, -1))
        color = STREAM_COLORS[si % len(STREAM_COLORS)]
        bands.append(
            f'<path class="sg-band" data-i="{si}" fill="{color}" d="{top_d} {bot_d} Z"></path>'
        )

    hit_cols = []
    x_labels = []
    seen_years = set()
    for i, ym in enumerate(months):
        year = ym[:4]
        if (ym[5:7] == "01" or i == 0) and year not in seen_years:
            seen_years.add(year)
            x_labels.append(f'<text x="{xs[i]:.1f}" y="{H - 6}" class="axislabel">{year}</text>')
        payload = [
            {"name": s["name"], "c": s["values"][i]}
            for s in series
            if s["values"][i]
        ]
        payload.sort(key=lambda x: -x["c"])
        hit_cols.append(
            f'<rect class="hitcol" x="{pad_l + col_w * i:.1f}" y="{pad_t}" width="{col_w:.1f}" height="{plot_h}" '
            f'data-label="{ym}" data-cx="{xs[i]:.1f}" data-people="{html.escape(json.dumps(payload), quote=True)}"></rect>'
        )

    legend = "".join(
        f'<div class="sg-leg" data-i="{i}">'
        f'<span class="sg-dot" style="background:{STREAM_COLORS[i % len(STREAM_COLORS)]}"></span>'
        f'{html.escape(s["name"])}</div>'
        for i, s in enumerate(series)
    )

    svg = f"""<div class="trendwrap"><svg viewBox="0 0 {W} {H}" class="trendsvg streamsvg" id="streamsvg" preserveAspectRatio="none">
{''.join(bands)}
{''.join(x_labels)}
{''.join(hit_cols)}
</svg><div class="trendtip sg-tip" id="streamtip"></div>
<div class="sg-legend">{legend}</div></div>"""

    script = """
(function() {
  const svg = document.getElementById('streamsvg');
  if (!svg) return;
  const tip = document.getElementById('streamtip');
  const bands = svg.querySelectorAll('.sg-band');
  svg.querySelectorAll('.hitcol').forEach(col => {
    col.addEventListener('mouseenter', () => {
      const box = svg.getBoundingClientRect();
      const people = JSON.parse(col.dataset.people || '[]');
      const lines = people.slice(0, 6).map(p => p.name + ': ' + p.c.toLocaleString());
      tip.style.left = (parseFloat(col.dataset.cx) * box.width / __W__) + 'px';
      tip.style.top = '20px';
      tip.textContent = col.dataset.label + (lines.length ? '\\n' + lines.join('\\n') : '');
      tip.style.opacity = 1;
    });
    col.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
  });
  document.querySelectorAll('.sg-leg').forEach(leg => {
    leg.addEventListener('mouseenter', () => {
      const i = leg.dataset.i;
      bands.forEach(b => b.classList.toggle('dim', b.dataset.i !== i));
      bands.forEach(b => b.classList.toggle('hot', b.dataset.i === i));
    });
    leg.addEventListener('mouseleave', () => {
      bands.forEach(b => b.classList.remove('dim', 'hot'));
    });
  });
})();
""".replace("__W__", str(W)).replace("__H__", str(H))

    return svg + f"<script>{script}</script>"


def render_word_cloud(words):
    if not words:
        return ""
    max_n = words[0]["n"] or 1
    parts = []
    for item in words:
        t = (item["n"] / max_n) ** 0.45
        size = 13 + t * 22
        q = quote(item["word"])
        parts.append(
            f'<a class="wc-word" href="/search?q={html.escape(q, quote=True)}" '
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
        f'<a class="vchip" href="/search?q={html.escape(quote(p["phrase"]), quote=True)}">'
        f'{html.escape(p["phrase"])}<span class="n">{p["n"]:,}</span></a>'
        for p in voice.get("phrases") or []
    )
    three_word_phrase_html = "".join(
        f'<a class="vchip" href="/search?q={html.escape(quote(p["phrase"]), quote=True)}">'
        f'{html.escape(p["phrase"])}<span class="n">{p["n"]:,}</span></a>'
        for p in voice.get("three_word_phrases") or []
    )
    people_html = []
    for p in voice.get("people") or []:
        chips = "".join(
            f'<a class="vchip" href="/search?q={html.escape(quote(w["word"]), quote=True)}">'
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
<div class="vchips">{phrase_html}</div>
<h2 class="section-h">Three-word phrases you reach for</h2>
<div class="vchips">{three_word_phrase_html}</div>
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
    stream_series, stream_months = people_over_time(
        conn, chat_per={c["handle"]: c["chat_id"] for c in contacts}
    )
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

    body = f"""<main class="page">
<div class="kpirow">{kpi_html}</div>

<h2 class="section-h">Activity, every day</h2>
{heatmap_html}

{render_voice_section({c["handle"]: c["chat_id"] for c in contacts})}

<h2 class="section-h">People fading in and out</h2>
<p class="section-sub">Monthly messages from your {len(stream_series)} most-texted people. Hover a name to isolate them.</p>
{render_people_stream(stream_series, stream_months)}

<h2 class="section-h">When you were closest</h2>
<p class="section-sub">The single busiest month with each of those people.</p>
{peak_html}

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
</main>"""
    return page("Stats", body, active="stats")


def render_db_error(detail):
    body = f"""<main class="page">
<h1 class="section-h">Cannot open the local Messages database</h1>
<p>This app reads <code>{html.escape(DB_PATH)}</code> on this Mac. macOS blocks that unless the app running this terminal has Full Disk Access.</p>
<p class="section-sub">System Settings → Privacy &amp; Security → Full Disk Access → enable Cursor (or Terminal), then restart the server.</p>
<p class="muted">{html.escape(detail)}</p>
</main>"""
    return page("Messages", body)
