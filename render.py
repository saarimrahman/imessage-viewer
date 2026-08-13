"""HTML and CSS for every page."""

import html
import json
import os
from datetime import datetime, timedelta
from urllib.parse import quote

from config import DB_PATH, PAGE_SIZE, SEARCH_LIMIT, START_NEWEST, load_prefs
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


def format_day_label(day_str):
    d = datetime.strptime(day_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    return d.strftime("%A, %B %-d, %Y")


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
            cell_html_parts.append(
                f'<div class="{cls}" data-day="{day_str}" style="background:{color}" title="{title}"{onclick}></div>'
            )

    return f"""<div class="heatmap-wrap"><div class="heatmap-scroll">
<div class="heatmap-months">{month_html}</div>
<div class="heatmap-grid">{''.join(cell_html_parts)}</div>
</div></div>"""


PAGE_CSS = """
body { font-family: -apple-system, sans-serif; background: #f2f2f7; margin: 0; color: #1c1c1e; }
header { background: #fff; padding: 12px 20px; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 10; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
header a { color: #007aff; text-decoration: none; }
header input[type=date] { border: 1px solid #ccc; border-radius: 6px; padding: 5px 8px; font-size: 13px; }
main { max-width: 900px; margin: 0 auto; padding: 16px; }
input#filter { width: 100%; padding: 10px; font-size: 15px; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; margin-bottom: 12px; }
.prefs { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 16px; font-size: 13px; color: #555; margin: -6px 0 12px; }
.prefs-label { color: #898781; margin-right: 4px; }
.prefs label { display: flex; align-items: center; gap: 6px; cursor: pointer; min-height: 40px; }
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
.searchbtn { padding: 6px 12px; border: none; border-radius: 6px; background: #007aff; color: #fff; font-size: 13px; cursor: pointer; transition-property: transform; transition-duration: 0.12s; }
.searchbtn:active { transform: scale(0.96); }
.searchform { display: flex; gap: 6px; align-items: center; }
header .searchform { margin-left: auto; }
mark { background: #ffe08a; color: inherit; padding: 0 1px; border-radius: 2px; }
.sr-note { font-size: 13px; color: #898781; margin: 0 0 12px; }
.chat-hits { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.chat-hit { display: flex; align-items: center; gap: 8px; background: #fff; border-radius: 20px; padding: 6px 12px 6px 6px; text-decoration: none; color: inherit; font-size: 13px; font-weight: 600; min-height: 40px; box-sizing: border-box; }
.chat-hit:hover { background: #f5f8ff; }
.chat-hit .avatar { width: 28px; height: 28px; font-size: 12px; }
.tile-chat { font-size: 11px; color: #fff; background: linear-gradient(transparent, rgba(0,0,0,0.55)); position: absolute; left: 0; right: 0; bottom: 0; padding: 14px 6px 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; opacity: 0; transition-property: opacity; transition-duration: 0.15s; }
.tile:hover .tile-chat { opacity: 1; }
.sg-band { opacity: 0.88; transition-property: opacity; transition-duration: 0.15s; }
.sg-band.dim { opacity: 0.12; }
.sg-band.hot { opacity: 1; }
.sg-legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 10px; }
.sg-leg { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #52514e; cursor: default; min-height: 28px; }
.sg-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sg-tip { white-space: pre; }
.mediagrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 3px; }
.tile { position: relative; display: block; aspect-ratio: 1; overflow: hidden; border-radius: 4px; background: #e5e5ea; outline: 1px solid rgba(0,0,0,0.1); }
.tile img, .tile video { width: 100%; height: 100%; object-fit: cover; }
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
body.mediapage main { max-width: 1100px; padding-right: 56px; }
body.chatpage main { padding-right: 56px; }
.searchresult { display: block; background: #fff; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; text-decoration: none; color: inherit; }
.searchresult:hover { background: #f5f8ff; }
.sr-meta { font-size: 11px; color: #888; margin-bottom: 4px; }
.sr-text { font-size: 14px; white-space: pre-wrap; }
.avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 14px; font-weight: 600; margin-right: 4px; flex-shrink: 0; object-fit: cover; }
tr.chatrow td.name { display: flex; align-items: center; gap: 10px; }
.heatmap-wrap { background: #fff; border-radius: 8px; padding: 10px 14px 12px; margin-bottom: 12px; overflow-x: auto; }
body.chatpage .heatmap-wrap { background: #f2f2f7; }
.heatmap-scroll { width: max-content; }
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
.lightbox-content img { outline: 1px solid rgba(255,255,255,0.1); }
.lightbox-close { position: absolute; top: 18px; right: 24px; color: #fff; font-size: 30px; cursor: pointer; background: none; border: none; line-height: 1; padding: 4px 10px; }
.lightbox-nav { position: absolute; top: 50%; transform: translateY(-50%); color: #fff; font-size: 26px; background: rgba(255,255,255,0.1); border: none; width: 44px; height: 44px; border-radius: 50%; cursor: pointer; }
.lightbox-nav:hover { background: rgba(255,255,255,0.2); }
.lightbox-prev { left: 18px; }
.lightbox-next { right: 18px; }
.lightbox-chat { position: absolute; top: 16px; left: 20px; color: #fff; background: rgba(255,255,255,0.14); border: none; border-radius: 20px; padding: 10px 16px; font-size: 14px; min-height: 40px; cursor: pointer; transition-property: background, transform; transition-duration: 0.15s; }
.lightbox-chat:hover { background: rgba(255,255,255,0.22); }
.lightbox-chat:active { transform: scale(0.96); }
.lightbox-chat[hidden] { display: none; }
.ctx-menu { position: fixed; background: #fff; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); padding: 4px; z-index: 1100; min-width: 150px; display: none; }
.ctx-menu.open { display: block; }
.ctx-menu-item { padding: 8px 12px; font-size: 13px; cursor: pointer; border-radius: 5px; }
.ctx-menu-item:hover { background: #f2f2f7; }
.wordcloud { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 10px; line-height: 1.35; padding: 4px 0 8px; }
.wc-word { color: #0b0b0b; text-decoration: none; font-weight: 600; }
.wc-word:hover { color: #007aff; }
.vchips { display: flex; flex-wrap: wrap; gap: 6px; }
.vchip { display: inline-flex; align-items: baseline; gap: 6px; background: #f2f2f7; border-radius: 16px; padding: 8px 12px; font-size: 13px; min-height: 40px; box-sizing: border-box; }
.vchip .n { font-size: 11px; color: #898781; font-variant-numeric: tabular-nums; }
.vchip .lift { font-size: 11px; color: #0b84ff; font-variant-numeric: tabular-nums; }
.voice-person { background: #fff; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; }
.voice-person-h { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
.voice-person-h a { color: inherit; text-decoration: none; font-weight: 600; font-size: 14px; }
.voice-person-h a:hover { color: #007aff; }
.voice-person-h .n { font-size: 12px; color: #898781; font-variant-numeric: tabular-nums; }
"""


SORT_OPTIONS = {
    "recent": "Most recent activity",
    "count": "Most messages",
    "name": "Name (A-Z)",
    "oldest": "Oldest conversation",
}


def search_form_html(action, query="", placeholder="Search messages...", wide=False):
    width = "240px" if wide else "180px"
    return (
        f'<form class="searchform" method="get" action="{html.escape(action)}">'
        f'<input class="searchbox" type="search" name="q" value="{html.escape(query or "")}" '
        f'placeholder="{html.escape(placeholder)}" style="width:{width};" autocomplete="off">'
        f'<button class="searchbtn" type="submit">Search</button></form>'
    )


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
    start = load_prefs()["start"]
    oldest_on = " checked" if start != START_NEWEST else ""
    newest_on = " checked" if start == START_NEWEST else ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Messages</title><style>{PAGE_CSS}</style></head><body>
<header><b>Messages</b> &middot; {len(rows)} conversations
<a href="/media">Photos</a>
<a href="/stats">Stats</a>
{search_form_html("/search", placeholder="Search all messages...", wide=True)}
<select id="sortSelect" onchange="location.href='/?sort='+this.value" style="padding:6px 8px; border:1px solid #ccc; border-radius:6px;">{opts_html}</select>
</header>
<main>
<input id="filter" placeholder="Filter conversations..." oninput="filterRows()">
<label style="display:flex; align-items:center; gap:6px; font-size:13px; color:#555; margin:-6px 0 12px;">
<input type="checkbox" id="knownOnly" onchange="filterRows()"> Known contacts only
</label>
<div class="prefs">
<span class="prefs-label">Open chats and photos</span>
<label><input type="radio" name="start" value="oldest"{oldest_on} onchange="setStart(this.value)"> At the oldest</label>
<label><input type="radio" name="start" value="newest"{newest_on} onchange="setStart(this.value)"> At the newest</label>
</div>
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
function setStart(start) {{
  const sort = document.getElementById('sortSelect').value;
  location.href = '/?sort=' + encodeURIComponent(sort) + '&start=' + encodeURIComponent(start);
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
{search_form_html(f"/chat/{chat_id}/search", placeholder="Search this conversation...")}
<span style="margin-left:auto; display:flex; align-items:center; gap:6px;">
<label for="datepicker" style="font-size:13px; color:#666;">Jump to date</label>
<input type="date" id="datepicker" value="{cur_date}" min="{min_date}" max="{max_date}">
</span>
</header>
<main>
{heatmap_html}
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

(function() {{
  const wrap = document.querySelector('.heatmap-wrap');
  const cell = document.querySelector('.hcell[data-day="' + document.getElementById('datepicker').value + '"]');
  if (!wrap || !cell) return;
  const r = cell.getBoundingClientRect();
  const w = wrap.getBoundingClientRect();
  wrap.scrollLeft += r.left + r.width / 2 - (w.left + w.width / 2);
}})();

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


def lightbox_script(chat_id=None):
    return f"""
(function() {{
  const defaultChatId = {json.dumps(chat_id)};
  const SELECTOR = '.tile, img.att, video.att';

  function mediaInfo(el) {{
    const inner = el.classList.contains('tile') ? el.querySelector('img,video') : el;
    if (!inner) return null;
    return {{ src: inner.getAttribute('data-full-src') || el.getAttribute('data-full-src') || inner.getAttribute('src'), isVideo: inner.tagName === 'VIDEO', msgId: el.dataset.msgId || inner.dataset.msgId, chatId: el.dataset.chatId || inner.dataset.chatId || defaultChatId, node: el }};
  }}

  function collectItems() {{
    return Array.from(document.querySelectorAll(SELECTOR)).map(mediaInfo).filter(Boolean);
  }}

  const overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  overlay.innerHTML =
    '<button type="button" class="lightbox-chat">View in chat</button>' +
    '<button class="lightbox-close" aria-label="Close">&times;</button>' +
    '<button class="lightbox-nav lightbox-prev" aria-label="Previous">&#8249;</button>' +
    '<div class="lightbox-content"></div>' +
    '<button class="lightbox-nav lightbox-next" aria-label="Next">&#8250;</button>';
  document.body.appendChild(overlay);
  const content = overlay.querySelector('.lightbox-content');
  const chatBtn = overlay.querySelector('.lightbox-chat');

  const menu = document.createElement('div');
  menu.className = 'ctx-menu';
  menu.innerHTML = '<div class="ctx-menu-item" id="ctxShowInChat">Show in chat</div>';
  document.body.appendChild(menu);
  let menuMsgId = null;
  let menuChatId = null;
  let items = [];
  let curIndex = 0;

  function chatUrl(chat, msg) {{
    if (!chat || !msg) return null;
    return '/chat/' + chat + '?around=' + msg;
  }}

  function render(i) {{
    curIndex = (i + items.length) % items.length;
    const it = items[curIndex];
    content.innerHTML = it.isVideo
      ? `<video src="${{it.src}}" controls autoplay></video>`
      : `<img src="${{it.src}}">`;
    chatBtn.hidden = !chatUrl(it.chatId, it.msgId);
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
  chatBtn.addEventListener('click', () => {{
    const it = items[curIndex];
    const url = it && chatUrl(it.chatId, it.msgId);
    if (url) location.href = url;
  }});
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
    const info = mediaInfo(el);
    menuMsgId = info.msgId;
    menuChatId = info.chatId;
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.classList.add('open');
  }});

  document.getElementById('ctxShowInChat').addEventListener('click', () => {{
    const url = chatUrl(menuChatId, menuMsgId);
    if (url) location.href = url;
  }});
  document.addEventListener('click', e => {{
    if (!menu.contains(e.target)) menu.classList.remove('open');
  }});
}})();
"""


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
        return empty_msg, ""
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
    section_html, rail_html = media_sections_html(
        items, '<p style="color:#999">No media in this conversation.</p>'
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Media: {html.escape(title)}</title><style>{PAGE_CSS}</style></head><body class="mediapage">
<header><a href="/chat/{chat_id}">&larr; {html.escape(title)}</a> <b>Media</b> &middot; {len(items)} items
<a href="/media">All photos</a></header>
<main>{section_html}</main>
{rail_html}
<script>{MEDIA_RAIL_SCRIPT}</script>
{media_start_script()}
<script>{lightbox_script(chat_id)}</script>
</body></html>"""


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

    section_html, rail_html = media_sections_html(
        items, '<p style="color:#999">No photos or videos yet.</p>'
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Photos</title><style>{PAGE_CSS}</style></head><body class="mediapage">
<header><a href="/">&larr; All conversations</a> <b>Photos</b> &middot; {len(items)} items
<a href="/search">Search</a>
<a href="/stats">Stats</a></header>
<main>{section_html}</main>
{rail_html}
<script>{MEDIA_RAIL_SCRIPT}</script>
{media_start_script()}
<script>{lightbox_script()}</script>
</body></html>"""


def render_indexing_page(query, chat_id=None):
    action = f"/chat/{chat_id}/search" if chat_id else "/search"
    back = f'<a href="/chat/{chat_id}">&larr; Conversation</a>' if chat_id else '<a href="/">&larr; All conversations</a>'
    err = f'<p style="color:#c00">{html.escape(index_error())}</p>' if index_error() else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="1">
<title>Search</title><style>{PAGE_CSS}</style></head><body>
<header>{back} {search_form_html(action, query, "Search messages...", wide=True)}</header>
<main>
<h2 class="section-h">Building search index</h2>
<p class="section-sub">One-time pass over your messages. This page refreshes on its own.</p>
{err}
</main>
</body></html>"""


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
            f'<a class="searchresult" href="/chat/{r["chat_id"]}?around={r["msg_id"]}">'
            f'<div class="sr-meta">{" &middot; ".join(meta_bits)}</div>'
            f'<div class="sr-text">{snippet_html(r["body"], phrases, terms)}</div></a>'
        )

    if q:
        shown = len(rows)
        total_note = f"{shown}+" if truncated else str(shown)
        word = "match" if shown == 1 and not truncated else "matches"
        status = f'{total_note} {word} for &ldquo;{html.escape(q)}&rdquo;'
        if broadened:
            status += " — no phrase-and-all-words hit, so this is anything that matched a word."
        status_html = f'<p style="color:#666">{status}</p>'
        if truncated:
            status_html += f'<p class="sr-note">Showing the first {SEARCH_LIMIT}.</p>'
        body = "".join(items) if items else '<p style="color:#999">No matches.</p>'
    else:
        status_html = (
            '<p class="sr-note">Quote a phrase for an exact run of words, like '
            "&ldquo;see you tomorrow&rdquo;. Other words match related forms "
            "(run / running) and prefixes, ranked by relevance.</p>"
        )
        body = ""

    if chat_id is not None:
        back = f'<a href="/chat/{chat_id}">&larr; {html.escape(chat_title)}</a>'
        action = f"/chat/{chat_id}/search"
        placeholder = "Search this conversation..."
        extra = f'<a href="/search?q={quote(q)}">Search all</a>' if q else '<a href="/search">Search all</a>'
        page_title = f"Search: {chat_title}"
    else:
        back = '<a href="/">&larr; All conversations</a>'
        action = "/search"
        placeholder = "Search all messages..."
        extra = '<a href="/media">Photos</a>'
        page_title = "Search"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(page_title)}</title><style>{PAGE_CSS}</style></head><body>
<header>
{back}
{extra}
{search_form_html(action, q, placeholder, wide=True)}
</header>
<main>
{status_html}
{conv_html}
{body}
</main>
</body></html>"""


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


def render_people_stream(series, months):
    if not series or not months:
        return "<p>No data.</p>"
    W, H = 900, 260
    pad_l, pad_r, pad_t, pad_b = 10, 10, 12, 24
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
            x_labels.append(f'<text x="{xs[i]:.1f}" y="{H - 4}" class="axislabel">{year}</text>')
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

    svg = f"""<div class="trendwrap"><svg viewBox="0 0 {W} {H}" class="trendsvg" id="streamsvg" preserveAspectRatio="none">
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
<p class="section-sub">{voice["msgs"]:,} texts you sent · {voice["avg_len"]} characters average. NLTK stopwords stripped. Click a word to search.</p>
<div class="panel">{render_word_cloud(voice.get("words") or [])}</div>
<h2 class="section-h">Phrases you reach for</h2>
<div class="vchips">{phrase_html}</div>
{people_block}"""


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
        lambda it: f"Friends since {it['first_dt'].year} &middot; {it['chat_count']} shared chat{'s' if it['chat_count'] != 1 else ''}",
    ) or '<p style="color:#999">No contacts cross the 1-year, active-in-90-days bar yet.</p>'

    fell_off_html = render_contact_cards(
        fell_off,
        lambda it: f"{it['gap_days']} days quiet &middot; last texted {it['last_dt'].strftime('%Y-%m-%d')}",
    ) or '<p style="color:#999">Nobody with 30+ messages has gone quiet for 180+ days. Good.</p>'

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
    ) or '<p style="color:#999">Not enough history to find a peak month yet.</p>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Stats</title><style>{PAGE_CSS}</style></head><body>
<header><a href="/">&larr; All conversations</a> <b>Stats</b>
<a href="/search">Search</a>
<a href="/media">Photos</a></header>
<main>
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
</main>
<script>{COUNTUP_SCRIPT}</script>
</body></html>"""


def render_db_error(detail):
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Messages</title><style>{PAGE_CSS}</style></head><body>
<header><b>Messages</b></header>
<main>
<h2 class="section-h">Cannot open the local Messages database</h2>
<p>This app reads <code>{html.escape(DB_PATH)}</code> on this Mac. macOS blocks that unless the app running this terminal has Full Disk Access.</p>
<p class="section-sub">System Settings → Privacy &amp; Security → Full Disk Access → enable Cursor (or Terminal), then restart the server.</p>
<p style="color:#666; font-size:13px">{html.escape(detail)}</p>
</main>
</body></html>"""
