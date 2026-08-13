"""Aggregates for the stats page. Rendering lives in render.py."""

from datetime import datetime

from config import APPLE_EPOCH
from contacts import resolve_contact
from db import REACTION_EXCLUDE_SQL


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


def month_range(start_ym, end_ym):
    y, m = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


STREAM_COLORS = ["#0b84ff", "#ff375f", "#34c759", "#af52de", "#ff9f0a", "#64d2ff", "#ff6482", "#30d158"]
STREAM_PEOPLE = 8


def people_over_time(conn, n=STREAM_PEOPLE, chat_per=None):
    """Monthly inbound counts for the n most-texted people, zero-filled so
    bands actually fade to nothing when someone drops out of your life."""
    top = conn.execute(
        f"""SELECT h.id as handle, count(*) as c
            FROM message m JOIN handle h ON h.ROWID = m.handle_id
            WHERE m.is_from_me=0 AND {REACTION_EXCLUDE_SQL}
            GROUP BY h.id ORDER BY c DESC LIMIT ?""",
        (n,),
    ).fetchall()
    if not top:
        return [], []
    handles = [r["handle"] for r in top]
    totals = {r["handle"]: r["c"] for r in top}
    qmarks = ",".join("?" * len(handles))
    rows = conn.execute(
        f"""SELECT h.id as handle,
                   strftime('%Y-%m', datetime(m.date/1000000000+978307200,'unixepoch','localtime')) as ym,
                   count(*) as c
            FROM message m JOIN handle h ON h.ROWID = m.handle_id
            WHERE m.is_from_me=0 AND {REACTION_EXCLUDE_SQL} AND h.id IN ({qmarks})
            GROUP BY h.id, ym""",
        handles,
    ).fetchall()
    by_handle = {}
    months_seen = []
    for r in rows:
        if r["ym"]:
            by_handle.setdefault(r["handle"], {})[r["ym"]] = r["c"]
            months_seen.append(r["ym"])
    if not months_seen:
        return [], []
    months = month_range(min(months_seen), max(months_seen))
    chat_per = chat_per or best_chat_per_handle(conn)
    series = []
    for handle in handles:
        counts = by_handle.get(handle, {})
        values = [counts.get(ym, 0) for ym in months]
        peak_i = max(range(len(values)), key=lambda i: values[i])
        series.append(
            {
                "handle": handle,
                "name": resolve_contact(handle) or handle,
                "chat_id": chat_per.get(handle),
                "values": values,
                "total": totals[handle],
                "peak_ym": months[peak_i],
                "peak_c": values[peak_i],
            }
        )
    return series, months
