"""Aggregates for the stats page. Rendering lives in render.py."""

from collections import Counter, defaultdict
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


PEAK_PEOPLE = 8
DRIFT_WINDOW_MONTHS = 12
DRIFT_PEOPLE = 5
DRIFT_MIN_MSGS = 150


def monthly_by_person(conn, chat_per=None):
    """Monthly inbound counts for every contact, zero-filled, newest month last.

    Apple stores a phone number and an Apple ID as two handles, so a query
    grouped by handle splits one person in two. Group by resolved name instead.
    """
    rows = conn.execute(
        f"""SELECT h.id as handle,
                   strftime('%Y-%m', datetime(m.date/1000000000+978307200,'unixepoch','localtime')) as ym,
                   count(*) as c
            FROM message m JOIN handle h ON h.ROWID = m.handle_id
            WHERE m.is_from_me=0 AND {REACTION_EXCLUDE_SQL}
            GROUP BY h.id, ym"""
    ).fetchall()
    by_person = defaultdict(Counter)
    handle_totals = defaultdict(Counter)
    months_seen = []
    for r in rows:
        if not r["ym"]:
            continue
        name = resolve_contact(r["handle"]) or r["handle"]
        by_person[name][r["ym"]] += r["c"]
        handle_totals[name][r["handle"]] += r["c"]
        months_seen.append(r["ym"])
    if not months_seen:
        return [], []
    months = month_range(min(months_seen), max(months_seen))
    chat_per = chat_per or best_chat_per_handle(conn)
    people = []
    for name, counts in by_person.items():
        handle = handle_totals[name].most_common(1)[0][0]
        values = [counts.get(ym, 0) for ym in months]
        people.append(
            {
                "handle": handle,
                "name": name,
                "chat_id": chat_per.get(handle),
                "values": values,
                "total": sum(values),
            }
        )
    people.sort(key=lambda p: -p["total"])
    return people, months


def people_over_time(people, months, n=PEAK_PEOPLE):
    """The n most-texted people, each with their busiest month over all history."""
    series = []
    for person in people[:n]:
        peak_i = max(range(len(months)), key=lambda i: person["values"][i])
        series.append(
            dict(person, peak_ym=months[peak_i], peak_c=person["values"][peak_i])
        )
    return series


def people_drift(people, months, n=DRIFT_PEOPLE, window=DRIFT_WINDOW_MONTHS):
    """Who gained and who lost your attention, by share of inbound messages.

    Share, not raw count: total inbound grew about 4x over the span, so a person
    with flat volume is in fact losing ground. Returns (rising, faded, months),
    both lists ordered from the largest gain to the largest loss.
    """
    now = datetime.now()
    trim = 1 if months and months[-1] == f"{now.year:04d}-{now.month:02d}" else 0
    if trim:
        months = months[:-1]
        people = [dict(p, values=p["values"][:-1]) for p in people]
    start = len(months) - 2 * window
    if start < 0:
        return [], [], []

    months = months[start:]
    prev_all = sum(sum(p["values"][start : start + window]) for p in people) or 1
    now_all = sum(sum(p["values"][start + window :]) for p in people) or 1

    ranked = []
    for person in people:
        values = person["values"][start:]
        prev, recent = sum(values[:window]), sum(values[window:])
        if prev + recent < DRIFT_MIN_MSGS:
            continue
        share_prev, share_now = prev / prev_all, recent / now_all
        ranked.append(
            dict(
                person,
                values=values,
                prev=prev,
                recent=recent,
                share_prev=share_prev,
                share_now=share_now,
                drift=share_now - share_prev,
            )
        )
    ranked.sort(key=lambda p: -p["drift"])
    rising = [p for p in ranked if p["drift"] > 0][:n]
    faded = [p for p in ranked if p["drift"] < 0][-n:]
    return rising, faded, months
