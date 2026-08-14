"""Aggregates for the stats page. Rendering lives in render.py."""

from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

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
    """(month, received, sent) for every month with traffic."""
    rows = conn.execute(
        f"""SELECT strftime('%Y-%m', datetime(m.date/1000000000+978307200,'unixepoch','localtime')) as ym,
                   sum(m.is_from_me = 0) as recv, sum(m.is_from_me = 1) as sent
            FROM message m WHERE {REACTION_EXCLUDE_SQL}
            GROUP BY ym ORDER BY ym"""
    ).fetchall()
    return [(r["ym"], r["recv"] or 0, r["sent"] or 0) for r in rows if r["ym"]]


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
ERA_MIN_MSGS = 250


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


def trim_partial_month(people, months):
    """Drop the month in progress. Its counts are incomplete, so a share taken
    from it reads as a false peak."""
    now = datetime.now()
    if months and months[-1] == f"{now.year:04d}-{now.month:02d}":
        return [dict(p, values=p["values"][:-1]) for p in people], months[:-1]
    return people, months


def people_drift(people, months, n=DRIFT_PEOPLE, window=DRIFT_WINDOW_MONTHS):
    """Who gained and who lost your attention, by share of inbound messages.

    Share, not raw count: total inbound grew about 4x over the span, so a person
    with flat volume is in fact losing ground. Returns (rising, faded, months),
    both lists ordered from the largest gain to the largest loss.
    """
    people, months = trim_partial_month(people, months)
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


def people_eras(people, months, min_total=ERA_MIN_MSGS):
    """Every notable person across all history, as a share of what you received
    each month, ordered by the month they held the largest share.

    Share, not raw count, for the same reason the drift view uses it: total
    inbound grew about 4x over the span. The order turns the list into a ladder
    of eras, so the oldest peak is first. Returns (people, months).
    """
    people, months = trim_partial_month(people, months)
    if not months:
        return [], []
    totals = [sum(p["values"][i] for p in people) or 1 for i in range(len(months))]
    out = []
    for person in people:
        values = person["values"]
        total = sum(values)
        if total < min_total:
            continue
        shares = [values[i] / totals[i] for i in range(len(months))]
        peak_i = max(range(len(months)), key=lambda i: shares[i])
        out.append(
            dict(
                person,
                values=values,
                total=total,
                shares=shares,
                peak_i=peak_i,
                peak_ym=months[peak_i],
                peak_share=shares[peak_i],
            )
        )
    out.sort(key=lambda p: (p["peak_i"], -p["total"]))
    return out, months


SESSION_GAP_SECONDS = 4 * 3600
REPLY_MAX_SECONDS = 24 * 3600
LATENCY_MIN_REPLIES = 40
LATENCY_MIN_MONTH_REPLIES = 5
LATENCY_PEOPLE = 10
INITIATION_MIN_SESSIONS = 25
INITIATION_PEOPLE = 14
REACTION_MIN_MSGS = 150
REACTION_PEOPLE = 12


def dm_streams(conn, chat_per=None):
    """Every one-to-one thread as one time-ordered list of (unix seconds,
    is_from_me).

    Group chats are left out on purpose. With three people in the room a reply
    gap and an opener have no single owner. A person who moved between SMS and
    iMessage owns two chat rows, so the events are keyed by resolved name and
    stay in date order across both.
    """
    participants = defaultdict(list)
    for r in conn.execute(
        """SELECT chj.chat_id as chat_id, h.id as handle
           FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id"""
    ):
        participants[r["chat_id"]].append(r["handle"])
    dm_handle = {c: hs[0] for c, hs in participants.items() if len(hs) == 1}

    streams = {}
    handle_totals = defaultdict(Counter)
    for r in conn.execute(
        f"""SELECT cmj.chat_id as chat_id, m.date as d, m.is_from_me as me
            FROM message m JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE {REACTION_EXCLUDE_SQL} ORDER BY m.date"""
    ):
        handle = dm_handle.get(r["chat_id"])
        if handle is None:
            continue
        name = resolve_contact(handle) or handle
        handle_totals[name][handle] += 1
        stream = streams.get(name)
        if stream is None:
            stream = streams[name] = {"name": name, "events": []}
        stream["events"].append((r["d"] / 1_000_000_000 + APPLE_EPOCH, r["me"]))

    chat_per = chat_per or best_chat_per_handle(conn)
    for name, stream in streams.items():
        handle = handle_totals[name].most_common(1)[0][0]
        stream["handle"] = handle
        stream["chat_id"] = chat_per.get(handle)
        stream["sent"] = sum(me for _, me in stream["events"])
        stream["received"] = len(stream["events"]) - stream["sent"]
    return streams


def sessions(events, gap=SESSION_GAP_SECONDS):
    """(start seconds, opener is_from_me) for each burst of talk. A silence
    longer than the gap closes one session and opens the next."""
    out = []
    prev = None
    for ts, me in events:
        if prev is None or ts - prev > gap:
            out.append((ts, me))
        prev = ts
    return out


def reply_gaps(events, cap=REPLY_MAX_SECONDS):
    """(seconds, replier is_from_me) for each change of turn.

    The clock starts on the first unanswered message of a run, not the last, so
    a burst of five texts counts the wait the sender really had. A turn that
    changes after more than the cap is a new conversation and not an answer, so
    it is dropped.
    """
    out = []
    pending_ts = pending_me = None
    for ts, me in events:
        if pending_me is None:
            pending_ts, pending_me = ts, me
        elif me != pending_me:
            if ts - pending_ts <= cap:
                out.append((ts, ts - pending_ts, me))
            pending_ts, pending_me = ts, me
    return out


def monthly_medians(gaps, month_index):
    """(month index, median seconds) for each month with enough replies to
    trust. A month below the bar is left out, so the line skips it."""
    by_month = defaultdict(list)
    for ts, secs, _ in gaps:
        by_month[datetime.fromtimestamp(ts).strftime("%Y-%m")].append(secs)
    points = [
        (month_index[ym], median(vals))
        for ym, vals in by_month.items()
        if len(vals) >= LATENCY_MIN_MONTH_REPLIES and ym in month_index
    ]
    points.sort()
    return points


def reply_latency(streams, months, n=LATENCY_PEOPLE):
    """How long each side takes to answer the other, month by month.

    Volume says how much you talk. This says how fast you turn around, which
    moves on its own: a thread can hold its message count while the median
    answer slides from minutes to hours.
    """
    month_index = {ym: i for i, ym in enumerate(months)}
    out = []
    for stream in streams.values():
        gaps = reply_gaps(stream["events"])
        mine = [g for g in gaps if g[2]]
        theirs = [g for g in gaps if not g[2]]
        if min(len(mine), len(theirs)) < LATENCY_MIN_REPLIES:
            continue
        out.append(
            {
                "name": stream["name"],
                "handle": stream["handle"],
                "chat_id": stream["chat_id"],
                "total": len(stream["events"]),
                "my_median": median(g[1] for g in mine),
                "their_median": median(g[1] for g in theirs),
                "my_replies": len(mine),
                "their_replies": len(theirs),
                "my_points": monthly_medians(mine, month_index),
                "their_points": monthly_medians(theirs, month_index),
            }
        )
    out.sort(key=lambda p: -p["total"])
    out = out[:n]
    out.sort(key=lambda p: p["my_median"])
    return out


def initiation(streams, n=INITIATION_PEOPLE):
    """Who speaks first, year by year.

    A share near 50% is a thread both people open. A share walking toward 0 or
    100 is one person carrying it. Returns (people, years).
    """
    out = []
    years_seen = set()
    for stream in streams.values():
        opens = sessions(stream["events"])
        if len(opens) < INITIATION_MIN_SESSIONS:
            continue
        by_year = defaultdict(lambda: [0, 0])
        for ts, me in opens:
            slot = by_year[datetime.fromtimestamp(ts).strftime("%Y")]
            slot[0] += me
            slot[1] += 1
        years_seen.update(by_year)
        mine = sum(me for _, me in opens)
        out.append(
            {
                "name": stream["name"],
                "handle": stream["handle"],
                "chat_id": stream["chat_id"],
                "years": {y: tuple(v) for y, v in by_year.items()},
                "mine": mine,
                "total": len(opens),
                "share": mine / len(opens),
            }
        )
    out.sort(key=lambda p: -p["total"])
    out = out[:n]
    out.sort(key=lambda p: -p["share"])
    return out, sorted(years_seen)


def effective_people(people, months):
    """(month, effective number of people) using the inverse Simpson index.

    One person holding everything gives 1. Ten people at an even tenth give 10.
    It reads as the width of your attention, which no count on this page shows:
    volume can climb while the circle narrows.
    """
    people, months = trim_partial_month(people, months)
    out = []
    for i, ym in enumerate(months):
        total = sum(p["values"][i] for p in people)
        if not total:
            continue
        concentration = sum((p["values"][i] / total) ** 2 for p in people)
        out.append((ym, 1 / concentration))
    return out


def reaction_stats(conn, streams, min_msgs=REACTION_MIN_MSGS, n=REACTION_PEOPLE):
    """Tapbacks traded in one-to-one threads.

    A later 3xxx row for the same (target, reactor) pair means the tapback was
    taken back, so the rows are replayed in date order rather than counted.
    Only one-to-one threads count, so the reactor is never ambiguous.
    """
    participants = defaultdict(list)
    for r in conn.execute(
        """SELECT chj.chat_id as chat_id, h.id as handle
           FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id"""
    ):
        participants[r["chat_id"]].append(r["handle"])
    dm_handle = {c: hs[0] for c, hs in participants.items() if len(hs) == 1}

    state = {}
    for r in conn.execute(
        """SELECT r.associated_message_guid as g, r.associated_message_type as t,
                  r.associated_message_emoji as emoji, r.is_from_me as reactor_me,
                  tgt.is_from_me as target_me, cmj.chat_id as chat_id
           FROM message r
           JOIN message tgt
             ON tgt.guid = substr(r.associated_message_guid,
                                  instr(r.associated_message_guid, '/') + 1)
           JOIN chat_message_join cmj ON cmj.message_id = tgt.ROWID
           WHERE r.associated_message_type BETWEEN 2000 AND 3999
           ORDER BY r.date"""
    ):
        handle = dm_handle.get(r["chat_id"])
        if handle is None:
            continue
        key = (r["g"], r["reactor_me"], r["chat_id"])
        if r["t"] >= 3000:
            state.pop(key, None)
        else:
            state[key] = (handle, r["t"], r["emoji"], r["reactor_me"], r["target_me"])

    got = defaultdict(Counter)
    gave = defaultdict(Counter)
    for handle, kind, emoji, reactor_me, target_me in state.values():
        name = resolve_contact(handle) or handle
        token = emoji if kind == 2006 and emoji else kind
        if reactor_me and not target_me:
            gave[name][token] += 1
        elif not reactor_me and target_me:
            got[name][token] += 1

    out = []
    for name, stream in streams.items():
        if stream["sent"] < min_msgs:
            continue
        mine, theirs = got[name], gave[name]
        if not sum(mine.values()):
            continue
        out.append(
            {
                "name": name,
                "handle": stream["handle"],
                "chat_id": stream["chat_id"],
                "got": mine,
                "gave": theirs,
                "count": sum(mine.values()),
                "rate": sum(mine.values()) / stream["sent"],
                "laughs": mine.get(2003, 0),
                "laugh_rate": mine.get(2003, 0) / stream["sent"],
                "my_msgs": stream["sent"],
                "their_msgs": stream["received"],
                "gave_count": sum(theirs.values()),
                "gave_rate": sum(theirs.values()) / (stream["received"] or 1),
            }
        )
    out.sort(key=lambda p: -p["rate"])
    return out[:n]


def year_owners(people, months):
    """The person who held the largest share of your inbound in each year."""
    people, months = trim_partial_month(people, months)
    out = []
    for year in sorted({m[:4] for m in months}):
        idx = [i for i, m in enumerate(months) if m.startswith(year)]
        year_total = sum(sum(p["values"][i] for i in idx) for p in people)
        if not year_total:
            continue
        best = max(people, key=lambda p: sum(p["values"][i] for i in idx))
        count = sum(best["values"][i] for i in idx)
        out.append(
            {
                "year": year,
                "name": best["name"],
                "handle": best["handle"],
                "chat_id": best["chat_id"],
                "count": count,
                "share": count / year_total,
            }
        )
    return out
