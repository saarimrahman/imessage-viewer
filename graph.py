"""Who shares a group chat with whom. The Circles page is a map of that overlap.

A group chat is any thread with two or more other people after contact cards
collapse SMS and iMessage handles onto one person. One-to-one chats never
become circles; they only supply a link back to a direct thread.
"""

from collections import defaultdict

from contacts import AVATAR_COLORS, avatar_id, person_key, resolve_contact
from db import REACTION_EXCLUDE_SQL

MIN_GROUP_MESSAGES = 8
MAX_GROUP_PEOPLE = 24
MAX_GROUPS = 48
MAX_PEOPLE = 90


def _color_for(name):
    return AVATAR_COLORS[sum(ord(c) for c in (name or "")) % len(AVATAR_COLORS)]


def _short_label(display_name, names):
    if display_name and display_name.strip():
        return display_name.strip()
    firsts = [(n or "?").split()[0] for n in names]
    if len(firsts) <= 3:
        return ", ".join(firsts)
    return f"{firsts[0]}, {firsts[1]} +{len(firsts) - 2}"


def _collapse_members(handles):
    """One record per person. The handle that still has a contact photo wins,
    then the one with a resolved name, then the first seen."""
    by_key = {}
    for handle in handles:
        if not handle:
            continue
        key = person_key(handle)
        if not key:
            continue
        name = resolve_contact(handle) or handle
        avatar = avatar_id(handle)
        cur = by_key.get(key)
        candidate = {
            "key": key,
            "handle": handle,
            "name": name,
            "avatar": avatar,
            "color": _color_for(name),
        }
        if cur is None:
            by_key[key] = candidate
            continue
        better_avatar = avatar and not cur["avatar"]
        better_name = (not cur["avatar"] and not avatar
                       and resolve_contact(handle) and not resolve_contact(cur["handle"]))
        if better_avatar or better_name:
            by_key[key] = candidate
    return list(by_key.values())


def build_circles(raw_groups, *, dm_by_key=None, solo_count=0):
    """Turn resolved group memberships into the JSON the Circles page draws.

    raw_groups: list of {chat_id, display_name, messages, members} where
    members is the output of _collapse_members (or the same shape).
    dm_by_key: person_key -> {chat_id, handle, messages}.
    """
    dm_by_key = dm_by_key or {}
    kept = []
    for raw in raw_groups:
        members = []
        seen = set()
        for member in raw.get("members") or []:
            key = member.get("key")
            if not key or key in seen:
                continue
            seen.add(key)
            members.append(member)
        if len(members) < 2 or len(members) > MAX_GROUP_PEOPLE:
            continue
        messages = int(raw.get("messages") or 0)
        if messages < MIN_GROUP_MESSAGES:
            continue
        kept.append(
            {
                "chat_id": raw["chat_id"],
                "display_name": raw.get("display_name") or "",
                "messages": messages,
                "members": members,
            }
        )

    merged = {}
    for group in kept:
        signature = frozenset(m["key"] for m in group["members"])
        prev = merged.get(signature)
        if prev is None:
            merged[signature] = dict(group)
            continue
        if group["display_name"] and not prev["display_name"]:
            prev["display_name"] = group["display_name"]
            prev["chat_id"] = group["chat_id"]
        elif not prev["display_name"] and group["messages"] > prev["messages"]:
            prev["chat_id"] = group["chat_id"]
        prev["messages"] += group["messages"]

    groups = sorted(merged.values(), key=lambda g: -g["messages"])[:MAX_GROUPS]

    people_rank = defaultdict(lambda: {"groups": 0, "messages": 0, "member": None})
    for group in groups:
        for member in group["members"]:
            slot = people_rank[member["key"]]
            slot["groups"] += 1
            slot["messages"] += group["messages"]
            slot["member"] = member
    for key, slot in people_rank.items():
        dm = dm_by_key.get(key)
        if dm:
            slot["messages"] += dm.get("messages") or 0

    ranked_keys = sorted(
        people_rank,
        key=lambda k: (-people_rank[k]["groups"], -people_rank[k]["messages"], k),
    )[:MAX_PEOPLE]
    allowed = set(ranked_keys)

    people_out = []
    id_of = {}
    by_id = {}
    for i, key in enumerate(ranked_keys):
        member = people_rank[key]["member"]
        pid = f"p{i}"
        id_of[key] = pid
        dm = dm_by_key.get(key) or {}
        person = {
            "id": pid,
            "name": member["name"],
            "handle": member["handle"],
            "avatar": member.get("avatar"),
            "color": member.get("color") or _color_for(member["name"]),
            "chat_id": dm.get("chat_id"),
            "messages": people_rank[key]["messages"],
            "group_ids": [],
        }
        people_out.append(person)
        by_id[pid] = person

    groups_out = []
    for group in groups:
        members = [m for m in group["members"] if m["key"] in allowed]
        if len(members) < 2:
            continue
        gid = f"g{len(groups_out)}"
        names = [m["name"] for m in members]
        member_ids = [id_of[m["key"]] for m in members]
        groups_out.append(
            {
                "id": gid,
                "name": _short_label(group["display_name"], names),
                "chat_id": group["chat_id"],
                "messages": group["messages"],
                "member_ids": member_ids,
            }
        )
        for pid in member_ids:
            by_id[pid]["group_ids"].append(gid)

    people_out = [p for p in people_out if p["group_ids"]]
    present = {p["id"] for p in people_out}
    for group in groups_out:
        group["member_ids"] = [pid for pid in group["member_ids"] if pid in present]
    groups_out = [g for g in groups_out if len(g["member_ids"]) >= 2]
    used_groups = {g["id"] for g in groups_out}
    for person in people_out:
        person["group_ids"] = [gid for gid in person["group_ids"] if gid in used_groups]
    people_out = [p for p in people_out if p["group_ids"]]

    return {
        "people": people_out,
        "groups": groups_out,
        "solo": int(solo_count),
    }


def load_circle_graph(conn):
    """Read chat.db and return the payload `build_circles` already shaped."""
    handles_by_chat = defaultdict(list)
    for row in conn.execute(
        """SELECT chj.chat_id as chat_id, h.id as handle
           FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id"""
    ):
        handles_by_chat[row["chat_id"]].append(row["handle"])

    chats = {
        row["id"]: row["display_name"] or ""
        for row in conn.execute("SELECT ROWID as id, display_name FROM chat")
    }
    counts = {
        row["chat_id"]: row["c"]
        for row in conn.execute(
            f"""SELECT cmj.chat_id as chat_id, count(*) as c
                FROM chat_message_join cmj
                JOIN message m ON m.ROWID = cmj.message_id
                WHERE {REACTION_EXCLUDE_SQL}
                GROUP BY cmj.chat_id"""
        )
    }

    raw_groups = []
    dm_by_key = {}
    in_group = set()
    in_dm = set()
    for chat_id, handles in handles_by_chat.items():
        members = _collapse_members(handles)
        if not members:
            continue
        messages = counts.get(chat_id, 0)
        if len(members) == 1:
            member = members[0]
            in_dm.add(member["key"])
            prev = dm_by_key.get(member["key"])
            if prev is None or messages > prev["messages"]:
                dm_by_key[member["key"]] = {
                    "chat_id": chat_id,
                    "handle": member["handle"],
                    "messages": messages,
                }
            continue
        in_group.update(m["key"] for m in members)
        # Named groups keep their title. Unnamed ones get a short label later
        # from member first names, not the long comma-separated chat_label list.
        display = chats.get(chat_id) or ""
        raw_groups.append(
            {
                "chat_id": chat_id,
                "display_name": display,
                "messages": messages,
                "members": members,
            }
        )

    solo = len(in_dm - in_group)
    return build_circles(raw_groups, dm_by_key=dm_by_key, solo_count=solo)
