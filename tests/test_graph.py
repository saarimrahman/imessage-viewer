import unittest

from graph import MAX_GROUPS, MIN_GROUP_MESSAGES, build_circles, _short_label
from render import NAV, json_script


def person(key, name=None):
    name = name or key
    return {
        "key": key,
        "handle": "+" + key,
        "name": name,
        "avatar": None,
        "color": "#007aff",
    }


def group(chat_id, members, messages=20, display_name=""):
    return {
        "chat_id": chat_id,
        "display_name": display_name,
        "messages": messages,
        "members": members,
    }


class ShortLabelTest(unittest.TestCase):
    def test_named_group_keeps_its_title(self):
        self.assertEqual(_short_label("Ski trip", ["Ada Lovelace", "Bob"]), "Ski trip")

    def test_unnamed_small_group_uses_first_names(self):
        self.assertEqual(_short_label("", ["Ada Lovelace", "Bob Jones"]), "Ada, Bob")

    def test_unnamed_large_group_counts_the_rest(self):
        names = ["Ada", "Bob", "Cara", "Dan"]
        self.assertEqual(_short_label(None, names), "Ada, Bob +2")


class BuildCirclesTest(unittest.TestCase):
    def test_one_group_of_two_people(self):
        data = build_circles([group(1, [person("a", "Ada"), person("b", "Bob")])])

        self.assertEqual(len(data["people"]), 2)
        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual(data["groups"][0]["name"], "Ada, Bob")
        self.assertEqual(set(data["groups"][0]["member_ids"]), {"p0", "p1"})

    def test_one_to_one_membership_is_not_a_circle(self):
        data = build_circles([group(1, [person("a", "Ada")])])

        self.assertEqual(data["people"], [])
        self.assertEqual(data["groups"], [])

    def test_quiet_groups_are_dropped(self):
        data = build_circles(
            [group(1, [person("a", "Ada"), person("b", "Bob")], messages=MIN_GROUP_MESSAGES - 1)]
        )

        self.assertEqual(data["groups"], [])

    def test_duplicate_keys_in_one_group_collapse(self):
        ada = person("a", "Ada")
        data = build_circles([group(1, [ada, dict(ada), person("b", "Bob")])])

        self.assertEqual(len(data["groups"][0]["member_ids"]), 2)

    def test_same_people_in_two_chats_merge(self):
        members = [person("a", "Ada"), person("b", "Bob")]
        data = build_circles(
            [
                group(1, members, messages=10),
                group(2, members, messages=15, display_name="Ski"),
            ]
        )

        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual(data["groups"][0]["name"], "Ski")
        self.assertEqual(data["groups"][0]["messages"], 25)
        self.assertEqual(data["groups"][0]["chat_id"], 2)

    def test_person_in_two_groups_is_a_bridge(self):
        ada = person("a", "Ada")
        data = build_circles(
            [
                group(1, [ada, person("b", "Bob")], display_name="Work"),
                group(2, [ada, person("c", "Cara")], display_name="Ski"),
            ]
        )

        ada_node = next(p for p in data["people"] if p["name"] == "Ada")
        self.assertEqual(len(ada_node["group_ids"]), 2)
        self.assertEqual({g["name"] for g in data["groups"]}, {"Work", "Ski"})

    def test_direct_chat_id_is_attached(self):
        data = build_circles(
            [group(1, [person("a", "Ada"), person("b", "Bob")])],
            dm_by_key={"a": {"chat_id": 9, "messages": 40}},
        )

        ada = next(p for p in data["people"] if p["name"] == "Ada")
        bob = next(p for p in data["people"] if p["name"] == "Bob")
        self.assertEqual(ada["chat_id"], 9)
        self.assertIsNone(bob["chat_id"])

    def test_direct_messages_count_once(self):
        data = build_circles(
            [
                group(1, [person("a", "Ada"), person("b", "Bob")], messages=10),
                group(2, [person("a", "Ada"), person("c", "Cara")], messages=10),
            ],
            dm_by_key={"a": {"chat_id": 9, "messages": 5}},
        )

        ada = next(p for p in data["people"] if p["name"] == "Ada")
        self.assertEqual(ada["messages"], 25)

    def test_solo_count_passes_through(self):
        data = build_circles(
            [group(1, [person("a", "Ada"), person("b", "Bob")])],
            solo_count=12,
        )

        self.assertEqual(data["solo"], 12)

    def test_keeps_the_busiest_groups_when_capped(self):
        raw = [
            group(i, [person("a", "Ada"), person(f"x{i}", f"P{i}")], messages=10 + i)
            for i in range(MAX_GROUPS + 5)
        ]
        data = build_circles(raw)

        self.assertEqual(len(data["groups"]), MAX_GROUPS)
        self.assertEqual(max(g["messages"] for g in data["groups"]), 10 + MAX_GROUPS + 4)
        self.assertEqual(min(g["messages"] for g in data["groups"]), 10 + 5)


class CirclesPageTest(unittest.TestCase):
    def test_nav_includes_circles(self):
        self.assertIn(("/circles", "Circles", "circles"), NAV)

    def test_json_script_cannot_break_out_of_the_tag(self):
        html = json_script("circlesData", {"name": "</script><img>"})
        self.assertIn('id="circlesData"', html)
        self.assertNotIn("</script><img>", html)
