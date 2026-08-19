import sqlite3
import unittest

import db


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            guid TEXT,
            text TEXT,
            attributedBody BLOB,
            date INTEGER,
            is_from_me INTEGER,
            handle_id INTEGER,
            associated_message_type INTEGER,
            associated_message_guid TEXT,
            associated_message_emoji TEXT
        );
        CREATE TABLE chat_message_join (
            chat_id INTEGER,
            message_id INTEGER,
            message_date INTEGER,
            PRIMARY KEY (chat_id, message_id)
        );
        CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, mime_type TEXT, filename TEXT);
        CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
        CREATE INDEX chat_message_join_idx_message_date_id_chat_id
            ON chat_message_join (chat_id, message_date, message_id);
        """
    )
    conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+15555550100')")
    conn.execute(
        "INSERT INTO chat (ROWID, chat_identifier, display_name) VALUES (1, '+15555550100', 'Ada')"
    )
    conn.execute("INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (1, 1)")
    return conn


def add_message(conn, rowid, date, text="hi", me=1, guid=None, assoc=None):
    guid = guid or f"g{rowid}"
    assoc_type, assoc_guid, assoc_emoji = (None, None, None)
    if assoc:
        assoc_type, assoc_guid, assoc_emoji = assoc
    conn.execute(
        """INSERT INTO message (
               ROWID, guid, text, attributedBody, date, is_from_me, handle_id,
               associated_message_type, associated_message_guid, associated_message_emoji
           ) VALUES (?, ?, ?, NULL, ?, ?, 1, ?, ?, ?)""",
        (rowid, guid, text, date, me, assoc_type, assoc_guid, assoc_emoji),
    )
    conn.execute(
        "INSERT INTO chat_message_join (chat_id, message_id, message_date) VALUES (1, ?, ?)",
        (rowid, date),
    )


class FetchMessagesTest(unittest.TestCase):
    def setUp(self):
        db.CHAT_GROUPS = None
        self.conn = make_conn()
        for i in range(1, 21):
            add_message(self.conn, i, i * 1000, text=f"m{i}")

    def tearDown(self):
        self.conn.close()
        db.CHAT_GROUPS = None

    def test_newest_page_is_the_latest_rows(self):
        rows = db.fetch_messages(self.conn, 1, from_end=True, limit=5)
        self.assertEqual([r["id"] for r in rows], [16, 17, 18, 19, 20])

    def test_oldest_page_is_the_earliest_rows(self):
        rows = db.fetch_messages(self.conn, 1, limit=5)
        self.assertEqual([r["id"] for r in rows], [1, 2, 3, 4, 5])

    def test_after_cursor_walks_forward(self):
        rows = db.fetch_messages(self.conn, 1, after=(5000, 5), limit=3)
        self.assertEqual([r["id"] for r in rows], [6, 7, 8])

    def test_before_cursor_walks_backward(self):
        rows = db.fetch_messages(self.conn, 1, before=(16000, 16), limit=3)
        self.assertEqual([r["id"] for r in rows], [13, 14, 15])

    def test_around_keeps_the_target_in_the_window(self):
        rows = db.fetch_messages_around(self.conn, 1, 10, half=2)
        self.assertEqual([r["id"] for r in rows], [9, 10, 11, 12])

    def test_has_neighbor_at_the_ends(self):
        self.assertFalse(db.has_neighbor(self.conn, 1, 1000, 1, "before"))
        self.assertTrue(db.has_neighbor(self.conn, 1, 1000, 1, "after"))
        self.assertTrue(db.has_neighbor(self.conn, 1, 20000, 20, "before"))
        self.assertFalse(db.has_neighbor(self.conn, 1, 20000, 20, "after"))


class LoadReactionsTest(unittest.TestCase):
    def setUp(self):
        db.CHAT_GROUPS = None
        self.conn = make_conn()
        add_message(self.conn, 1, 1000, guid="target-a")
        add_message(self.conn, 2, 2000, guid="target-b")
        add_message(
            self.conn, 3, 3000, text=None, me=0,
            assoc=(2000, "p:0/target-a", None),
        )
        add_message(
            self.conn, 4, 4000, text=None, me=0,
            assoc=(2001, "target-b", None),
        )
        # A tapback on a message that is not on this page must not be required
        # for the page query to return the ones that are.
        add_message(self.conn, 5, 5000, guid="target-c")
        add_message(
            self.conn, 6, 6000, text=None, me=0,
            assoc=(2002, "p:3/target-c", None),
        )

    def tearDown(self):
        self.conn.close()
        db.CHAT_GROUPS = None

    def test_prefixed_and_bare_guids_resolve(self):
        out = db.load_reactions(self.conn, 1, ["target-a", "target-b"])
        self.assertEqual(set(out), {"target-a", "target-b"})
        self.assertEqual(out["target-a"][0][0], "❤️")
        self.assertEqual(out["target-b"][0][0], "👍")
        self.assertNotIn("target-c", out)

    def test_later_removal_clears_the_badge(self):
        add_message(
            self.conn, 7, 7000, text=None, me=0,
            assoc=(3000, "p:0/target-a", None),
        )
        out = db.load_reactions(self.conn, 1, ["target-a"])
        self.assertEqual(out, {})


def add_photo(conn, att_id, date, mime="image/jpeg"):
    add_message(conn, att_id, date, text=None)
    conn.execute(
        "INSERT INTO attachment (ROWID, mime_type, filename) VALUES (?, ?, ?)",
        (att_id, mime, f"/tmp/{att_id}.jpg"),
    )
    conn.execute(
        "INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (?, ?)",
        (att_id, att_id),
    )


class FetchMediaTest(unittest.TestCase):
    def setUp(self):
        db.CHAT_GROUPS = None
        self.conn = make_conn()
        for i in range(1, 11):
            add_photo(self.conn, i, i * 1000)

    def tearDown(self):
        self.conn.close()
        db.CHAT_GROUPS = None

    def test_newest_page_is_the_latest_photos(self):
        rows = db.fetch_media(self.conn, 1, from_end=True, limit=3, visual_only=True)
        self.assertEqual([r["att_id"] for r in rows], [8, 9, 10])

    def test_after_cursor_walks_forward(self):
        rows = db.fetch_media(self.conn, 1, after=(3000, 3), limit=2, visual_only=True)
        self.assertEqual([r["att_id"] for r in rows], [4, 5])

    def test_before_cursor_walks_backward(self):
        rows = db.fetch_media(self.conn, 1, before=(8000, 8), limit=2, visual_only=True)
        self.assertEqual([r["att_id"] for r in rows], [6, 7])

    def test_visual_only_drops_files(self):
        add_photo(self.conn, 20, 20000, mime="application/pdf")
        rows = db.fetch_media(self.conn, 1, from_end=True, limit=3, visual_only=True)
        self.assertEqual([r["att_id"] for r in rows], [8, 9, 10])
        rows = db.fetch_media(self.conn, 1, from_end=True, limit=3, visual_only=False)
        self.assertEqual([r["att_id"] for r in rows], [9, 10, 20])

    def test_bounds_and_count(self):
        bounds = db.media_date_bounds(self.conn, 1, visual_only=True)
        self.assertEqual(bounds["lo"], 1000)
        self.assertEqual(bounds["hi"], 10000)
        self.assertEqual(db.media_count(self.conn, 1, visual_only=True), 10)
