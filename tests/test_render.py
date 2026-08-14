import re
import unittest
from datetime import datetime, timedelta

from config import APPLE_EPOCH
from db import apple_date
from render import TIME_BREAK_NS, format_time, is_time_break, render_message_blocks


def apple_ns(dt):
    return int((dt.timestamp() - APPLE_EPOCH) * 1_000_000_000)


def msg(i, dt, me=1, handle="+15555550100"):
    return {
        "id": i,
        "guid": f"g{i}",
        "text": f"m{i}",
        "attributedBody": None,
        "date": apple_ns(dt),
        "is_from_me": me,
        "handle": handle,
    }


def row_classes(html, msg_id):
    match = re.search(rf'<div class="([^"]+)" id="msg-{msg_id}"', html)
    return match.group(1).split() if match else []


class TimeBreakTest(unittest.TestCase):
    NOON = datetime(2024, 6, 15, 12, 0, 0)

    def test_hour_of_silence_is_a_break(self):
        self.assertTrue(is_time_break(0, TIME_BREAK_NS))
        self.assertFalse(is_time_break(0, TIME_BREAK_NS - 1))
        self.assertFalse(is_time_break(None, TIME_BREAK_NS))

    def test_same_day_gap_inserts_a_centered_time(self):
        later = self.NOON + timedelta(hours=2)
        html = render_message_blocks([msg(1, self.NOON), msg(2, later)], {})

        self.assertEqual(html.count("timeSep"), 1)
        self.assertIn(format_time(apple_ns(later)), html)
        self.assertIn("group-start", row_classes(html, 2))

    def test_newer_page_keeps_the_break_across_the_cursor(self):
        later = self.NOON + timedelta(hours=2)
        html = render_message_blocks(
            [msg(2, later)],
            {},
            prev_day=apple_date(apple_ns(self.NOON))[:10],
            prev_sender=(1, "+15555550100"),
            prev_date=apple_ns(self.NOON),
        )

        self.assertEqual(html.count("timeSep"), 1)
        self.assertNotIn("dateSep", html)
        self.assertIn('class="day" data-day="2024-06-15"', html)
        self.assertIn("group-start", row_classes(html, 2))

    def test_short_gap_stays_one_group(self):
        later = self.NOON + timedelta(minutes=20)
        html = render_message_blocks([msg(1, self.NOON), msg(2, later)], {})

        self.assertNotIn("timeSep", html)
        self.assertNotIn("group-start", row_classes(html, 2))

    def test_new_day_uses_the_date_pill_not_a_time_break(self):
        nxt = self.NOON + timedelta(days=1)
        html = render_message_blocks([msg(1, self.NOON), msg(2, nxt)], {})

        self.assertNotIn("timeSep", html)
        self.assertGreaterEqual(html.count("dateSep"), 2)
        self.assertEqual(
            re.findall(r'<div class="day" data-day="([^"]+)">', html),
            ["2024-06-15", "2024-06-16"],
        )

    def test_same_day_messages_share_one_wrapper(self):
        later = self.NOON + timedelta(hours=2)
        html = render_message_blocks([msg(1, self.NOON), msg(2, later)], {})
        self.assertEqual(html.count('class="day"'), 1)

    def test_older_page_skips_the_pill_for_the_day_already_on_screen(self):
        nxt = self.NOON + timedelta(days=1)
        html = render_message_blocks(
            [msg(1, self.NOON), msg(2, nxt)],
            {},
            next_day=apple_date(apple_ns(nxt))[:10],
        )
        self.assertEqual(html.count("dateSep"), 1)
        self.assertIn('data-day="2024-06-15"', html)
        self.assertIn('data-day="2024-06-16"', html)

    def test_older_page_puts_the_break_before_the_loaded_thread(self):
        later = self.NOON + timedelta(hours=3)
        html = render_message_blocks(
            [msg(1, self.NOON)],
            {},
            next_day=apple_date(apple_ns(later))[:10],
            next_sender=(1, "+15555550100"),
            next_date=apple_ns(later),
        )

        self.assertTrue(html.endswith(_time_sep_snippet(later) + "</div>"))
        self.assertIn("group-start", row_classes(html, 1))


def _time_sep_snippet(dt):
    return f'<div class="timeSep">{format_time(apple_ns(dt))}</div>'
