import unittest
from datetime import datetime

from stats import effective_people, monthly_medians, reply_gaps, sessions

HOUR = 3600


class SessionsTest(unittest.TestCase):
    def test_one_burst_is_one_session(self):
        events = [(0, 0), (60, 0), (300, 1)]

        self.assertEqual(sessions(events), [(0, 0)])

    def test_silence_longer_than_the_gap_opens_a_session(self):
        events = [(0, 0), (300, 1), (300 + 5 * HOUR, 1)]

        self.assertEqual(sessions(events), [(0, 0), (300 + 5 * HOUR, 1)])

    def test_opener_is_whoever_sent_the_first_message(self):
        events = [(0, 1), (60, 0)]

        self.assertEqual(sessions(events), [(0, 1)])


class ReplyGapsTest(unittest.TestCase):
    def test_clock_starts_on_the_first_unanswered_message(self):
        events = [(0, 0), (60, 0), (300, 1)]

        self.assertEqual(reply_gaps(events), [(300, 300, 1)])

    def test_each_change_of_turn_counts_once(self):
        events = [(0, 0), (100, 1), (400, 0), (500, 1)]

        self.assertEqual([gap for _, gap, _ in reply_gaps(events)], [100, 300, 100])

    def test_drops_a_turn_that_changes_after_the_cap(self):
        events = [(0, 0), (25 * HOUR, 1)]

        self.assertEqual(reply_gaps(events), [])

    def test_records_which_side_replied(self):
        events = [(0, 0), (100, 1), (400, 0)]

        self.assertEqual([me for _, _, me in reply_gaps(events)], [1, 0])


class MonthlyMediansTest(unittest.TestCase):
    # Mid-month and midday, so the local time zone cannot move the timestamp
    # into the month before or after.
    MARCH = datetime(2020, 3, 15, 12).timestamp()
    INDEX = {"2020-03": 0}

    def test_skips_a_month_below_the_bar(self):
        gaps = [(self.MARCH, 10, 1), (self.MARCH + 60, 20, 1)]

        self.assertEqual(monthly_medians(gaps, self.INDEX), [])

    def test_takes_the_median_of_a_month_that_clears_the_bar(self):
        gaps = [(self.MARCH, secs, 1) for secs in (10, 20, 30, 40, 500)]

        self.assertEqual(monthly_medians(gaps, self.INDEX), [(0, 30)])

    def test_skips_a_month_outside_the_index(self):
        gaps = [(self.MARCH, 10, 1) for _ in range(9)]

        self.assertEqual(monthly_medians(gaps, {"2021-07": 0}), [])


class EffectivePeopleTest(unittest.TestCase):
    def test_one_person_holding_everything_reads_as_one(self):
        people = [{"values": [10]}, {"values": [0]}]

        _, value = effective_people(people, ["2020-01"])[0]
        self.assertAlmostEqual(value, 1.0)

    def test_an_even_split_reads_as_the_head_count(self):
        people = [{"values": [5]} for _ in range(4)]

        _, value = effective_people(people, ["2020-01"])[0]
        self.assertAlmostEqual(value, 4.0)

    def test_skips_a_month_with_no_traffic(self):
        people = [{"values": [0, 5]}]

        self.assertEqual([ym for ym, _ in effective_people(people, ["2020-01", "2020-02"])], ["2020-02"])


if __name__ == "__main__":
    unittest.main()
