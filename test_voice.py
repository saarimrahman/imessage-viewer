import unittest
from collections import Counter

from voice import frequent_phrases, phrase_counts


class PhraseCountsTest(unittest.TestCase):
    def test_counts_overlapping_trigrams(self):
        counts = phrase_counts(
            ["see", "you", "later", "see", "you", "later"],
            min_n=3,
            max_n=3,
        )

        self.assertEqual(counts["see you later"], 2)
        self.assertEqual(counts["you later see"], 1)
        self.assertEqual(counts["later see you"], 1)

    def test_counts_several_lengths(self):
        counts = phrase_counts(["love", "you", "so", "much"], min_n=2, max_n=4)

        self.assertEqual(counts["love you"], 1)
        self.assertEqual(counts["love you so"], 1)
        self.assertEqual(counts["love you so much"], 1)

    def test_skips_phrase_that_starts_or_ends_on_filler(self):
        counts = phrase_counts(["see", "you", "in", "the", "morning"], min_n=3, max_n=3)

        self.assertEqual(counts, {})

    def test_keeps_longer_phrase_with_filler_inside(self):
        counts = phrase_counts(["see", "you", "in", "the", "morning"], min_n=2, max_n=5)

        self.assertEqual(counts["see you in the morning"], 1)
        self.assertEqual(counts["see you"], 1)
        self.assertNotIn("you in the", counts)
        self.assertNotIn("in the morning", counts)


class FrequentPhrasesTest(unittest.TestCase):
    def test_drops_short_phrase_covered_by_a_longer_one(self):
        ranked = frequent_phrases(
            Counter({"see you later": 12, "see you": 12, "you later": 12}),
            min_count=3,
        )
        phrases = [item["phrase"] for item in ranked]

        self.assertEqual(phrases, ["see you later"])

    def test_keeps_short_phrase_with_its_own_count(self):
        ranked = frequent_phrases(
            Counter({"thank you so much": 8, "thank you": 40}),
            min_count=3,
        )
        phrases = [item["phrase"] for item in ranked]

        self.assertEqual(phrases, ["thank you", "thank you so much"])

    def test_ignores_phrases_below_min_count(self):
        ranked = frequent_phrases(Counter({"rare bird": 2}), min_count=3)

        self.assertEqual(ranked, [])

    def test_drops_phrases_made_only_of_function_words(self):
        ranked = frequent_phrases(
            Counter({"will be": 50, "are you": 40, "sounds good": 20}),
            stop={"will", "be", "are", "you"},
            min_count=3,
        )

        self.assertEqual([item["phrase"] for item in ranked], ["sounds good"])

    def test_keeps_phrase_with_one_content_word(self):
        ranked = frequent_phrases(
            Counter({"thank you": 40, "for sure": 20}),
            stop={"you", "for"},
            min_count=3,
        )
        phrases = [item["phrase"] for item in ranked]

        self.assertEqual(phrases, ["thank you", "for sure"])

    def test_reserves_slots_for_longer_phrases(self):
        counts = Counter({f"word{i} extra{i}": 100 - i for i in range(15)})
        counts["see you later"] = 12
        ranked = frequent_phrases(counts, limit=10, min_count=3)
        phrases = [item["phrase"] for item in ranked]

        self.assertIn("see you later", phrases)
        self.assertEqual(len(phrases), 10)
