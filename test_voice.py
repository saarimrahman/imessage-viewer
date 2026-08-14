import unittest
from collections import Counter

from voice import _tokenize, frequent_phrases, phrase_counts


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

    def test_skips_grammar_edges(self):
        counts = phrase_counts(
            ["can", "get", "that", "would", "be", "going", "to", "be", "or", "some", "shit"],
            min_n=2,
            max_n=3,
        )

        self.assertNotIn("can get", counts)
        self.assertNotIn("that would be", counts)
        self.assertNotIn("going to be", counts)
        self.assertNotIn("or some shit", counts)
        self.assertEqual(counts["some shit"], 1)

    def test_keeps_phrases_that_start_with_for(self):
        counts = phrase_counts(["for", "sure", "figure", "it", "out"], min_n=2, max_n=3)

        self.assertEqual(counts["for sure"], 1)
        self.assertEqual(counts["figure it out"], 1)


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

    def test_drops_longer_phrase_that_only_adds_filler(self):
        ranked = frequent_phrases(
            Counter({"thank you": 40, "thank you for": 8, "some shit": 20, "or some shit": 10}),
            stop={"you", "for", "or", "some"},
            min_count=3,
        )
        phrases = [item["phrase"] for item in ranked]

        self.assertEqual(phrases, ["thank you", "some shit"])

    def test_drops_light_verb_glue(self):
        ranked = frequent_phrases(
            Counter({"need to get": 33, "can get": 20, "pull up": 20}),
            stop={"to", "can", "up"},
            min_count=3,
        )

        self.assertEqual([item["phrase"] for item in ranked], ["pull up"])


class TokenizeTest(unittest.TestCase):
    def test_strips_emails(self):
        self.assertEqual(_tokenize("email me at saarimmm@gmail.com later"), ["email", "me", "at", "later"])
