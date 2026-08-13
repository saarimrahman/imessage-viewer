import unittest

from voice import phrase_counts


class PhraseCountsTest(unittest.TestCase):
    def test_counts_overlapping_trigrams(self):
        counts = phrase_counts(
            ["see", "you", "later", "see", "you", "later"],
            size=3,
        )

        self.assertEqual(counts["see you later"], 2)
        self.assertEqual(counts["you later see"], 1)
        self.assertEqual(counts["later see you"], 1)

    def test_skips_phrase_when_any_token_is_common_filler(self):
        counts = phrase_counts(["see", "you", "in", "the", "morning"], size=3)

        self.assertEqual(counts, {})


if __name__ == "__main__":
    unittest.main()
