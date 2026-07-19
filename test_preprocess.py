"""Tests for dataset label normalization."""

import unittest

import pandas as pd

from preprocess import normalize_sentiment_label


class NormalizeSentimentLabelTests(unittest.TestCase):
    def test_preserves_integer_valued_float_labels(self):
        self.assertEqual(
            [normalize_sentiment_label(value) for value in (0.0, 1.0, 2.0)],
            [0, 1, 2],
        )

    def test_preserves_float_labels_from_nullable_numeric_series(self):
        labels = pd.Series([0, 1, 2, None], dtype="float64")
        normalized = labels.apply(normalize_sentiment_label)

        self.assertEqual(normalized.dropna().astype(int).tolist(), [0, 1, 2])

    def test_preserves_supported_text_labels(self):
        self.assertEqual(
            [
                normalize_sentiment_label(" Positive "),
                normalize_sentiment_label("NEG"),
                normalize_sentiment_label("neu"),
            ],
            [0, 1, 2],
        )

    def test_rejects_missing_fractional_and_unknown_labels(self):
        for value in (None, float("nan"), 1.5, "mixed"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_sentiment_label(value))


if __name__ == "__main__":
    unittest.main()
