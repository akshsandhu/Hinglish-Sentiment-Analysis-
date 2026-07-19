"""Tests for the optional Xquik tweet loader."""

from __future__ import annotations

import json
import unittest
from urllib import request

from xquik_source import XquikSourceError, fetch_tweet_texts


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class XquikSourceTests(unittest.TestCase):
    def test_fetches_texts_from_tweets_key(self) -> None:
        def opener(api_request: request.Request, timeout: int) -> FakeResponse:
            self.assertEqual(timeout, 15)
            self.assertEqual(api_request.headers["X-api-key"], "test-key")
            self.assertIn("q=hinglish+review", api_request.full_url)
            self.assertIn("limit=5", api_request.full_url)
            return FakeResponse({"tweets": [{"text": "Bahut accha phone"}, {"text": "Average hai"}]})

        texts = fetch_tweet_texts("hinglish review", api_key="test-key", opener=opener)

        self.assertEqual(texts, ["Bahut accha phone", "Average hai"])

    def test_fetches_nested_legacy_text(self) -> None:
        def opener(api_request: request.Request, timeout: int) -> FakeResponse:
            self.assertEqual(api_request.headers["X-api-key"], "test-key")
            return FakeResponse({"data": {"results": [{"legacy": {"full_text": "Zabardast service"}}]}})

        texts = fetch_tweet_texts("service", api_key="test-key", opener=opener)

        self.assertEqual(texts, ["Zabardast service"])

    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(XquikSourceError, "XQUIK_API_KEY"):
            fetch_tweet_texts("review", api_key="")

    def test_rejects_invalid_limits(self) -> None:
        for limit in (0, 201, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(XquikSourceError, "between 1 and 200"):
                    fetch_tweet_texts("review", limit=limit, api_key="test-key")

    def test_reports_empty_results(self) -> None:
        def opener(api_request: request.Request, timeout: int) -> FakeResponse:
            self.assertEqual(api_request.get_method(), "GET")
            return FakeResponse({"tweets": []})

        with self.assertRaisesRegex(XquikSourceError, "No tweet text"):
            fetch_tweet_texts("review", api_key="test-key", opener=opener)


if __name__ == "__main__":
    unittest.main()
