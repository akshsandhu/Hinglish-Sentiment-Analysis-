"""Optional Xquik tweet search loader for the Streamlit app."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, parse, request

API_URL = "https://xquik.com/api/v1/x/tweets/search"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_LIMIT = 5
MAX_LIMIT = 200


class XquikSourceError(RuntimeError):
    """Raised when Xquik results cannot be loaded."""


def fetch_tweet_texts(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    api_key: str | None = None,
    opener: Any | None = None,
) -> list[str]:
    """Return tweet text snippets for an X search query."""
    normalized_query = query.strip()
    if len(normalized_query) < 2:
        raise XquikSourceError("Enter at least 2 characters for the X search query.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise XquikSourceError(f"Limit must be between 1 and {MAX_LIMIT}.")

    resolved_key = (api_key or os.getenv("XQUIK_API_KEY") or "").strip()
    if not resolved_key:
        raise XquikSourceError("Set XQUIK_API_KEY before loading X results.")

    payload = _load_payload(normalized_query, limit, resolved_key, opener)
    texts: list[str] = []
    for candidate in _iter_tweet_candidates(payload):
        text = _extract_text(candidate)
        if text and text not in texts:
            texts.append(text)
        if len(texts) >= limit:
            break

    if not texts:
        raise XquikSourceError("No tweet text was returned for that query.")

    return texts


def _load_payload(query: str, limit: int, api_key: str, opener: Any | None) -> Any:
    url = f"{API_URL}?{parse.urlencode({'q': query, 'limit': limit})}"
    api_request = request.Request(
        url,
        headers={
            "accept": "application/json",
            "x-api-key": api_key,
        },
        method="GET",
    )
    open_request = opener or request.urlopen

    try:
        with open_request(api_request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise XquikSourceError(f"Xquik request failed with HTTP {exc.code}.") from exc
    except error.URLError as exc:
        raise XquikSourceError("Xquik request could not connect.") from exc
    except TimeoutError as exc:
        raise XquikSourceError("Xquik request timed out.") from exc

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise XquikSourceError("Xquik returned invalid JSON.") from exc


def _iter_tweet_candidates(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ("tweets", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return _iter_tweet_candidates(data)

    return [payload]


def _extract_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate.strip()

    if not isinstance(candidate, dict):
        return ""

    for key in ("text", "fullText", "full_text", "tweetText", "content", "rawContent"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    legacy = candidate.get("legacy")
    if isinstance(legacy, dict):
        return _extract_text(legacy)

    return ""
