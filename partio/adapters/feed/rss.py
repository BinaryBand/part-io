"""Fetch and parse podcast RSS/Atom feeds into :class:`FeedEpisode` records."""

from __future__ import annotations

from datetime import UTC, datetime

import feedparser
import httpx

from partio.core.models import FeedEpisode

_FEED_TIMEOUT_SECONDS = 30.0


def _first_audio_enclosure(entry: feedparser.FeedParserDict) -> tuple[str, int | None] | None:
    """Return the ``(href, size_bytes)`` of the entry's first audio enclosure."""
    for enclosure in entry.get("enclosures", []):
        if str(enclosure.get("type", "")).startswith("audio"):
            href = enclosure.get("href")
            if href:
                return str(href), _enclosure_length(enclosure)
    return None


def _enclosure_length(enclosure: feedparser.FeedParserDict) -> int | None:
    """Return the enclosure's declared byte length, if it parses as one."""
    raw = str(enclosure.get("length", "")).strip()
    return int(raw) if raw.isdigit() else None


def _published_at(entry: feedparser.FeedParserDict) -> datetime | None:
    """Return the entry's publication time as a UTC datetime, if provided."""
    parsed = entry.get("published_parsed")
    if parsed is None:
        return None
    return datetime(*parsed[:6], tzinfo=UTC)


def parse_feed(content: bytes) -> list[FeedEpisode]:
    """Parse feed *content* into episodes that carry a downloadable audio enclosure.

    Entries without an ``audio/*`` enclosure are skipped. Feed order is
    preserved, which for podcast feeds means newest episodes first.
    """
    parsed = feedparser.parse(content)
    episodes: list[FeedEpisode] = []
    for entry in parsed.entries:
        enclosure = _first_audio_enclosure(entry)
        if enclosure is None:
            continue
        audio_url, size_bytes = enclosure
        episodes.append(
            FeedEpisode(
                title=str(entry.get("title", "")),
                audio_url=audio_url,
                guid=str(entry.get("id") or audio_url),
                published=_published_at(entry),
                size_bytes=size_bytes,
            )
        )
    return episodes


def fetch_feed_content(feed_url: str, *, max_bytes: int | None = None) -> bytes:
    """Download the raw feed document at *feed_url*.

    Exposed separately from :func:`parse_feed` so a caller that only wants the
    recent episodes can bound both costs, which are wildly different: a
    long-running podcast feed is tens of megabytes to move and seconds to
    parse.

    *max_bytes* asks the server for just that much via a ranged request.
    Servers that do not honour ranges reply with the whole document instead, so
    this is a best-effort saving, never a correctness assumption -- the caller
    still has to treat what comes back as possibly complete.
    """
    headers = {} if max_bytes is None else {"Range": f"bytes=0-{max_bytes - 1}"}
    response = httpx.get(
        feed_url, follow_redirects=True, timeout=_FEED_TIMEOUT_SECONDS, headers=headers
    )
    response.raise_for_status()
    return response.content


def fetch_episodes(feed_url: str) -> list[FeedEpisode]:
    """Download the feed at *feed_url* and parse all of it into episodes."""
    return parse_feed(fetch_feed_content(feed_url))


def parse_feed_title(content: bytes) -> str:
    """Return the feed's own title, or an empty string when it declares none."""
    parsed = feedparser.parse(content)
    return str(parsed.feed.get("title", ""))


def fetch_feed_title(feed_url: str) -> str:
    """Download the feed at *feed_url* and return its title.

    Doubles as validation when remembering a feed: an unreachable or non-2xx
    URL raises :class:`httpx.HTTPError` rather than being stored silently.
    """
    response = httpx.get(feed_url, follow_redirects=True, timeout=_FEED_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_feed_title(response.content)


__all__ = ["fetch_episodes", "fetch_feed_title", "parse_feed", "parse_feed_title"]
