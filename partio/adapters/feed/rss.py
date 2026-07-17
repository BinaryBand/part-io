"""Fetch and parse podcast RSS/Atom feeds into :class:`FeedEpisode` records."""

from __future__ import annotations

from datetime import UTC, datetime

import feedparser
import httpx

from partio.core.models import FeedEpisode

_FEED_TIMEOUT_SECONDS = 30.0


def _first_audio_url(entry: feedparser.FeedParserDict) -> str | None:
    """Return the href of the entry's first audio enclosure, if any."""
    for enclosure in entry.get("enclosures", []):
        if str(enclosure.get("type", "")).startswith("audio"):
            href = enclosure.get("href")
            if href:
                return str(href)
    return None


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
        audio_url = _first_audio_url(entry)
        if audio_url is None:
            continue
        episodes.append(
            FeedEpisode(
                title=str(entry.get("title", "")),
                audio_url=audio_url,
                guid=str(entry.get("id") or audio_url),
                published=_published_at(entry),
            )
        )
    return episodes


def fetch_episodes(feed_url: str) -> list[FeedEpisode]:
    """Download the feed at *feed_url* and parse it into episodes."""
    response = httpx.get(feed_url, follow_redirects=True, timeout=_FEED_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_feed(response.content)


__all__ = ["fetch_episodes", "parse_feed"]
