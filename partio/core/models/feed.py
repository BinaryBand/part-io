"""Domain model for a single podcast feed episode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003


@dataclass(frozen=True)
class FeedEpisode:
    """One downloadable episode parsed from an RSS/Atom feed.

    *audio_url* is the enclosure URL to download; *guid* uniquely identifies the
    episode within its feed; *published* is the publication time when the feed
    provides one; *size_bytes* is the enclosure's declared length, when it
    declares one.
    """

    title: str
    audio_url: str
    guid: str
    published: datetime | None
    size_bytes: int | None = None


__all__ = ["FeedEpisode"]
