"""Domain model for a single podcast feed episode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003


@dataclass(frozen=True)
class FeedEpisode:
    """One downloadable episode parsed from an RSS/Atom feed.

    *audio_url* is the enclosure URL to download; *guid* uniquely identifies the
    episode within its feed; *published* is the publication time when the feed
    provides one.
    """

    title: str
    audio_url: str
    guid: str
    published: datetime | None


@dataclass(frozen=True)
class DownloadPlan:
    """A resolved intent to download one *episode* to *destination_path*."""

    episode: FeedEpisode
    destination_path: Path
    label: str


__all__ = ["DownloadPlan", "FeedEpisode"]
