"""Adapters for fetching podcast feeds and downloading their episodes."""

from __future__ import annotations

from partio.adapters.feed.download import download_file
from partio.adapters.feed.rss import (
    fetch_episodes,
    fetch_feed_content,
    fetch_feed_title,
    parse_feed,
    parse_feed_title,
)

__all__ = [
    "download_file",
    "fetch_episodes",
    "fetch_feed_content",
    "fetch_feed_title",
    "parse_feed",
    "parse_feed_title",
]
