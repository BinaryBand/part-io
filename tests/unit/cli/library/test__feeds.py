"""Tests for the cli.library._feeds module."""

from __future__ import annotations

from partio.cli.library import _feeds
from partio.core.ports import FeedEntry


def test_feed_store_round_trips_an_entry() -> None:
    """A remembered feed survives a store round trip unchanged."""
    entry = FeedEntry(id="f1", url="https://example.com/rss", label="The Daily")

    _feeds.feed_store().add_item(entry)

    assert _feeds.feed_store().list_items() == [entry]


def test_feeds_returns_every_remembered_feed() -> None:
    """feeds() reads straight through to the store."""
    _feeds.feed_store().add_item(FeedEntry(id="f1", url="https://a", label="A"))
    _feeds.feed_store().add_item(FeedEntry(id="f2", url="https://b", label="B"))

    assert [entry.label for entry in _feeds.feeds()] == ["A", "B"]


def test_feeds_is_empty_when_nothing_is_remembered() -> None:
    """An absent store reads as no feeds rather than as an error."""
    assert _feeds.feeds() == []


def test_feeds_tolerates_a_corrupt_store(monkeypatch, tmp_path) -> None:
    """A broken feeds file degrades to "no feeds" instead of blocking a prompt."""
    broken = tmp_path / "feeds.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(_feeds, "DEFAULT_FEEDS_PATH", broken)

    assert _feeds.feeds() == []
