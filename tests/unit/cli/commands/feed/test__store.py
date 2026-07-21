"""Tests for the feed store wiring."""

from __future__ import annotations

import pytest

from partio.cli.commands.feed import _store as store_module
from partio.core.ports import FeedEntry


@pytest.fixture(autouse=True)
def _feeds_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")


def test_entries_round_trip_through_json():
    """A stored feed reads back identical across store instances."""
    entry = FeedEntry(id="f1", url="https://a/rss.xml", label="Show A")
    store_module.default_store().add_item(entry)

    assert store_module.default_store().list_items() == [entry]


def test_get_and_remove_by_id():
    """Feeds are addressable by their id."""
    store = store_module.default_store()
    store.add_item(FeedEntry(id="f1", url="https://a", label="A"))

    assert store.get_item("f1") is not None
    store.remove_item("f1")
    assert store.get_item("f1") is None


def test_missing_file_lists_empty():
    """A store whose file does not exist yet behaves as empty."""
    assert store_module.default_store().list_items() == []
