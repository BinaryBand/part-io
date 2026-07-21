"""Tests for the `feed remove` CLI command."""

from __future__ import annotations

import pytest

from partio.cli.commands.feed import _store as store_module
from partio.cli.commands.feed.remove import remove
from partio.cli.output import ExitCode
from partio.core.ports import FeedEntry


@pytest.fixture(autouse=True)
def _feeds_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")


def test_remove_forgets_the_feed(capsys):
    """Removing an existing id drops it from the store."""
    store = store_module.default_store()
    store.add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    store.add_item(FeedEntry(id="f2", url="https://b", label="Show B"))

    remove(ctx=None, feed_id="f1")

    assert [e.id for e in store_module.default_store().list_items()] == ["f2"]
    assert "Removed f1" in capsys.readouterr().out


def test_remove_unknown_id_fails():
    """An unknown id is a user error, not a silent no-op."""
    with pytest.raises(SystemExit) as exc_info:
        remove(ctx=None, feed_id="nope")

    assert exc_info.value.code == ExitCode.USER_ERROR
