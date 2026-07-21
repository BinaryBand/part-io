"""Tests for the `feed remove` CLI command."""

from __future__ import annotations

import pytest

from partio.cli.commands.feed import remove as remove_module
from partio.cli.commands.feed.remove import remove
from partio.cli.library import _feeds as store_module
from partio.cli.output import ExitCode
from partio.cli.select import GO_BACK
from partio.core.ports import FeedEntry


@pytest.fixture(autouse=True)
def _feeds_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")


def test_remove_forgets_the_feed(capsys):
    """Removing an existing id drops it from the store."""
    store = store_module.feed_store()
    store.add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    store.add_item(FeedEntry(id="f2", url="https://b", label="Show B"))

    remove(ctx=None, feed_id="f1")

    assert [e.id for e in store_module.feed_store().list_items()] == ["f2"]
    assert "Removed f1" in capsys.readouterr().out


def test_remove_unknown_id_fails():
    """An unknown id is a user error, not a silent no-op."""
    with pytest.raises(SystemExit) as exc_info:
        remove(ctx=None, feed_id="nope")

    assert exc_info.value.code == ExitCode.USER_ERROR


def test_remove_without_an_id_picks_a_feed_by_name(monkeypatch):
    """Nobody should have to look up an id, so the default is a picker."""
    store_module.feed_store().add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    captured = {}

    def _pick(_message, options, **_kwargs):
        captured["titles"] = [option.title for option in options]
        return "f1"

    monkeypatch.setattr(remove_module, "select_one", _pick)

    remove(ctx=None)

    assert captured["titles"] == ["Show A"]
    assert store_module.feed_store().list_items() == []


def test_remove_without_feeds_explains_how_to_add_one(capsys):
    """An empty store points at `feed add` instead of opening an empty picker."""
    with pytest.raises(SystemExit) as exc_info:
        remove(ctx=None)

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert "feed add" in capsys.readouterr().out


def test_esc_in_the_picker_exits_cleanly(monkeypatch):
    """The picker is the command's first screen, so esc backs out of it."""
    store_module.feed_store().add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    monkeypatch.setattr(remove_module, "select_one", lambda *_a, **_k: GO_BACK)

    with pytest.raises(SystemExit) as exc_info:
        remove(ctx=None)

    assert exc_info.value.code == ExitCode.OK
    assert len(store_module.feed_store().list_items()) == 1
