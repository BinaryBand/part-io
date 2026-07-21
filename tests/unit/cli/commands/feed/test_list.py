"""Tests for the `feed list` CLI command."""

from __future__ import annotations

import pytest

from partio.cli.commands.feed import _store as store_module
from partio.cli.commands.feed.list import list_feeds
from partio.cli.output import ExitCode
from partio.core.ports import FeedEntry


@pytest.fixture(autouse=True)
def _feeds_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")


def test_list_prints_every_feed(capsys):
    """Each remembered feed prints its id, label, and URL."""
    store = store_module.default_store()
    store.add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    store.add_item(FeedEntry(id="f2", url="https://b", label="Show B"))

    list_feeds(ctx=None)

    out = capsys.readouterr().out
    assert "f1" in out
    assert "Show A" in out
    assert "https://b" in out


def test_list_empty_exits_no_result(capsys):
    """An empty store reports no feeds and exits NO_RESULT."""
    with pytest.raises(SystemExit) as exc_info:
        list_feeds(ctx=None)

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert "No feeds found." in capsys.readouterr().out
