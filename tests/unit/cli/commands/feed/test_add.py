"""Tests for the `feed add` CLI command."""

from __future__ import annotations

import httpx
import pytest

from partio.cli.commands.feed import add as feed_add
from partio.cli.library import _feeds as store_module
from partio.cli.output import ExitCode
from partio.core.ports import FeedEntry


@pytest.fixture(autouse=True)
def _feeds_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")


def test_add_remembers_a_feed_using_its_own_title(monkeypatch, capsys):
    """With no --label, the feed's declared title becomes the label."""
    monkeypatch.setattr(feed_add, "fetch_feed_title", lambda _url: "The Daily")

    feed_add.add(ctx=None, url="https://feeds.example/daily")

    entries = store_module.feed_store().list_items()
    assert [(e.url, e.label) for e in entries] == [("https://feeds.example/daily", "The Daily")]
    assert "The Daily" in capsys.readouterr().out


def test_add_prefers_an_explicit_label(monkeypatch):
    """An explicit --label wins and skips the title fetch entirely."""

    def _never(_url):
        raise AssertionError("should not fetch when a label is supplied")

    monkeypatch.setattr(feed_add, "fetch_feed_title", _never)

    feed_add.add(ctx=None, url="https://feeds.example/daily", label="My Show")

    assert store_module.feed_store().list_items()[0].label == "My Show"


def test_add_falls_back_to_the_url_when_the_feed_is_untitled(monkeypatch):
    """A feed declaring no title is labelled by its URL rather than blank."""
    monkeypatch.setattr(feed_add, "fetch_feed_title", lambda _url: "")

    feed_add.add(ctx=None, url="https://feeds.example/daily")

    assert store_module.feed_store().list_items()[0].label == "https://feeds.example/daily"


def test_add_rejects_an_unreachable_feed(monkeypatch):
    """A bad URL fails at add time instead of being stored silently."""

    def _boom(_url):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(feed_add, "fetch_feed_title", _boom)

    with pytest.raises(SystemExit) as exc_info:
        feed_add.add(ctx=None, url="https://typo")

    assert exc_info.value.code == ExitCode.USER_ERROR
    assert store_module.feed_store().list_items() == []


def test_add_rejects_a_duplicate_url(monkeypatch):
    """The same feed cannot be remembered twice."""
    store_module.feed_store().add_item(
        FeedEntry(id="f1", url="https://feeds.example/daily", label="The Daily")
    )
    monkeypatch.setattr(feed_add, "fetch_feed_title", lambda _url: "The Daily")

    with pytest.raises(SystemExit) as exc_info:
        feed_add.add(ctx=None, url="https://feeds.example/daily")

    assert exc_info.value.code == ExitCode.USER_ERROR
    assert len(store_module.feed_store().list_items()) == 1
