"""Tests for the `feed download` CLI command."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from partio.cli.commands.feed import _store as feed_store_module
from partio.cli.commands.feed import download as feed_download
from partio.cli.commands.library import _store as library_store_module
from partio.cli.output import ExitCode
from partio.core.models import FeedEpisode
from partio.core.ports import AudioPathKind, FeedEntry


@pytest.fixture(autouse=True)
def _stores(tmp_path, monkeypatch):
    """Point both stores at scratch files for every test in this module."""
    monkeypatch.setattr(library_store_module, "DEFAULT_LIBRARY_PATH", tmp_path / "library.json")
    monkeypatch.setattr(feed_store_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")


def _episode(title: str, *, size=None) -> FeedEpisode:
    return FeedEpisode(
        title=title,
        audio_url=f"https://x/{title}.mp3",
        guid=title,
        published=None,
        size_bytes=size,
    )


def _fake_download(monkeypatch):
    """Replace download_file with a stub that writes a placeholder file."""
    calls: list[str] = []

    def _download(*, url, destination_path, on_progress=None):
        calls.append(url)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"audio")
        if on_progress is not None:
            on_progress(5, 5)

    monkeypatch.setattr(feed_download, "download_file", _download)
    return calls


def _sequence(values):
    """A picker stub answering with *values* in order across calls."""
    remaining = list(values)
    return lambda *_a, **_k: remaining.pop(0)


def _fake_episodes(monkeypatch, episodes):
    monkeypatch.setattr(feed_download, "fetch_episodes", lambda _url: episodes)


# -- --count path (picker bypassed) ------------------------------------------


def test_download_with_count_registers_sources(monkeypatch, capsys, tmp_path):
    """--count takes the N latest episodes and remembers each as a SOURCE."""
    _fake_episodes(monkeypatch, [_episode("Ep 2"), _episode("Ep 1")])
    downloads = _fake_download(monkeypatch)

    feed_download.download(ctx=None, url="https://feed", count=2, dest=tmp_path / "dl")

    assert len(downloads) == 2
    entries = library_store_module.default_store().list_items()
    assert [e.label for e in entries] == ["Ep 2", "Ep 1"]
    assert all(e.kind is AudioPathKind.SOURCE for e in entries)
    assert "Downloaded Ep 2" in capsys.readouterr().out


def test_download_respects_count(monkeypatch, tmp_path):
    """Only the requested number of latest episodes are downloaded."""
    _fake_episodes(monkeypatch, [_episode("Ep 3"), _episode("Ep 2"), _episode("Ep 1")])
    downloads = _fake_download(monkeypatch)

    feed_download.download(ctx=None, url="https://feed", count=1, dest=tmp_path / "dl")

    assert len(downloads) == 1
    assert [e.label for e in library_store_module.default_store().list_items()] == ["Ep 3"]


def test_download_skips_already_downloaded(monkeypatch, tmp_path):
    """A second run over the same feed downloads nothing new."""
    _fake_episodes(monkeypatch, [_episode("Ep 1")])
    _fake_download(monkeypatch)
    dest = tmp_path / "dl"

    feed_download.download(ctx=None, url="https://feed", count=1, dest=dest)
    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, url="https://feed", count=1, dest=dest)

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert len(library_store_module.default_store().list_items()) == 1


def test_download_no_episodes_exits_no_result(monkeypatch, tmp_path):
    """An empty feed exits NO_RESULT without touching the store."""
    _fake_episodes(monkeypatch, [])

    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, url="https://feed", count=1, dest=tmp_path)

    assert exc_info.value.code == ExitCode.NO_RESULT


def test_download_fails_on_http_error(monkeypatch, tmp_path):
    """A feed fetch HTTP error exits USER_ERROR via fail()."""

    def _boom(_url):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(feed_download, "fetch_episodes", _boom)

    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, url="https://feed", count=1, dest=tmp_path)

    assert exc_info.value.code == ExitCode.USER_ERROR


# -- episode picker path -----------------------------------------------------


def test_download_without_count_uses_the_episode_picker(monkeypatch, tmp_path):
    """With no --count, only the checked episodes are downloaded."""
    episodes = [_episode("Ep 3"), _episode("Ep 2"), _episode("Ep 1")]
    _fake_episodes(monkeypatch, episodes)
    _fake_download(monkeypatch)
    monkeypatch.setattr(feed_download, "select_many", lambda *_a, **_k: [episodes[0], episodes[2]])

    feed_download.download(ctx=None, url="https://feed", dest=tmp_path / "dl")

    labels = [e.label for e in library_store_module.default_store().list_items()]
    assert labels == ["Ep 3", "Ep 1"]


def test_picker_disables_already_downloaded_episodes(monkeypatch, tmp_path):
    """Episodes already in the library are offered but not selectable."""
    episodes = [_episode("Ep 2"), _episode("Ep 1")]
    _fake_episodes(monkeypatch, episodes)
    _fake_download(monkeypatch)
    dest = tmp_path / "dl"
    feed_download.download(ctx=None, url="https://feed", count=1, dest=dest)  # grabs Ep 2

    captured = {}

    def _capture(_message, options, **_kwargs):
        captured["options"] = options
        return []

    monkeypatch.setattr(feed_download, "select_many", _capture)
    with pytest.raises(SystemExit):
        feed_download.download(ctx=None, url="https://feed", dest=dest)

    by_title = {option.title: option for option in captured["options"]}
    assert by_title["Ep 2"].disabled == "already in library"
    assert by_title["Ep 1"].disabled is None


def test_picker_cancelled_exits_cleanly(monkeypatch, tmp_path):
    """Cancelling the episode picker exits OK without downloading."""
    _fake_episodes(monkeypatch, [_episode("Ep 1")])
    downloads = _fake_download(monkeypatch)
    monkeypatch.setattr(feed_download, "select_many", lambda *_a, **_k: None)

    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, url="https://feed", dest=tmp_path / "dl")

    assert exc_info.value.code == ExitCode.OK
    assert downloads == []


def test_episode_detail_shows_date_and_size() -> None:
    """The dimmed column carries the publication date and enclosure size."""
    from datetime import UTC, datetime

    episode = FeedEpisode(
        title="Ep",
        audio_url="https://x/e.mp3",
        guid="g",
        published=datetime(2026, 7, 19, tzinfo=UTC),
        size_bytes=40_668_000,
    )

    detail = feed_download._episode_detail(episode)

    assert "2026-07-19" in detail
    assert "38.8 MB" in detail


def test_episode_detail_tolerates_missing_metadata() -> None:
    """A feed that declares neither date nor length still renders."""
    assert feed_download._episode_detail(_episode("Ep")) == ""


# -- feed picker -------------------------------------------------------------


def test_download_picks_a_remembered_feed(monkeypatch, tmp_path):
    """With no --url, the remembered feeds are offered and the choice is fetched."""
    feed_store_module.default_store().add_item(
        FeedEntry(id="f1", url="https://remembered", label="The Daily")
    )
    fetched: list[str] = []

    def _fetch(url):
        fetched.append(url)
        return [_episode("Ep 1")]

    monkeypatch.setattr(feed_download, "fetch_episodes", _fetch)
    _fake_download(monkeypatch)
    monkeypatch.setattr(feed_download, "select_one", lambda *_a, **_k: "https://remembered")

    feed_download.download(ctx=None, count=1, dest=tmp_path / "dl")

    assert fetched == ["https://remembered"]


def test_download_without_feeds_explains_how_to_add_one(capsys, tmp_path):
    """An empty feed store points the user at `feed add` instead of prompting."""
    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, dest=tmp_path / "dl")

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert "feed add" in capsys.readouterr().out


def test_feed_picker_cancelled_exits_cleanly(monkeypatch, tmp_path):
    """Cancelling the feed picker exits OK without fetching anything."""
    feed_store_module.default_store().add_item(
        FeedEntry(id="f1", url="https://remembered", label="The Daily")
    )
    monkeypatch.setattr(feed_download, "select_one", lambda *_a, **_k: None)

    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, dest=tmp_path / "dl")

    assert exc_info.value.code == ExitCode.OK


def test_download_dest_defaults_under_static() -> None:
    """The default destination keeps downloads inside static/downloads."""
    import inspect

    default = inspect.signature(feed_download.download).parameters["dest"].default
    assert default == Path("static") / "downloads"


# -- esc / go back -----------------------------------------------------------


def test_esc_in_the_episode_picker_returns_to_the_feed_picker(monkeypatch, tmp_path):
    """GO_BACK from the episode list re-opens the feed picker instead of exiting."""
    from partio.cli.select import GO_BACK

    store = feed_store_module.default_store()
    store.add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    store.add_item(FeedEntry(id="f2", url="https://b", label="Show B"))

    episodes = [_episode("Ep 1")]
    fetched: list[str] = []

    def _fetch(url):
        fetched.append(url)
        return episodes

    monkeypatch.setattr(feed_download, "fetch_episodes", _fetch)
    _fake_download(monkeypatch)
    # First feed picked -> esc out of the episodes -> pick the second feed -> confirm.
    monkeypatch.setattr(feed_download, "select_one", _sequence(["https://a", "https://b"]))
    monkeypatch.setattr(feed_download, "select_many", _sequence([GO_BACK, episodes]))

    feed_download.download(ctx=None, dest=tmp_path / "dl")

    assert fetched == ["https://a", "https://b"]
    assert [e.label for e in library_store_module.default_store().list_items()] == ["Ep 1"]


def test_esc_in_the_episode_picker_with_explicit_url_exits(monkeypatch, tmp_path):
    """With --url there is no feed picker to return to, so esc backs out cleanly."""
    from partio.cli.select import GO_BACK

    _fake_episodes(monkeypatch, [_episode("Ep 1")])
    downloads = _fake_download(monkeypatch)
    monkeypatch.setattr(feed_download, "select_many", _sequence([GO_BACK]))

    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, url="https://feed", dest=tmp_path / "dl")

    assert exc_info.value.code == ExitCode.OK
    assert downloads == []


def test_esc_in_the_feed_picker_exits_cleanly(monkeypatch, tmp_path):
    """The feed picker is the command's first screen, so esc backs out of it."""
    from partio.cli.select import GO_BACK

    feed_store_module.default_store().add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    monkeypatch.setattr(feed_download, "select_one", _sequence([GO_BACK]))

    with pytest.raises(SystemExit) as exc_info:
        feed_download.download(ctx=None, dest=tmp_path / "dl")

    assert exc_info.value.code == ExitCode.OK
