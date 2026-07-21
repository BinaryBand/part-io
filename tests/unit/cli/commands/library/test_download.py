"""Tests for the `library download` CLI command."""

from __future__ import annotations

import httpx
import pytest

from partio.cli.commands.library import _store
from partio.cli.commands.library import download as library_download
from partio.cli.output import ExitCode
from partio.core.models import FeedEpisode
from partio.core.ports import AudioPathKind


@pytest.fixture(autouse=True)
def _library_path(tmp_path, monkeypatch):
    """Point the default store at a scratch file for every test in this module."""
    monkeypatch.setattr(_store, "DEFAULT_LIBRARY_PATH", tmp_path / "library.json")


def _episode(title: str) -> FeedEpisode:
    return FeedEpisode(title=title, audio_url=f"https://x/{title}.mp3", guid=title, published=None)


def _fake_download(monkeypatch):
    """Replace download_file with a stub that writes a placeholder file."""
    calls: list[str] = []

    def _download(*, url, destination_path, on_progress=None):
        calls.append(url)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"audio")
        if on_progress is not None:
            on_progress(5, 5)

    monkeypatch.setattr(library_download, "download_file", _download)
    return calls


def test_download_registers_sources(monkeypatch, capsys, tmp_path):
    """download() fetches, saves, and remembers each episode as a SOURCE."""
    monkeypatch.setattr(
        library_download, "fetch_episodes", lambda _feed: [_episode("Ep 2"), _episode("Ep 1")]
    )
    downloads = _fake_download(monkeypatch)

    library_download.download(ctx=None, feed="https://feed", count=2, dest=tmp_path / "dl")

    assert len(downloads) == 2
    entries = _store.default_store().list_items()
    assert [e.label for e in entries] == ["Ep 2", "Ep 1"]
    assert all(e.kind is AudioPathKind.SOURCE for e in entries)
    assert "Downloaded Ep 2" in capsys.readouterr().out


def test_download_respects_count(monkeypatch, tmp_path):
    """Only the requested number of latest episodes are downloaded."""
    monkeypatch.setattr(
        library_download,
        "fetch_episodes",
        lambda _feed: [_episode("Ep 3"), _episode("Ep 2"), _episode("Ep 1")],
    )
    downloads = _fake_download(monkeypatch)

    library_download.download(ctx=None, feed="https://feed", count=1, dest=tmp_path / "dl")

    assert len(downloads) == 1
    assert [e.label for e in _store.default_store().list_items()] == ["Ep 3"]


def test_download_skips_already_downloaded(monkeypatch, tmp_path):
    """A second run over the same feed downloads nothing new."""
    monkeypatch.setattr(library_download, "fetch_episodes", lambda _feed: [_episode("Ep 1")])
    _fake_download(monkeypatch)
    dest = tmp_path / "dl"

    library_download.download(ctx=None, feed="https://feed", count=1, dest=dest)
    with pytest.raises(SystemExit) as exc_info:
        library_download.download(ctx=None, feed="https://feed", count=1, dest=dest)

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert len(_store.default_store().list_items()) == 1


def test_download_no_episodes_exits_no_result(monkeypatch, tmp_path):
    """An empty feed exits NO_RESULT without touching the store."""
    monkeypatch.setattr(library_download, "fetch_episodes", lambda _feed: [])

    with pytest.raises(SystemExit) as exc_info:
        library_download.download(ctx=None, feed="https://feed", count=1, dest=tmp_path)

    assert exc_info.value.code == ExitCode.NO_RESULT


def test_download_fails_on_http_error(monkeypatch, tmp_path):
    """A feed fetch HTTP error exits USER_ERROR via fail()."""

    def _boom(_feed):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(library_download, "fetch_episodes", _boom)

    with pytest.raises(SystemExit) as exc_info:
        library_download.download(ctx=None, feed="https://feed", count=1, dest=tmp_path)

    assert exc_info.value.code == ExitCode.USER_ERROR
