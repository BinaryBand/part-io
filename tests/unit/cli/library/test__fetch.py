"""Tests for the cli.library._fetch module."""

from __future__ import annotations

from pathlib import Path

import httpx

from partio.cli.library import _cache, _fetch
from partio.cli.library._tracks import Track
from partio.core.models import FeedEpisode
from partio.core.ports import AudioPathKind


def _track(path: Path, *, remote: bool = True) -> Track:
    episode = (
        FeedEpisode(title="Ep 1", audio_url="https://x/ep1.mp3", guid="g", published=None)
        if remote
        else None
    )
    return Track(label="Ep 1", path=path, kind=AudioPathKind.SOURCE, group="Show", episode=episode)


def _stub_download(monkeypatch) -> list[str]:
    """Replace the downloader with a stub that writes a placeholder file."""
    calls: list[str] = []

    def _download(*, url, destination_path, on_progress=None):
        calls.append(url)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"audio")
        if on_progress is not None:
            on_progress(5, 5)

    monkeypatch.setattr(_fetch, "download_file", _download)
    return calls


def test_ensure_local_downloads_a_track_that_is_not_here(monkeypatch, tmp_path) -> None:
    """Choosing a virtual episode is what triggers its download."""
    downloads = _stub_download(monkeypatch)
    destination = tmp_path / "dl" / "ep1.mp3"

    result = _fetch.ensure_local(_track(destination))

    assert result == destination
    assert destination.read_bytes() == b"audio"
    assert downloads == ["https://x/ep1.mp3"]


def test_a_download_is_indexed_so_it_reads_as_on_disk_next_time(monkeypatch, tmp_path) -> None:
    """The fetched file joins the index, which is what marks it available later."""
    _stub_download(monkeypatch)

    _fetch.ensure_local(_track(tmp_path / "dl" / "ep1.mp3"))

    (entry,) = _cache.cached()
    assert (entry.label, entry.kind) == ("Ep 1", AudioPathKind.SOURCE)


def test_ensure_local_downloads_nothing_when_the_file_is_here(monkeypatch, tmp_path) -> None:
    """Only download on request means: never re-download what is already here."""
    downloads = _stub_download(monkeypatch)
    destination = tmp_path / "ep1.mp3"
    destination.write_bytes(b"audio")

    assert _fetch.ensure_local(_track(destination)) == destination
    assert downloads == []


def test_a_failed_download_returns_none_rather_than_exiting(monkeypatch, tmp_path, capsys) -> None:
    """A flaky network costs a retry at the prompt, not the whole session."""

    def _boom(*, url, destination_path, on_progress=None):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(_fetch, "download_file", _boom)

    assert _fetch.ensure_local(_track(tmp_path / "dl" / "ep1.mp3")) is None
    assert "Download failed" in capsys.readouterr().err


def test_local_audio_that_has_vanished_returns_none(monkeypatch, tmp_path, capsys) -> None:
    """Nothing can be downloaded for a local-only track, so say so and move on."""
    _stub_download(monkeypatch)

    assert _fetch.ensure_local(_track(tmp_path / "gone.mp3", remote=False)) is None
    assert "No longer on disk" in capsys.readouterr().err
