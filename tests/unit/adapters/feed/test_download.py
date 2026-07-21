"""Tests for the streaming HTTP file downloader."""

from __future__ import annotations

import httpx
import pytest

from partio.adapters.feed import download as download_module
from partio.adapters.feed import download_file


class _FakeResponse:
    def __init__(self, chunks, *, status_error=False, headers=None):
        self._chunks = chunks
        self._status_error = status_error
        self.headers = headers or {}
        self.num_bytes_downloaded = 0

    def raise_for_status(self):
        if self._status_error:
            raise httpx.HTTPError("bad status")

    def iter_bytes(self):
        for chunk in self._chunks:
            self.num_bytes_downloaded += len(chunk)
            yield chunk


class _FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *_exc):
        return False


def test_download_file_writes_all_chunks(monkeypatch, tmp_path):
    """download_file() streams every chunk into the destination, creating parents."""
    response = _FakeResponse([b"ab", b"cd"])
    monkeypatch.setattr(download_module.httpx, "stream", lambda *_a, **_k: _FakeStream(response))
    dest = tmp_path / "nested" / "episode.mp3"

    download_file(url="https://x/e.mp3", destination_path=dest)

    assert dest.read_bytes() == b"abcd"


def test_download_file_leaves_no_file_on_error(monkeypatch, tmp_path):
    """A non-2xx response raises before any file is created."""
    response = _FakeResponse([], status_error=True)
    monkeypatch.setattr(download_module.httpx, "stream", lambda *_a, **_k: _FakeStream(response))
    dest = tmp_path / "episode.mp3"

    with pytest.raises(httpx.HTTPError):
        download_file(url="https://x/e.mp3", destination_path=dest)

    assert not dest.exists()


def test_download_file_reports_progress(monkeypatch, tmp_path):
    """on_progress receives cumulative bytes and the Content-Length total per chunk."""
    response = _FakeResponse([b"ab", b"cd"], headers={"content-length": "4"})
    monkeypatch.setattr(download_module.httpx, "stream", lambda *_a, **_k: _FakeStream(response))
    dest = tmp_path / "episode.mp3"
    events: list[tuple[int, int | None]] = []

    download_file(
        url="https://x/e.mp3",
        destination_path=dest,
        on_progress=lambda downloaded, total: events.append((downloaded, total)),
    )

    assert events == [(2, 4), (4, 4)]
