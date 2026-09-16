"""Tests for the range-capable audio file endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.settings import PARTIO_ROOT

client = TestClient(app)

_LIBRARY_FILE = PARTIO_ROOT / "static" / "downloads" / "klan-camp.mp3"


def test_serves_a_file_under_an_allowed_root() -> None:
    response = client.get("/api/audio/file", params={"path": str(_LIBRARY_FILE)})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"


def test_supports_range_requests() -> None:
    response = client.get(
        "/api/audio/file",
        params={"path": str(_LIBRARY_FILE)},
        headers={"Range": "bytes=0-999"},
    )
    assert response.status_code == 206
    assert response.headers["content-range"].startswith("bytes 0-999/")
    assert len(response.content) == 1000


def test_rejects_a_path_outside_the_library(tmp_path: Path) -> None:
    outsider = tmp_path / "not-in-the-library.mp3"
    outsider.write_bytes(b"not really audio")
    response = client.get("/api/audio/file", params={"path": str(outsider)})
    assert response.status_code == 403


def test_404s_a_missing_file_inside_an_allowed_root() -> None:
    missing = PARTIO_ROOT / "static" / "downloads" / "does-not-exist.mp3"
    response = client.get("/api/audio/file", params={"path": str(missing)})
    assert response.status_code == 404
