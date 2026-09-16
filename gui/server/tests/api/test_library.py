"""Tests for the feed/track/download library endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from partio.cli.library import _cache, _tracks
from partio.core.ports import AudioPathKind

from app.main import app
from app.services import tracks_cache

client = TestClient(app)


def _feed_document(*episodes: str) -> bytes:
    items = "".join(
        f"""
        <item>
          <title>{title}</title>
          <guid>{title}-guid</guid>
          <enclosure url="https://feeds.example/{title}.mp3" length="123" type="audio/mpeg"/>
          <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
        </item>
        """
        for title in episodes
    )
    return f"<rss><channel>{items}</channel></rss>".encode()


def _stub_feed(monkeypatch, *episodes: str, dest=None) -> None:
    document = _feed_document(*episodes)
    monkeypatch.setattr(_tracks, "fetch_feed_content", lambda _url, **_kw: document)
    if dest is not None:
        monkeypatch.setattr(_tracks, "DOWNLOAD_DIR", dest)


def test_add_list_remove_feed(monkeypatch) -> None:
    monkeypatch.setattr("app.api.library.fetch_feed_title", lambda _url: "The Daily")

    created = client.post("/api/feeds", json={"url": "https://feeds.example/daily"})
    assert created.status_code == 201
    body = created.json()
    assert body["label"] == "The Daily"

    listed = client.get("/api/feeds").json()
    assert [f["url"] for f in listed] == ["https://feeds.example/daily"]

    removed = client.delete(f"/api/feeds/{body['id']}")
    assert removed.status_code == 204
    assert client.get("/api/feeds").json() == []


def test_add_feed_rejects_a_duplicate_url(monkeypatch) -> None:
    monkeypatch.setattr("app.api.library.fetch_feed_title", lambda _url: "The Daily")
    client.post("/api/feeds", json={"url": "https://feeds.example/daily"})

    conflict = client.post("/api/feeds", json={"url": "https://feeds.example/daily"})
    assert conflict.status_code == 409


def test_remove_unknown_feed_404s() -> None:
    response = client.delete("/api/feeds/does-not-exist")
    assert response.status_code == 404


def test_list_tracks_offers_feed_episodes_before_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.api.library.fetch_feed_title", lambda _url: "Daily")
    client.post("/api/feeds", json={"url": "https://feeds.example/daily"})
    _stub_feed(monkeypatch, "Episode One", dest=tmp_path)

    response = client.get("/api/tracks")
    assert response.status_code == 200
    tracks = response.json()
    assert len(tracks) == 1
    assert tracks[0]["label"] == "Episode One"
    assert tracks[0]["on_disk"] is False
    assert tracks[0]["episode"]["audio_url"] == "https://feeds.example/Episode One.mp3"


def test_list_tracks_marks_an_on_disk_seed(tmp_path) -> None:
    seed = tmp_path / "seed.mp3"
    seed.write_bytes(b"not really audio")
    _cache.remember(seed, label="My Seed", kind=AudioPathKind.SAMPLE)

    response = client.get("/api/tracks", params={"kind": "sample"})
    tracks = response.json()
    assert [t["label"] for t in tracks] == ["My Seed"]
    assert tracks[0]["on_disk"] is True
    assert tracks[0]["episode"] is None


def test_tracks_status_reports_has_more() -> None:
    response = client.get("/api/tracks/status")
    assert response.status_code == 200
    assert response.json() == {"has_more": False}


def test_download_unknown_path_404s() -> None:
    response = client.post("/api/tracks/download", json={"path": "/nowhere/at/all.mp3"})
    assert response.status_code == 404


def test_download_an_already_on_disk_track_skips_the_network(monkeypatch, tmp_path) -> None:
    tracks_cache._by_path.clear()
    local = tmp_path / "already-here.mp3"
    local.write_bytes(b"already here")
    track = _tracks.Track(
        label="Already here", path=local, kind=AudioPathKind.SOURCE, group="on disk"
    )
    tracks_cache.remember_tracks([track])

    def _never(*_args, **_kwargs):
        raise AssertionError("should not download an on-disk track")

    monkeypatch.setattr("app.services.tracks_cache.download_file", _never)

    with client.stream("POST", "/api/tracks/download", json={"path": str(local)}) as response:
        assert response.status_code == 200
        events = list(response.iter_lines())
    assert any('"type": "result"' in line for line in events)


def test_download_a_remote_track_streams_progress(monkeypatch, tmp_path) -> None:
    tracks_cache._by_path.clear()
    destination = tmp_path / "episode.mp3"
    episode = _tracks.FeedEpisode(
        title="Episode One",
        audio_url="https://feeds.example/episode-one.mp3",
        guid="guid-1",
        published=datetime(2024, 1, 1, tzinfo=UTC),
        size_bytes=None,
    )
    track = _tracks.Track(
        label="Episode One",
        path=destination,
        kind=AudioPathKind.SOURCE,
        group="Daily",
        episode=episode,
    )
    tracks_cache.remember_tracks([track])

    def _fake_download(*, url, destination_path, on_progress=None):  # noqa: ARG001
        if on_progress is not None:
            on_progress(5, 10)
            on_progress(10, 10)
        destination_path.write_bytes(b"downloaded")

    monkeypatch.setattr("app.services.tracks_cache.download_file", _fake_download)

    with client.stream("POST", "/api/tracks/download", json={"path": str(destination)}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert '"type": "progress"' in body
    assert '"bytes": 10' in body
    assert '"type": "result"' in body
    assert destination.read_bytes() == b"downloaded"


@pytest.fixture(autouse=True)
def _clear_tracks_cache():
    yield
    tracks_cache._by_path.clear()
