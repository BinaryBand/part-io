"""Tests for the cli.library._tracks module."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx

from partio.cli.library import _cache, _tracks
from partio.cli.library._feeds import feed_store
from partio.core.models import FeedEpisode
from partio.core.ports import AudioPathKind, FeedEntry


def _episode(title: str, *, published=None, size=None) -> FeedEpisode:
    return FeedEpisode(
        title=title,
        audio_url=f"https://x/{title}.mp3",
        guid=title,
        published=published,
        size_bytes=size,
    )


def _remember_feed(url: str = "https://feed", label: str = "Show") -> None:
    feed_store().add_item(FeedEntry(id=url, url=url, label=label))


def _stub_feed(monkeypatch, episodes, *, dest=None) -> None:
    """Serve *episodes* for any feed URL and land downloads under *dest*."""
    monkeypatch.setattr(_tracks, "fetch_episodes", lambda _url: list(episodes))
    if dest is not None:
        monkeypatch.setattr(_tracks, "DOWNLOAD_DIR", dest)
    _tracks.refresh()


# -- enumeration -------------------------------------------------------------


def test_tracks_offers_every_episode_before_anything_is_downloaded(monkeypatch, tmp_path) -> None:
    """The whole back catalogue is selectable with an empty disk -- the point of it."""
    _remember_feed()
    _stub_feed(monkeypatch, [_episode("Ep 2"), _episode("Ep 1")], dest=tmp_path)

    listed = _tracks.tracks()

    assert [track.label for track in listed] == ["Ep 2", "Ep 1"]
    assert not any(track.on_disk for track in listed)


def test_tracks_groups_episodes_under_their_feed(monkeypatch, tmp_path) -> None:
    """Each row is filed under the feed it came from, for the picker's headings."""
    _remember_feed(label="Behind the Bastards")
    _stub_feed(monkeypatch, [_episode("Klan Camp")], dest=tmp_path)

    (track,) = _tracks.tracks()

    assert track.group == "Behind the Bastards"
    assert track.episode is not None


def test_tracks_marks_an_episode_already_on_disk(monkeypatch, tmp_path) -> None:
    """A downloaded episode keeps its feed row but reads as available."""
    _remember_feed()
    _stub_feed(monkeypatch, [_episode("Ep 1")], dest=tmp_path)
    (planned,) = _tracks.tracks()
    planned.path.parent.mkdir(parents=True, exist_ok=True)
    planned.path.write_bytes(b"audio")

    (track,) = _tracks.tracks()

    assert track.on_disk
    assert track.mark == _tracks.ON_DISK_MARK


def test_a_downloaded_episode_is_not_listed_twice(monkeypatch, tmp_path) -> None:
    """Indexing a download must not duplicate the feed row it came from."""
    _remember_feed()
    _stub_feed(monkeypatch, [_episode("Ep 1")], dest=tmp_path)
    (planned,) = _tracks.tracks()
    planned.path.parent.mkdir(parents=True, exist_ok=True)
    planned.path.write_bytes(b"audio")
    _cache.remember(planned.path, label="Ep 1", kind=AudioPathKind.SOURCE)

    assert len(_tracks.tracks()) == 1


def test_local_audio_no_feed_accounts_for_is_listed_separately(monkeypatch, tmp_path) -> None:
    """A bootstrapped seed still has to be offerable, so it lands under "on disk"."""
    _remember_feed()
    _stub_feed(monkeypatch, [_episode("Ep 1")], dest=tmp_path)
    seed = tmp_path / "seed.mp3"
    seed.write_bytes(b"audio")
    _cache.remember(seed, label="Seed", kind=AudioPathKind.SAMPLE)

    groups = {track.label: track.group for track in _tracks.tracks()}

    assert groups == {"Ep 1": "Show", "Seed": "on disk"}


def test_samples_are_never_remote(monkeypatch, tmp_path) -> None:
    """Feeds carry episodes, not reference clips, so --sample skips them entirely."""
    _remember_feed()
    _stub_feed(monkeypatch, [_episode("Ep 1")], dest=tmp_path)
    seed = tmp_path / "seed.mp3"
    seed.write_bytes(b"audio")
    _cache.remember(seed, label="Seed", kind=AudioPathKind.SAMPLE)

    assert [t.label for t in _tracks.tracks(AudioPathKind.SAMPLE)] == ["Seed"]


def test_indexed_audio_that_has_vanished_is_not_offered(monkeypatch, tmp_path) -> None:
    """A file deleted behind partio's back must not be offered as available."""
    _stub_feed(monkeypatch, [], dest=tmp_path)
    _cache.remember(tmp_path / "gone.mp3", label="Gone", kind=AudioPathKind.SOURCE)

    assert _tracks.tracks() == []


# -- fetching ----------------------------------------------------------------


def test_feeds_are_fetched_once_per_session(monkeypatch, tmp_path) -> None:
    """Every prompt enumerates the library, so repeated fetches are memoized away."""
    _remember_feed()
    calls: list[str] = []

    def _fetch(url):
        calls.append(url)
        return [_episode("Ep 1")]

    monkeypatch.setattr(_tracks, "fetch_episodes", _fetch)
    monkeypatch.setattr(_tracks, "DOWNLOAD_DIR", tmp_path)
    _tracks.refresh()

    _tracks.tracks()
    _tracks.tracks()

    assert calls == ["https://feed"]


def test_refresh_forces_a_re_fetch(monkeypatch, tmp_path) -> None:
    """Dropping the memo makes the next enumeration see the feed again."""
    _remember_feed()
    calls: list[str] = []

    def _fetch(url):
        calls.append(url)
        return []

    monkeypatch.setattr(_tracks, "fetch_episodes", _fetch)
    monkeypatch.setattr(_tracks, "DOWNLOAD_DIR", tmp_path)
    _tracks.refresh()

    _tracks.tracks()
    _tracks.refresh()
    _tracks.tracks()

    assert len(calls) == 2


def test_an_unreachable_feed_does_not_break_the_library(monkeypatch, tmp_path) -> None:
    """Offline, the picker still offers whatever is already on disk."""
    _remember_feed()
    local = tmp_path / "local.mp3"
    local.write_bytes(b"audio")
    _cache.remember(local, label="Local", kind=AudioPathKind.SOURCE)

    def _boom(_url):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(_tracks, "fetch_episodes", _boom)
    monkeypatch.setattr(_tracks, "DOWNLOAD_DIR", tmp_path)
    _tracks.refresh()

    assert [track.label for track in _tracks.tracks()] == ["Local"]


# -- rendering ---------------------------------------------------------------


def test_detail_shows_an_episode_date_and_size() -> None:
    """The dimmed column carries the publication date and enclosure size."""
    track = _tracks.Track(
        label="Ep",
        path=Path("ep.mp3"),
        kind=AudioPathKind.SOURCE,
        group="Show",
        episode=_episode("Ep", published=datetime(2026, 7, 19, tzinfo=UTC), size=40_668_000),
    )

    assert track.detail == "2026-07-19   38.8 MB"


def test_detail_tolerates_missing_episode_metadata() -> None:
    """A feed that declares neither date nor length still renders."""
    track = _tracks.Track(
        label="Ep",
        path=Path("ep.mp3"),
        kind=AudioPathKind.SOURCE,
        group="Show",
        episode=_episode("Ep"),
    )

    assert track.detail == ""


def test_detail_of_local_audio_is_its_path() -> None:
    """Local audio has no feed metadata, so the path is the useful detail."""
    track = _tracks.Track(
        label="Seed",
        path=Path("static/jingles/seed.mp3"),
        kind=AudioPathKind.SAMPLE,
        group="on disk",
    )

    assert track.detail == "static/jingles/seed.mp3"
