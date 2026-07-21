"""Tests for the `feed list` CLI command."""

from __future__ import annotations

import pytest

from partio.cli.commands.feed.list import list_feeds
from partio.cli.library import _cache, _tracks
from partio.cli.library import _feeds as store_module
from partio.cli.output import ExitCode
from partio.core.models import FeedEpisode
from partio.core.ports import AudioPathKind, FeedEntry


@pytest.fixture(autouse=True)
def _feeds_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")
    monkeypatch.setattr(_tracks, "DOWNLOAD_DIR", tmp_path / "downloads")


def _episode(title: str) -> FeedEpisode:
    return FeedEpisode(title=title, audio_url=f"https://x/{title}.mp3", guid=title, published=None)


def _stub_feed(monkeypatch, episodes) -> None:
    items = "".join(
        f"<item><title>{e.title}</title><guid>{e.title}</guid>"
        f'<enclosure url="{e.audio_url}" type="audio/mpeg" length="1000"/></item>'
        for e in episodes
    )
    document = f"<rss version='2.0'><channel><title>S</title>{items}</channel></rss>".encode()
    monkeypatch.setattr(_tracks, "fetch_feed_content", lambda _url, **_kw: document)
    _tracks.refresh()


def test_list_shows_each_feed_and_its_episodes(monkeypatch, capsys):
    """The listing is the library: feeds as headings, their episodes beneath."""
    store_module.feed_store().add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    _stub_feed(monkeypatch, [_episode("Ep 2"), _episode("Ep 1")])

    list_feeds(ctx=None)

    out = capsys.readouterr().out
    assert "Show A" in out
    assert "Ep 2" in out
    assert "Ep 1" in out


def test_list_marks_what_is_already_downloaded(monkeypatch, capsys, tmp_path):
    """The glyph is the whole point: it says which rows would cost a download."""
    store_module.feed_store().add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    _stub_feed(monkeypatch, [_episode("Ep 1")])
    downloaded = tmp_path / "downloads" / "ep-1.mp3"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"audio")

    list_feeds(ctx=None)

    out = capsys.readouterr().out
    assert f"{_tracks.ON_DISK_MARK} Ep 1" in out
    assert _tracks.MARK_LEGEND in out


def test_list_includes_local_audio_no_feed_accounts_for(monkeypatch, capsys, tmp_path):
    """Bootstrapped seeds are part of the library, so the listing shows them too."""
    _stub_feed(monkeypatch, [])
    seed = tmp_path / "seed.mp3"
    seed.write_bytes(b"audio")
    _cache.remember(seed, label="Seed", kind=AudioPathKind.SAMPLE)

    list_feeds(ctx=None)

    out = capsys.readouterr().out
    assert "on disk" in out
    assert "Seed" in out


def test_list_truncates_a_long_back_catalogue(monkeypatch, capsys):
    """A feed with thousands of episodes must not print thousands of lines."""
    store_module.feed_store().add_item(FeedEntry(id="f1", url="https://a", label="Show A"))
    _stub_feed(monkeypatch, [_episode(f"Ep {n}") for n in range(60)])

    list_feeds(ctx=None)

    out = capsys.readouterr().out
    assert "Ep 0" in out
    assert "Ep 59" not in out
    assert "... and 40 more" in out


def test_list_empty_library_points_at_feed_add(capsys):
    """With nothing remembered and nothing on disk, say how to start."""
    with pytest.raises(SystemExit) as exc_info:
        list_feeds(ctx=None)

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert "feed add" in capsys.readouterr().out
