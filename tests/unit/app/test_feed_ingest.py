"""Tests for the pure feed download-planning logic."""

from __future__ import annotations

from pathlib import Path

from partio.app.feed_ingest import plan_downloads
from partio.core.models import FeedEpisode


def _episode(title: str) -> FeedEpisode:
    return FeedEpisode(title=title, audio_url=f"https://x/{title}.mp3", guid=title, published=None)


def test_plan_respects_count_and_feed_order():
    """plan_downloads() keeps feed order and stops at *count*."""
    episodes = [_episode("Ep 3"), _episode("Ep 2"), _episode("Ep 1")]

    plans = plan_downloads(episodes, count=2, dest_dir=Path("out"), existing_paths=set())

    assert [plan.episode.title for plan in plans] == ["Ep 3", "Ep 2"]
    assert plans[0].destination_path == Path("out") / "ep-3.mp3"
    assert plans[0].label == "Ep 3"


def test_plan_skips_already_downloaded():
    """Episodes whose destination already exists in the library are skipped."""
    episodes = [_episode("Ep 2"), _episode("Ep 1")]
    existing = {Path("out") / "ep-2.mp3"}

    plans = plan_downloads(episodes, count=5, dest_dir=Path("out"), existing_paths=existing)

    assert [plan.episode.title for plan in plans] == ["Ep 1"]


def test_plan_labels_untitled_episode_from_stem():
    """An episode with no title still gets a usable slug and label."""
    plans = plan_downloads([_episode("")], count=1, dest_dir=Path("out"), existing_paths=set())

    assert plans[0].destination_path == Path("out") / "untitled.mp3"
    assert plans[0].label == "untitled"
