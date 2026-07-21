"""Tests for the pure feed destination-path logic."""

from __future__ import annotations

from pathlib import Path

from partio.core.feed_ingest import destination_for
from partio.core.models import FeedEpisode


def _episode(title: str) -> FeedEpisode:
    return FeedEpisode(title=title, audio_url=f"https://x/{title}.mp3", guid=title, published=None)


def test_destination_slugifies_the_title():
    """An episode's destination is derived from its title, inside the given dir."""
    assert destination_for(_episode("Ep 3"), dest_dir=Path("out")) == Path("out") / "ep-3.mp3"


def test_destination_is_stable_across_calls():
    """Stability is what lets a second look tell "downloaded" from "not yet"."""
    episode = _episode("Ep 3")

    assert destination_for(episode, dest_dir=Path("out")) == destination_for(
        episode, dest_dir=Path("out")
    )


def test_destination_of_an_untitled_episode_is_still_usable():
    """An episode with no title still gets a usable filename."""
    assert destination_for(_episode(""), dest_dir=Path("out")) == Path("out") / "untitled.mp3"
