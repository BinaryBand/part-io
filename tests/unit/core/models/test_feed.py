"""Tests for the feed domain models."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from partio.core.models import DownloadPlan, FeedEpisode


def test_feed_episode_is_frozen():
    """FeedEpisode is an immutable value object."""
    episode = FeedEpisode(
        title="Ep 1",
        audio_url="https://x/1.mp3",
        guid="guid-1",
        published=datetime(2021, 1, 1, tzinfo=UTC),
    )
    frozen_field = "title"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(episode, frozen_field, "changed")


def test_download_plan_pairs_episode_with_destination():
    """DownloadPlan carries the episode, its destination path, and label."""
    episode = FeedEpisode(title="Ep 1", audio_url="https://x/1.mp3", guid="g", published=None)
    plan = DownloadPlan(episode=episode, destination_path=Path("out/ep-1.mp3"), label="Ep 1")

    assert plan.episode is episode
    assert plan.destination_path == Path("out/ep-1.mp3")
    assert plan.label == "Ep 1"
