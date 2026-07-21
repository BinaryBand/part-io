"""Tests for the feed domain models."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from partio.core.models import FeedEpisode


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


def test_feed_episode_size_is_optional():
    """A feed that declares no enclosure length still parses into an episode."""
    episode = FeedEpisode(title="Ep 1", audio_url="https://x/1.mp3", guid="g", published=None)

    assert episode.size_bytes is None
