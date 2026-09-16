"""Request/response schemas for the library (feeds, tracks, downloads)."""

from __future__ import annotations

from datetime import datetime

from partio.cli.library import Track
from pydantic import BaseModel


class FeedOut(BaseModel):
    """A remembered podcast feed."""

    id: str
    url: str
    label: str


class FeedCreate(BaseModel):
    """Request body for remembering a new feed."""

    url: str
    label: str | None = None


class FeedEpisodeOut(BaseModel):
    """A parsed feed episode, as offered by a track that isn't on disk yet."""

    title: str
    audio_url: str
    guid: str
    published: datetime | None
    size_bytes: int | None


class TrackOut(BaseModel):
    """One selectable piece of audio: a feed episode or something already local."""

    label: str
    path: str
    kind: str
    group: str
    on_disk: bool
    mark: str
    detail: str
    episode: FeedEpisodeOut | None


def track_to_out(track: Track) -> TrackOut:
    """Translate a `partio.cli.library.Track` into its HTTP representation."""
    episode = (
        FeedEpisodeOut(
            title=track.episode.title,
            audio_url=track.episode.audio_url,
            guid=track.episode.guid,
            published=track.episode.published,
            size_bytes=track.episode.size_bytes,
        )
        if track.episode is not None
        else None
    )
    return TrackOut(
        label=track.label,
        path=str(track.path),
        kind=track.kind.value,
        group=track.group,
        on_disk=track.on_disk,
        mark=track.mark,
        detail=track.detail,
        episode=episode,
    )
