"""Feed, track, and download endpoints -- the library/picker surface."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from partio.adapters.feed import fetch_feed_title
from partio.cli.library import feed_store, feeds, has_more, refresh, tracks
from partio.core.ports import AudioPathKind, FeedEntry
from pydantic import BaseModel

from app.models.library import FeedCreate, FeedOut, TrackOut, track_to_out
from app.services.jobs import sse_encode, stream_job
from app.services.tracks_cache import download_with_progress, lookup, remember_tracks

router = APIRouter(prefix="/api", tags=["library"])


class LibraryStatusOut(BaseModel):
    """Whether any remembered feed's partial read was cut short."""

    has_more: bool


class TrackDownloadRequest(BaseModel):
    """Request body for downloading a previously-listed track."""

    path: str


@router.get("/feeds")
def list_feeds() -> list[FeedOut]:
    """List every remembered podcast feed."""
    return [FeedOut(id=entry.id, url=entry.url, label=entry.label) for entry in feeds()]


@router.post("/feeds", status_code=201)
def add_feed(body: FeedCreate) -> FeedOut:
    """Remember a podcast feed, fetching its title when no label is given."""
    store = feed_store()
    if any(entry.url == body.url for entry in store.list_items()):
        raise HTTPException(status_code=409, detail=f"Feed already remembered: {body.url}")

    label = body.label
    if label is None:
        try:
            label = fetch_feed_title(body.url)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    entry = FeedEntry(id=uuid.uuid4().hex[:8], url=body.url, label=label or body.url)
    store.add_item(entry)
    return FeedOut(id=entry.id, url=entry.url, label=entry.label)


@router.delete("/feeds/{feed_id}", status_code=204)
def remove_feed(feed_id: str) -> None:
    """Forget a remembered feed. Anything already downloaded stays on disk."""
    store = feed_store()
    if store.get_item(feed_id) is None:
        raise HTTPException(status_code=404, detail=f"No feed with id {feed_id!r}")
    store.remove_item(feed_id)


@router.post("/feeds/refresh", status_code=204)
def refresh_feeds() -> None:
    """Forget memoized feed reads so the next listing re-fetches.

    partio.cli.library's feed reads are cached for the process lifetime
    (fine for the CLI, one process per invocation) -- this backend is
    long-running, so the Library screen's refresh affordance must call this
    before re-listing, or it would keep serving the same stale read forever.
    """
    refresh()


@router.get("/tracks")
def list_tracks(
    kind: Annotated[
        AudioPathKind | None, Query(description="Restrict to 'source' or 'sample'.")
    ] = None,
    full: Annotated[
        bool, Query(description="Read every episode of every feed, not just the newest.")
    ] = False,
) -> list[TrackOut]:
    """List everything the library can offer, feed episodes included.

    Populates the download-path cache as a side effect, so a client can pick
    one of these paths in a follow-up `/api/tracks/download` call.
    """
    found = tracks(kind, full=full)
    remember_tracks(found)
    return [track_to_out(track) for track in found]


@router.get("/tracks/status")
def tracks_status() -> LibraryStatusOut:
    """Whether any feed's partial read was cut short (the "load every episode" row)."""
    return LibraryStatusOut(has_more=has_more())


@router.post("/tracks/download")
async def download_track(body: TrackDownloadRequest) -> StreamingResponse:
    """Download a track chosen from a prior `/api/tracks` listing, as an SSE stream.

    Streams real byte-accurate progress (`partio.adapters.feed.download_file`'s
    own progress hook), then a final `result` event with the local path, or an
    `error` event -- never raises through the stream once it has started.
    """
    track = lookup(Path(body.path))
    if track is None:
        raise HTTPException(status_code=404, detail="Unknown track path; list /api/tracks first.")

    async def events() -> AsyncIterator[str]:
        async for event in stream_job(
            lambda report: download_with_progress(track, report),
            on_success=lambda result_path: {"type": "result", "path": str(result_path)},
        ):
            yield sse_encode(event)

    return StreamingResponse(events(), media_type="text/event-stream")
