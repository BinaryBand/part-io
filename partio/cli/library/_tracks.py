"""Enumerating the library: what it can offer, regardless of what it holds.

A :class:`Track` is a piece of audio the user may choose.  It carries the path
the audio *would* occupy, so an episode nobody has downloaded is just as
selectable as one sitting on disk -- the difference shows up as a glyph in the
picker and as work inside :func:`partio.cli.library.ensure_local`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from partio.adapters.feed import fetch_feed_content, parse_feed
from partio.app.feed_ingest import destination_for
from partio.cli.library._cache import cached
from partio.cli.library._feeds import feeds
from partio.core.ports import AudioPathKind

if TYPE_CHECKING:
    from partio.core.models import FeedEpisode

DOWNLOAD_DIR = Path("static") / "downloads"

# How much of a feed document to read for the default, partial read.  Both
# costs are linear in bytes and podcast feeds are newest-first, so this is the
# whole trade: 256 KB of The Daily is its ~40 newest episodes in well under a
# second, where all 17.5 MB is 2923 episodes in sixteen -- sixteen seconds in
# front of every prompt.  :func:`tracks` takes ``full=True`` to pay it
# deliberately, on request.
HEAD_BYTES = 256 * 1024

ON_DISK_MARK = "●"
REMOTE_MARK = "○"
MARK_LEGEND = f"{ON_DISK_MARK} on disk   {REMOTE_MARK} downloads when picked"

_LOCAL_GROUP = "on disk"
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class Track:
    """One selectable piece of audio, on disk or still only in a feed.

    *path* is where the audio lives or will live; *episode* is what to download
    to put it there, and is ``None`` for audio that only ever existed locally.
    *group* is the heading a picker files the row under -- the feed's name, or
    "on disk" for local-only audio.
    """

    label: str
    path: Path
    kind: AudioPathKind
    group: str
    episode: FeedEpisode | None = None

    @property
    def on_disk(self) -> bool:
        """Whether the bytes are already here.

        Answered from the filesystem rather than the index, so a file deleted
        behind partio's back reads as missing instead of as a broken promise.
        """
        return self.path.exists()

    @property
    def mark(self) -> str:
        """Listing glyph: filled when choosing this costs nothing."""
        return ON_DISK_MARK if self.on_disk else REMOTE_MARK

    @property
    def detail(self) -> str:
        """The dimmed trailing column: publication date and size, or the path."""
        if self.episode is None:
            return str(self.path)
        parts = []
        if self.episode.published is not None:
            parts.append(self.episode.published.date().isoformat())
        if self.episode.size_bytes:
            parts.append(f"{self.episode.size_bytes / _BYTES_PER_MB:.1f} MB")
        return "   ".join(parts)


@functools.cache
def _content(url: str, *, full: bool) -> bytes:
    """Download *url*'s feed document once per session, ranged unless *full*.

    An unreachable feed contributes nothing rather than failing the prompt, so
    partio still works offline over whatever is already on disk.
    """
    try:
        return fetch_feed_content(url, max_bytes=None if full else HEAD_BYTES)
    except httpx.HTTPError:
        return b""


@functools.cache
def _episodes(url: str, *, full: bool) -> tuple[FeedEpisode, ...]:
    """Parse *url*'s episodes, all of them only when *full*.

    Memoized per depth because every prompt re-enumerates: the partial read is
    paid once, and so is the full one if the user ever asks to expand.  The
    parse is bounded here as well as at the request, since a server free to
    ignore a range header is free to send back the whole catalogue.  Cutting
    mid-element is fine -- feedparser recovers, losing at most the one entry
    the cut landed in.
    """
    content = _content(url, full=full)
    return tuple(parse_feed(content if full else content[:HEAD_BYTES]))


def refresh() -> None:
    """Forget the memoized feed reads so the next enumeration re-fetches."""
    _content.cache_clear()
    _episodes.cache_clear()


def has_more() -> bool:
    """Whether any feed likely holds episodes older than the partial read.

    A read that filled its whole budget is taken to have been cut short.  Free
    once the library has been enumerated -- the documents it measures are
    already in hand.
    """
    return any(len(_content(feed.url, full=False)) >= HEAD_BYTES for feed in feeds())


def tracks(kind: AudioPathKind | None = None, *, full: bool = False) -> list[Track]:
    """Everything the library can offer as *kind*, feeds first.

    Feed episodes are listed whether or not they have been downloaded -- that
    is what makes the library virtual.  Only the newest of each feed are read
    unless *full*, because a whole back catalogue costs seconds to parse and is
    almost never what the next command wants.  Local audio no feed accounts for
    (bootstrapped seeds, manually entered paths) follows under "on disk".
    Samples are never remote: feeds carry episodes, not reference clips.
    """
    listed = [] if kind is AudioPathKind.SAMPLE else _feed_tracks(full=full)
    claimed = {track.path for track in listed}
    listed.extend(
        Track(label=entry.label, path=entry.path, kind=entry.kind, group=_LOCAL_GROUP)
        for entry in cached(kind)
        if entry.path not in claimed and entry.path.exists()
    )
    return listed


def _feed_tracks(*, full: bool) -> list[Track]:
    """A track for every episode read from every remembered feed, in feed order."""
    return [
        Track(
            label=episode.title or "(untitled)",
            path=destination_for(episode, dest_dir=DOWNLOAD_DIR),
            kind=AudioPathKind.SOURCE,
            group=feed.label,
            episode=episode,
        )
        for feed in feeds()
        for episode in _episodes(feed.url, full=full)
    ]
