"""In-memory path -> Track cache backing the picker's two-step flow.

``GET /api/tracks`` lists candidates, including feed episodes that aren't on
disk yet; ``POST /api/tracks/download`` looks a chosen path up here rather
than requiring the client to resend the whole episode payload. A plain
module-level dict is enough: this backend runs as one local, single-process,
single-user server.
"""

from __future__ import annotations

from pathlib import Path

from partio.adapters.feed import download_file
from partio.cli.library import Track, remember
from partio.core.ports import AudioPathKind

from app.services.jobs import Report

_by_path: dict[Path, Track] = {}


def remember_tracks(found: list[Track]) -> None:
    """Record *found* so their paths can be looked up by a download request."""
    for track in found:
        _by_path[track.path] = track


def lookup(path: Path) -> Track | None:
    """Return the Track most recently listed at *path*, if any."""
    return _by_path.get(path)


def download_with_progress(track: Track, report: Report) -> Path:
    """Materialize *track* locally, reporting real byte progress as it downloads.

    Deliberately bypasses ``partio.cli.library.ensure_local``: that function
    draws its own Rich progress bar for a terminal and exposes no progress
    callback of its own. Calling the lower-level adapter directly, the same
    way ``ensure_local`` does internally, gets the same
    ``partio.adapters.feed.download_file`` byte-accurate progress hook
    reported to the GUI instead.
    """
    if track.on_disk:
        return track.path
    if track.episode is None:
        raise FileNotFoundError(f"No longer on disk: {track.path}")

    track.path.parent.mkdir(parents=True, exist_ok=True)

    def on_progress(downloaded: int, total: int | None) -> None:
        report({"type": "progress", "bytes": downloaded, "total": total})

    download_file(url=track.episode.audio_url, destination_path=track.path, on_progress=on_progress)
    remember(track.path, label=track.label, kind=AudioPathKind.SOURCE)
    return track.path
