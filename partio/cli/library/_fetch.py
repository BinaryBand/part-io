"""Turning a chosen track into bytes on disk -- the "download on request" half.

Kept separate from enumeration so the expensive thing happens to exactly one
track: whichever the user picked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from partio.adapters.feed import download_file
from partio.cli.library._cache import remember
from partio.core.ports import AudioPathKind

if TYPE_CHECKING:
    from pathlib import Path

    from partio.cli.library._tracks import Track

# Progress is drawn on stderr so it never mingles with --json output on stdout.
console = Console(stderr=True)


def ensure_local(track: Track) -> Path | None:
    """Return *track*'s local path, downloading it first if it is not there.

    Returns ``None`` when the audio could not be produced -- a failed download,
    or a local file that has since disappeared.  The reason is reported here;
    callers treat ``None`` as "ask again" rather than as a fatal error, so a
    flaky network costs a retry instead of the whole session.
    """
    if track.on_disk:
        return track.path
    if track.episode is None:
        console.print(f"[red]No longer on disk:[/red] {track.path}")
        return None

    track.path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download(url=track.episode.audio_url, destination=track.path, label=track.label)
    except httpx.HTTPError as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        return None

    remember(track.path, label=track.label, kind=AudioPathKind.SOURCE)
    console.print(f"Downloaded [bold]{track.label}[/bold] -> {track.path}")
    return track.path


def _download(*, url: str, destination: Path, label: str) -> None:
    """Stream *url* to *destination* behind a live progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(label, total=None)

        def _advance(downloaded: int, size: int | None) -> None:
            progress.update(task, completed=downloaded, total=size)

        download_file(url=url, destination_path=destination, on_progress=_advance)
