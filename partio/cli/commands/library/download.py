"""CLI for downloading podcast episodes from an RSS feed into the library."""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from typing import Annotated

import httpx
import typer
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

from partio.adapters.feed import download_file, fetch_episodes
from partio.app.feed_ingest import plan_downloads
from partio.cli.commands.library._store import default_store
from partio.cli.output import ExitCode, _json_flag, emit, fail, no_match
from partio.cli.registry import command
from partio.core.models import DownloadPlan  # noqa: TC001
from partio.core.ports import AudioPathEntry, AudioPathKind

# Progress UI is drawn on stderr so it never mingles with --json output on stdout.
console = Console(stderr=True)


@command("library", "download", help="Download episodes from an RSS feed and remember them.")
def download(
    ctx: typer.Context,
    feed: Annotated[
        str,
        typer.Option("--feed", prompt="RSS feed URL", help="Podcast RSS/Atom feed URL."),
    ],
    count: Annotated[
        int,
        typer.Option(help="How many latest episodes to download."),
    ] = 1,
    dest: Annotated[
        Path,
        typer.Option(help="Directory to download episodes into."),
    ] = Path("static") / "downloads",
) -> None:
    """Download the latest episodes from *feed* and remember them as sources."""
    as_json = _json_flag(ctx)
    try:
        with _status(f"Fetching feed {feed}", show=not as_json):
            episodes = fetch_episodes(feed)
    except httpx.HTTPError as exc:
        fail(exc)

    if not episodes:
        emit(no_match("feed episodes"), as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    store = default_store()
    existing_paths = {entry.path for entry in store.list_items()}
    plans = plan_downloads(episodes, count=count, dest_dir=dest, existing_paths=existing_paths)
    if not plans:
        emit("Nothing new to download.", as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    if not as_json:
        noun = "episode" if len(plans) == 1 else "episodes"
        console.print(f"Downloading [bold]{len(plans)}[/bold] {noun}...")

    for index, plan in enumerate(plans, start=1):
        try:
            _download_episode(plan, position=index, total=len(plans), show_progress=not as_json)
        except httpx.HTTPError as exc:
            fail(exc)
        entry = AudioPathEntry(
            id=uuid.uuid4().hex[:8],
            path=plan.destination_path,
            label=plan.label,
            kind=AudioPathKind.SOURCE,
        )
        store.add_item(entry)
        emit(f"Downloaded {entry.label} -> {entry.path} ({entry.id})", as_json=as_json)


def _status(message: str, *, show: bool) -> contextlib.AbstractContextManager[object]:
    """A Rich spinner status when *show*, else a no-op context manager."""
    if show:
        return console.status(message)
    return contextlib.nullcontext()


def _download_episode(
    plan: DownloadPlan, *, position: int, total: int, show_progress: bool
) -> None:
    """Download one planned episode, drawing a live progress bar when enabled."""
    if not show_progress:
        download_file(url=plan.episode.audio_url, destination_path=plan.destination_path)
        return

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
        task = progress.add_task(f"[{position}/{total}] {plan.label}", total=None)

        def _advance(downloaded: int, size: int | None) -> None:
            progress.update(task, completed=downloaded, total=size)

        download_file(
            url=plan.episode.audio_url,
            destination_path=plan.destination_path,
            on_progress=_advance,
        )
