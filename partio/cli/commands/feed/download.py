"""CLI for downloading podcast episodes from a remembered feed."""

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
from partio.app.feed_ingest import destination_for, plan_downloads
from partio.cli.commands.feed._store import default_store as feed_store
from partio.cli.commands.library._store import default_store as library_store
from partio.cli.output import ExitCode, _json_flag, emit, fail, no_match
from partio.cli.registry import command
from partio.cli.select import GoBack, Option, select_many, select_one
from partio.core.models import DownloadPlan, FeedEpisode  # noqa: TC001
from partio.core.ports import AudioPathEntry, AudioPathKind, ItemStore

# Progress UI is drawn on stderr so it never mingles with --json output on stdout.
console = Console(stderr=True)

_BYTES_PER_MB = 1024 * 1024


@command("feed", "download", help="Download episodes from a remembered feed.")
def download(
    ctx: typer.Context,
    url: Annotated[
        str | None,
        typer.Option("--url", help="Feed URL (default: pick from remembered feeds)."),
    ] = None,
    count: Annotated[
        int | None,
        typer.Option(help="Skip the episode picker and take this many latest episodes."),
    ] = None,
    dest: Annotated[
        Path,
        typer.Option(help="Directory to download episodes into."),
    ] = Path("static") / "downloads",
) -> None:
    """Download episodes from a feed and remember them as library sources."""
    as_json = _json_flag(ctx)
    store = library_store()

    # Loop so esc in the episode list steps back to the feed picker, rather
    # than dropping the user out of the command entirely.
    while True:
        feed_url = url if url is not None else _pick_feed(as_json=as_json)

        try:
            with _status(f"Fetching feed {feed_url}", show=not as_json):
                episodes = fetch_episodes(feed_url)
        except httpx.HTTPError as exc:
            fail(exc)

        if not episodes:
            emit(no_match("feed episodes"), as_json=as_json)
            raise SystemExit(ExitCode.NO_RESULT)

        existing_paths = {entry.path for entry in store.list_items()}
        if count is not None:
            break

        chosen = _pick_episodes(episodes, dest=dest, existing_paths=existing_paths)
        if isinstance(chosen, GoBack):
            if url is not None:
                # An explicit --url means there is no feed picker to return to.
                emit("Cancelled.", as_json=as_json)
                raise SystemExit(ExitCode.OK)
            continue
        episodes = chosen
        break

    plans = plan_downloads(
        episodes,
        count=count if count is not None else len(episodes),
        dest_dir=dest,
        existing_paths=existing_paths,
    )
    if not plans:
        emit("Nothing new to download.", as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    _download_all(plans, store=store, as_json=as_json)


def _pick_feed(*, as_json: bool) -> str:
    """Choose one of the remembered feeds, or explain how to add one."""
    feeds = feed_store().list_items()
    if not feeds:
        emit("No feeds remembered yet -- add one with `partio feed add`.", as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    options = [
        Option(title=entry.label, value=entry.url, help=entry.url, group="remembered feeds")
        for entry in feeds
    ]
    chosen = select_one("Pick a feed", options, console=console)
    if chosen is None or isinstance(chosen, GoBack):
        # The feed picker is this command's first screen: esc backs out of it.
        emit("Cancelled.", as_json=as_json)
        raise SystemExit(ExitCode.OK)
    return chosen


def _pick_episodes(
    episodes: list[FeedEpisode], *, dest: Path, existing_paths: set[Path]
) -> list[FeedEpisode] | GoBack:
    """Let the user check which episodes to download.

    Returns :data:`GO_BACK` when the user pressed esc, so the caller can
    redisplay the feed picker.
    """
    options = [
        Option(
            title=episode.title or "(untitled)",
            value=episode,
            help=_episode_detail(episode),
            disabled=(
                "already in library"
                if destination_for(episode, dest_dir=dest) in existing_paths
                else None
            ),
        )
        for episode in episodes
    ]
    chosen = select_many("Select episodes to download", options, console=console)
    if chosen is None:
        emit("Cancelled.")
        raise SystemExit(ExitCode.OK)
    return chosen


def _episode_detail(episode: FeedEpisode) -> str:
    """Render the dimmed right-hand column: publication date and size."""
    parts = []
    if episode.published is not None:
        parts.append(episode.published.date().isoformat())
    if episode.size_bytes:
        parts.append(f"{episode.size_bytes / _BYTES_PER_MB:.1f} MB")
    return "   ".join(parts)


def _download_all(
    plans: list[DownloadPlan], *, store: ItemStore[AudioPathEntry], as_json: bool
) -> None:
    """Download each planned episode and remember it as a library source."""
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
