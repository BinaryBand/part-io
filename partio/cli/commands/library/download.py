"""CLI for downloading podcast episodes from an RSS feed into the library."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import httpx
import typer

from partio.adapters.feed import download_file, fetch_episodes
from partio.app.feed_ingest import plan_downloads
from partio.cli.commands.library._store import default_store
from partio.cli.output import ExitCode, _json_flag, emit, fail, no_match
from partio.cli.registry import command
from partio.core.ports import AudioPathEntry, AudioPathKind


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

    for plan in plans:
        try:
            download_file(url=plan.episode.audio_url, destination_path=plan.destination_path)
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
