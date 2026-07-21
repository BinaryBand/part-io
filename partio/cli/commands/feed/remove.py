"""CLI for forgetting a remembered podcast feed."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from partio.cli.library import feed_store, feeds
from partio.cli.output import ExitCode, _json_flag, emit, fail
from partio.cli.registry import command
from partio.cli.select import GoBack, Option, select_one

console = Console(stderr=True)


@command("feed", "remove", help="Forget a remembered feed.")
def remove(
    ctx: typer.Context,
    feed_id: Annotated[
        str | None,
        typer.Option("--id", help="Id of the feed to forget (default: pick one)."),
    ] = None,
) -> None:
    """Forget a remembered podcast feed.

    The id is optional so nobody has to look one up: with no ``--id`` this
    picks from the remembered feeds by name.  Only the feed is forgotten --
    anything already downloaded from it stays on disk.
    """
    as_json = _json_flag(ctx)
    store = feed_store()
    chosen = feed_id if feed_id is not None else _pick_feed(as_json=as_json)

    if store.get_item(chosen) is None:
        fail(ValueError(f"No feed with id {chosen!r}"))
    store.remove_item(chosen)
    emit(f"Removed {chosen}", as_json=as_json)


def _pick_feed(*, as_json: bool) -> str:
    """Choose which remembered feed to forget, by name rather than by id."""
    remembered = feeds()
    if not remembered:
        emit("No feeds remembered yet -- add one with `partio feed add`.", as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    options = [
        Option(title=entry.label, value=entry.id, help=entry.url, group="remembered feeds")
        for entry in remembered
    ]
    chosen = select_one("Pick a feed to forget", options, console=console)
    if chosen is None or isinstance(chosen, GoBack):
        # The picker is this command's first screen, so esc backs out of it.
        emit("Cancelled.", as_json=as_json)
        raise SystemExit(ExitCode.OK)
    return chosen
