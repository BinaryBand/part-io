"""CLI for forgetting a remembered podcast feed."""

from __future__ import annotations

from typing import Annotated

import typer

from partio.cli.commands.feed._store import default_store
from partio.cli.output import _json_flag, emit, fail
from partio.cli.registry import command


@command("feed", "remove", help="Forget a remembered feed.")
def remove(
    ctx: typer.Context,
    feed_id: Annotated[
        str,
        typer.Option("--id", prompt="Feed id", help="Id of the feed to forget (see `feed list`)."),
    ],
) -> None:
    """Forget a remembered podcast feed."""
    store = default_store()
    if store.get_item(feed_id) is None:
        fail(ValueError(f"No feed with id {feed_id!r}"))
    store.remove_item(feed_id)
    emit(f"Removed {feed_id}", as_json=_json_flag(ctx))
