"""CLI for forgetting a remembered audio path."""

from __future__ import annotations

from typing import Annotated

import typer

from part_io.cli.commands.library._store import default_store
from part_io.cli.output import _json_flag, emit, fail
from part_io.cli.registry import command


@command("library", "remove", help="Forget a remembered audio path.")
def remove(
    ctx: typer.Context,
    item_id: Annotated[
        str,
        typer.Option(
            "--id", prompt="Entry id", help="Id of the entry to forget (see `library list`)."
        ),
    ],
) -> None:
    """Forget a remembered audio path."""
    store = default_store()
    if store.get_item(item_id) is None:
        fail(ValueError(f"No library entry with id {item_id!r}"))
    store.remove_item(item_id)
    emit(f"Removed {item_id}", as_json=_json_flag(ctx))
