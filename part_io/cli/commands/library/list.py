"""CLI for listing remembered audio paths."""

from __future__ import annotations

import typer  # noqa: TC002

from part_io.cli.commands.library._store import default_store
from part_io.cli.output import ExitCode, _json_flag, emit, no_match
from part_io.cli.registry import command


@command("library", "list", help="List remembered audio paths.")
def list_entries(ctx: typer.Context) -> None:
    """List every remembered audio path."""
    entries = default_store().list_items()
    if not entries:
        emit(no_match("library entries"), as_json=_json_flag(ctx))
        raise SystemExit(ExitCode.NO_RESULT)

    for entry in entries:
        emit(
            f"{entry.id}  {entry.kind.value:<6}  {entry.label}  ({entry.path})",
            as_json=_json_flag(ctx),
        )
