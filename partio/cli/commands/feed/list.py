"""CLI for listing remembered podcast feeds."""

from __future__ import annotations

import typer  # noqa: TC002

from partio.cli.commands.feed._store import default_store
from partio.cli.output import ExitCode, _json_flag, emit, no_match
from partio.cli.registry import command


@command("feed", "list", help="List remembered feeds.")
def list_feeds(ctx: typer.Context) -> None:
    """List every remembered podcast feed."""
    entries = default_store().list_items()
    if not entries:
        emit(no_match("feeds"), as_json=_json_flag(ctx))
        raise SystemExit(ExitCode.NO_RESULT)

    for entry in entries:
        emit(f"{entry.id}  {entry.label}  ({entry.url})", as_json=_json_flag(ctx))
