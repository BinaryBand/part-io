"""CLI for listing saved cut rules."""

from __future__ import annotations

import typer

from partio.cli.library import cut_rules
from partio.cli.output import ExitCode, _json_flag, emit
from partio.cli.registry import command


@command("rule", "list", help="List saved cut rules.")
def list_rules(ctx: typer.Context) -> None:
    """List every saved cut rule, by id, name, and its two jingle paths."""
    as_json = _json_flag(ctx)
    rules = cut_rules()
    if not rules:
        emit("No cut rules saved yet -- add one with `partio rule add`.", as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    if as_json:
        emit(
            {
                "rules": [
                    {
                        "id": entry.id,
                        "label": entry.label,
                        "opening_path": str(entry.opening_path),
                        "closing_path": str(entry.closing_path),
                    }
                    for entry in rules
                ]
            },
            as_json=True,
        )
        return
    emit(
        [
            f"{entry.id}  {entry.label}  ({entry.opening_path} / {entry.closing_path})"
            for entry in rules
        ]
    )
