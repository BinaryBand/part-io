"""CLI for forgetting a saved cut rule."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from partio.cli.library import cut_rule_store, cut_rules
from partio.cli.output import ExitCode, _json_flag, emit, fail
from partio.cli.registry import command
from partio.cli.select import GoBack, Option, select_one

console = Console(stderr=True)


@command("rule", "remove", help="Forget a saved cut rule.")
def remove(
    ctx: typer.Context,
    rule_id: Annotated[
        str | None,
        typer.Option("--id", help="Id of the rule to forget (default: pick one)."),
    ] = None,
) -> None:
    """Forget a saved cut rule.

    The id is optional so nobody has to look one up: with no ``--id`` this
    picks from the saved rules by name.
    """
    as_json = _json_flag(ctx)
    store = cut_rule_store()
    chosen = rule_id if rule_id is not None else _pick_rule(as_json=as_json)

    if store.get_item(chosen) is None:
        fail(ValueError(f"No cut rule with id {chosen!r}"))
    store.remove_item(chosen)
    emit(f"Removed {chosen}", as_json=as_json)


def _pick_rule(*, as_json: bool) -> str:
    """Choose which saved cut rule to forget, by name rather than by id."""
    saved = cut_rules()
    if not saved:
        emit("No cut rules saved yet -- add one with `partio rule add`.", as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    options = [
        Option(
            title=entry.label,
            value=entry.id,
            help=f"{entry.opening_path} / {entry.closing_path}",
            group="saved rules",
        )
        for entry in saved
    ]
    chosen = select_one("Pick a cut rule to forget", options, console=console)
    if chosen is None or isinstance(chosen, GoBack):
        # The picker is this command's first screen, so esc backs out of it.
        emit("Cancelled.", as_json=as_json)
        raise SystemExit(ExitCode.OK)
    return chosen
