"""CLI for saving a reusable cut rule."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import typer

from partio.cli.library import cut_rule_store
from partio.cli.output import _json_flag, emit
from partio.cli.registry import command
from partio.core.ports import CutRuleEntry


@command("rule", "add", help="Save a reusable opening/closing jingle pair for `audio cut`.")
def add(
    ctx: typer.Context,
    label: Annotated[
        str,
        typer.Option("--label", prompt="Rule name", help="Friendly name, e.g. the show's title."),
    ],
    opening: Annotated[
        Path,
        typer.Option("--opening", prompt="Opening jingle", help="Jingle that starts a break."),
    ],
    closing: Annotated[
        Path,
        typer.Option("--closing", prompt="Closing jingle", help="Jingle that ends a break."),
    ],
) -> None:
    """Save an opening/closing jingle pair so `audio cut --rule` can reuse it.

    Point --opening/--closing at existing sample clips -- typically ones
    written by `audio bootstrap` -- rather than raw episode audio.
    """
    store = cut_rule_store()
    entry = CutRuleEntry(
        id=uuid.uuid4().hex[:8], label=label, opening_path=opening, closing_path=closing
    )
    store.add_item(entry)
    emit(f"Saved cut rule {entry.label} as {entry.id}", as_json=_json_flag(ctx))
