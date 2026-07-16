"""CLI for remembering an audio path (source recording or reference sample)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import typer

from part_io.cli.commands.library._store import default_store
from part_io.cli.output import _json_flag, emit, fail
from part_io.cli.registry import command
from part_io.core.ports import AudioPathEntry, AudioPathKind


@command("library", "add", help="Remember an audio path for reuse across commands.")
def add(
    ctx: typer.Context,
    path: Annotated[
        Path,
        typer.Option("--path", prompt="Audio file path", help="Audio file to remember."),
    ],
    label: Annotated[
        str | None,
        typer.Option(help="Friendly name (defaults to the file stem)."),
    ] = None,
    kind: Annotated[
        AudioPathKind,
        typer.Option(help="Whether this is a long source recording or a short reference sample."),
    ] = AudioPathKind.SOURCE,
) -> None:
    """Remember an audio path so it can be reused across commands."""
    if not path.exists():
        fail(FileNotFoundError(f"Audio file not found: {path}"))

    entry = AudioPathEntry(
        id=uuid.uuid4().hex[:8],
        path=path,
        label=label or path.stem,
        kind=kind,
    )
    default_store().add_item(entry)
    emit(f"Remembered {entry.label} ({entry.kind.value}) as {entry.id}", as_json=_json_flag(ctx))
