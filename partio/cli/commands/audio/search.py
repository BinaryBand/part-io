"""CLI for finding a reference audio sample inside a longer MP3."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from partio.adapters.audio.matcher import find_audio_sample_matches
from partio.cli.output import ExitCode, _json_flag, emit, fail, match_line, no_match
from partio.cli.registry import command


@command("audio", "search", help="Find repeated occurrences of an audio sample.")
def search(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Option("--source", prompt="Source audio file", help="Longer audio file to scan."),
    ],
    sample: Annotated[
        Path,
        typer.Option("--sample", prompt="Reference sample", help="Reference sample to search for."),
    ],
    threshold: Annotated[float, typer.Option(help="Match score threshold.")] = 0.8,
) -> None:
    """Find repeated occurrences of an audio sample."""
    try:
        matches = find_audio_sample_matches(
            source_path=source,
            sample_path=sample,
            score_threshold=threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    if not matches:
        emit(no_match("matches"), as_json=_json_flag(ctx))
        raise SystemExit(ExitCode.NO_RESULT)

    for match in matches:
        emit(
            match_line(match.start_seconds, match.end_seconds, match.score),
            as_json=_json_flag(ctx),
        )
