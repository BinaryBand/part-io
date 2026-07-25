"""CLI for cutting an ad break bounded by an opening and a closing jingle.

Locates the single best occurrence of each jingle in the episode, then removes
the whole span from the start of the opening jingle to the end of the closing
one -- both stingers included -- and writes the surviving audio to a new MP3.
Use ``--min-prominence`` to reject weak jingle peaks (the same z-score used by
``audio locate``); this removes exactly one break per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from partio.adapters.audio.clips import audio_duration_seconds, cut_segments
from partio.adapters.audio.matcher import BestMatch, find_best_sample_match
from partio.cli.output import ExitCode, _json_flag, cut_summary, emit, fail
from partio.cli.registry import command
from partio.core.audio_cut import plan_cut


def _default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_cut{source.suffix or '.mp3'}")


def _resolve_output(*, source: Path, output: Path | None, overwrite: bool) -> Path:
    output_path = output if output is not None else _default_output_path(source)
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {output_path} (use --overwrite)")
    return output_path


def _require_jingle(
    *, ctx: typer.Context | None, match: BestMatch | None, min_prominence: float, label: str
) -> BestMatch:
    if match is None or match.prominence < min_prominence:
        emit(f"No confident match for the {label} jingle.", as_json=_json_flag(ctx))
        raise SystemExit(ExitCode.NO_RESULT)
    return match


@command("audio", "cut", help="Remove the ad break between an opening and closing jingle.")
def cut(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Option("--source", prompt="Source audio file", help="Episode to edit."),
    ],
    opening: Annotated[
        Path,
        typer.Option("--opening", prompt="Opening jingle", help="Jingle that starts the break."),
    ],
    closing: Annotated[
        Path,
        typer.Option("--closing", prompt="Closing jingle", help="Jingle that ends the break."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Destination MP3 (default: <source>_cut.mp3 beside the source)."),
    ] = None,
    step_seconds: Annotated[float, typer.Option(help="Sliding-window step.")] = 0.1,
    min_prominence: Annotated[
        float, typer.Option(help="Reject jingle peaks whose prominence z-score is below this.")
    ] = 0.0,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Allow overwriting an existing output."),
    ] = False,
) -> None:
    """Remove the ad break between an opening and closing jingle."""
    try:
        opening_match = find_best_sample_match(
            source_path=source, sample_path=opening, step_seconds=step_seconds
        )
        closing_match = find_best_sample_match(
            source_path=source, sample_path=closing, step_seconds=step_seconds
        )
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    opening_match = _require_jingle(
        ctx=ctx, match=opening_match, min_prominence=min_prominence, label="opening"
    )
    closing_match = _require_jingle(
        ctx=ctx, match=closing_match, min_prominence=min_prominence, label="closing"
    )

    try:
        total_seconds = audio_duration_seconds(source)
        segments = plan_cut(
            opening_start=opening_match.start_seconds,
            opening_end=opening_match.end_seconds,
            closing_start=closing_match.start_seconds,
            closing_end=closing_match.end_seconds,
            total_seconds=total_seconds,
        )
        output_path = _resolve_output(source=source, output=output, overwrite=overwrite)
        cut_segments(source_path=source, destination_path=output_path, keep_segments=segments)
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    emit(
        cut_summary(
            output_path=output_path,
            removed_start=opening_match.start_seconds,
            removed_end=closing_match.end_seconds,
            total_seconds=total_seconds,
        ),
        as_json=_json_flag(ctx),
    )
