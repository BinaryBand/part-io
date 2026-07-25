"""CLI for cutting ad breaks bounded by an opening and a closing jingle.

By default this removes a single break: it locates the best occurrence of each
jingle (prominence-scored, like ``audio locate``) and drops the span from the
start of the opening jingle to the end of the closing one -- both stingers
included. Pass ``--all`` to instead find every occurrence above ``--threshold``
and remove each opening/closing pair, which suits episodes with several
mid-rolls sharing the same stingers. The surviving audio is written to a new MP3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from partio.adapters.audio.clips import audio_duration_seconds, cut_segments
from partio.adapters.audio.matcher import (
    AudioMatch,
    BestMatch,
    find_audio_sample_matches,
    find_best_sample_match,
)
from partio.cli.output import ExitCode, _json_flag, cut_summary, emit, fail, multi_cut_summary
from partio.cli.registry import command
from partio.core.audio_cut import plan_cut, plan_multi_cut


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


def _spans(matches: list[AudioMatch]) -> list[tuple[float, float]]:
    return [(match.start_seconds, match.end_seconds) for match in matches]


def _cut_single_break(
    *,
    ctx: typer.Context,
    source: Path,
    opening: Path,
    closing: Path,
    output: Path | None,
    step_seconds: float,
    min_prominence: float,
    overwrite: bool,
) -> None:
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


def _cut_all_breaks(
    *,
    ctx: typer.Context,
    source: Path,
    opening: Path,
    closing: Path,
    output: Path | None,
    step_seconds: float,
    threshold: float,
    dedupe_overlap: float,
    overwrite: bool,
) -> None:
    try:
        opening_matches = find_audio_sample_matches(
            source_path=source,
            sample_path=opening,
            score_threshold=threshold,
            step_seconds=step_seconds,
            dedupe_overlap=dedupe_overlap,
        )
        closing_matches = find_audio_sample_matches(
            source_path=source,
            sample_path=closing,
            score_threshold=threshold,
            step_seconds=step_seconds,
            dedupe_overlap=dedupe_overlap,
        )
        total_seconds = audio_duration_seconds(source)
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    removed, segments = plan_multi_cut(
        openings=_spans(opening_matches),
        closings=_spans(closing_matches),
        total_seconds=total_seconds,
    )
    if not removed:
        emit("No ad breaks found between the jingles.", as_json=_json_flag(ctx))
        raise SystemExit(ExitCode.NO_RESULT)

    try:
        output_path = _resolve_output(source=source, output=output, overwrite=overwrite)
        cut_segments(source_path=source, destination_path=output_path, keep_segments=segments)
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    emit(
        multi_cut_summary(
            output_path=output_path,
            removed=[(span.start_seconds, span.end_seconds) for span in removed],
            total_seconds=total_seconds,
        ),
        as_json=_json_flag(ctx),
    )


@command("audio", "cut", help="Remove ad breaks bounded by an opening and closing jingle.")
def cut(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Option("--source", prompt="Source audio file", help="Episode to edit."),
    ],
    opening: Annotated[
        Path,
        typer.Option("--opening", prompt="Opening jingle", help="Jingle that starts a break."),
    ],
    closing: Annotated[
        Path,
        typer.Option("--closing", prompt="Closing jingle", help="Jingle that ends a break."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Destination MP3 (default: <source>_cut.mp3 beside the source)."),
    ] = None,
    all_breaks: Annotated[
        bool,
        typer.Option("--all/--single", help="Remove every jingle-bracketed break, not just one."),
    ] = False,
    step_seconds: Annotated[float, typer.Option(help="Sliding-window step.")] = 0.1,
    min_prominence: Annotated[
        float,
        typer.Option(help="Single mode: reject jingle peaks below this prominence z-score."),
    ] = 0.0,
    threshold: Annotated[
        float, typer.Option(help="--all mode: minimum match score to accept a jingle occurrence.")
    ] = 0.8,
    dedupe_overlap: Annotated[
        float, typer.Option(help="--all mode: suppress overlapping matches above this ratio.")
    ] = 0.5,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Allow overwriting an existing output."),
    ] = False,
) -> None:
    """Remove ad breaks bounded by an opening and closing jingle."""
    if all_breaks:
        _cut_all_breaks(
            ctx=ctx,
            source=source,
            opening=opening,
            closing=closing,
            output=output,
            step_seconds=step_seconds,
            threshold=threshold,
            dedupe_overlap=dedupe_overlap,
            overwrite=overwrite,
        )
        return
    _cut_single_break(
        ctx=ctx,
        source=source,
        opening=opening,
        closing=closing,
        output=output,
        step_seconds=step_seconds,
        min_prominence=min_prominence,
        overwrite=overwrite,
    )
