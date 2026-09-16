"""CLI for cutting ad breaks bounded by an opening and a closing jingle.

By default this removes a single break: it locates the best occurrence of each
jingle (prominence-scored, like ``audio locate``) and drops the span from the
start of the opening jingle to the end of the closing one -- both stingers
included. Pass ``--all`` to instead find every occurrence above ``--threshold``
and remove each opening/closing pair, which suits episodes with several
mid-rolls sharing the same stingers. The surviving audio is written to a new MP3.

The jingle pair can be named once with ``partio rule add`` and reused via
``--rule`` instead of retyping ``--opening``/``--closing`` every time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from partio.adapters.audio.clips import audio_duration_seconds, cut_segments
from partio.adapters.audio.matcher import (
    AudioMatch,
    BestMatch,
    find_audio_sample_matches,
    find_best_sample_match,
)
from partio.cli.library import cut_rules
from partio.cli.output import ExitCode, _json_flag, cut_summary, emit, fail, multi_cut_summary
from partio.cli.prompting import prompt_sample_path
from partio.cli.registry import command
from partio.cli.select import GoBack, Option, select_one
from partio.core.audio_cut import plan_cut, plan_multi_cut
from partio.core.ports import CutRuleEntry

console = Console(stderr=True)
_MANUAL_CHOICE = "manual"


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


def _find_rule(ref: str) -> CutRuleEntry | None:
    """Look up a saved rule by id or by label."""
    for entry in cut_rules():
        if ref in (entry.id, entry.label):
            return entry
    return None


def _prompt_jingles(ctx: typer.Context) -> tuple[Path, Path]:
    """Interactively resolve opening/closing jingles: a saved rule, or by hand."""
    as_json = _json_flag(ctx)
    saved = cut_rules()
    if saved:
        options: list[Option[CutRuleEntry | str]] = [
            Option(
                title=entry.label,
                value=entry,
                help=f"{entry.opening_path} / {entry.closing_path}",
                group="saved rules",
            )
            for entry in saved
        ]
        options.append(Option(title="enter jingles manually", value=_MANUAL_CHOICE))
        chosen = select_one("Pick a cut rule", options, console=console)
        if chosen is None or isinstance(chosen, GoBack):
            emit("Cancelled.", as_json=as_json)
            raise SystemExit(ExitCode.OK)
        if isinstance(chosen, CutRuleEntry):
            return chosen.opening_path, chosen.closing_path

    opening = prompt_sample_path("opening")
    closing = prompt_sample_path("closing")
    if opening is None or closing is None:
        emit("Cancelled.", as_json=as_json)
        raise SystemExit(ExitCode.OK)
    return opening, closing


def _resolve_jingles(
    *, ctx: typer.Context, opening: Path | None, closing: Path | None, rule: str | None
) -> tuple[Path, Path]:
    """Resolve the opening/closing jingles from --rule, explicit paths, or a prompt."""
    if rule is not None:
        if opening is not None or closing is not None:
            fail(ValueError("Pass --rule on its own, or --opening/--closing -- not both."))
        entry = _find_rule(rule)
        if entry is None:
            fail(ValueError(f"No cut rule {rule!r}. List saved rules with `partio rule list`."))
        return entry.opening_path, entry.closing_path
    if opening is not None and closing is not None:
        return opening, closing
    if opening is not None or closing is not None:
        fail(ValueError("Provide both --opening and --closing, or use --rule."))
    return _prompt_jingles(ctx)


@command("audio", "cut", help="Remove ad breaks bounded by an opening and closing jingle.")
def cut(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Option("--source", prompt="Source audio file", help="Episode to edit."),
    ],
    opening: Annotated[
        Path | None,
        typer.Option("--opening", help="Jingle that starts a break (or use --rule)."),
    ] = None,
    closing: Annotated[
        Path | None,
        typer.Option("--closing", help="Jingle that ends a break (or use --rule)."),
    ] = None,
    rule: Annotated[
        str | None,
        typer.Option("--rule", help="Saved cut rule (id or name) providing the jingle pair."),
    ] = None,
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
    resolved_opening, resolved_closing = _resolve_jingles(
        ctx=ctx, opening=opening, closing=closing, rule=rule
    )
    if all_breaks:
        _cut_all_breaks(
            ctx=ctx,
            source=source,
            opening=resolved_opening,
            closing=resolved_closing,
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
        opening=resolved_opening,
        closing=resolved_closing,
        output=output,
        step_seconds=step_seconds,
        min_prominence=min_prominence,
        overwrite=overwrite,
    )
