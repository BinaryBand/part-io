"""CLI for interactively bootstrapping a jingle seed clip.

Cold-start discovery: when no reference sample exists yet, this walks the user
through a hinted region of an episode -- auditioning clips via ffplay and
asking yes/no questions -- until the jingle's onset and offset are pinned down,
then writes a canonical seed clip. The seed feeds ``audio_locate`` /
``find_best_sample_match`` to batch the remaining episodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from partio.adapters.audio.clips import extract_audio_clip
from partio.app.audio_bootstrap import locate_jingle_span, locate_jingle_spans
from partio.cli.commands.audio._auditor import build_interactive_auditor
from partio.cli.output import ExitCode, _json_flag, emit, fail, seed_written

if TYPE_CHECKING:
    from partio.core.ports.audio import AuditorFn
from partio.cli.registry import command


def _write_seed(source: Path, output: Path, onset: float, offset: float) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        extract_audio_clip(
            source_path=source,
            destination_path=output,
            start_seconds=onset,
            duration_seconds=offset - onset,
        )
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)


def _tuning_kwargs(
    *,
    region_start: float,
    region_end: float,
    tile_seconds: float,
    probe_seconds: float,
    resolution: float,
) -> dict[str, float]:
    return {
        "region_start": region_start,
        "region_end": region_end,
        "tile_seconds": tile_seconds,
        "probe_seconds": probe_seconds,
        "resolution": resolution,
    }


def _bootstrap_single(
    *,
    ctx: typer.Context,
    source: Path,
    output: Path | None,
    auditor: AuditorFn,
    **tuning: float,
) -> None:
    try:
        span = locate_jingle_span(auditor=auditor, **tuning)
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    if span is None:
        emit("No jingle found in the search region.", as_json=_json_flag(ctx))
        raise SystemExit(ExitCode.NO_RESULT)

    onset, offset = span
    dest = output or Path("static") / "jingles" / f"{source.stem}_seed.mp3"
    _write_seed(source, dest, onset, offset)
    emit(seed_written(dest, onset, offset), as_json=_json_flag(ctx))


def _bootstrap_multi(
    *,
    ctx: typer.Context,
    source: Path,
    output: Path | None,
    max_occurrences: int,
    auditor: AuditorFn,
    **tuning: float,
) -> None:
    try:
        spans = locate_jingle_spans(auditor=auditor, max_occurrences=max_occurrences, **tuning)
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    if not spans:
        emit("No jingle found in the search region.", as_json=_json_flag(ctx))
        raise SystemExit(ExitCode.NO_RESULT)

    output_dir = output or Path("static") / "jingles"
    for index, (onset, offset) in enumerate(spans, start=1):
        dest = output_dir / f"{source.stem}_seed_{index:02d}.mp3"
        _write_seed(source, dest, onset, offset)
        emit(seed_written(dest, onset, offset), as_json=_json_flag(ctx))


@command("audio", "bootstrap", help="Interactively locate a jingle and write a seed clip.")
def bootstrap(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Option(
            "--source", prompt="Source audio file", help="Audio file to search for the jingle."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Seed clip destination (default: static/jingles/<source stem>_seed.mp3); "
                "with --max-occurrences > 1 this is a directory for the numbered seed clips."
            )
        ),
    ] = None,
    max_occurrences: Annotated[
        int, typer.Option(help="Maximum number of jingle occurrences to locate in the region.")
    ] = 1,
    region_start: Annotated[float, typer.Option(help="Search region start in seconds.")] = 0.0,
    region_end: Annotated[float, typer.Option(help="Search region end in seconds.")] = 120.0,
    tile_seconds: Annotated[float, typer.Option(help="Discovery tile width in seconds.")] = 10.0,
    probe_seconds: Annotated[
        float, typer.Option(help="Tuning probe clip length in seconds.")
    ] = 1.5,
    resolution: Annotated[
        float, typer.Option(help="Stop bisecting below this bracket width.")
    ] = 0.5,
) -> None:
    """Interactively locate a jingle in an episode and write a seed clip."""
    try:
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")  # noqa: TRY301
        auditor = build_interactive_auditor(source_path=source)
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    tuning = _tuning_kwargs(
        region_start=region_start,
        region_end=region_end,
        tile_seconds=tile_seconds,
        probe_seconds=probe_seconds,
        resolution=resolution,
    )

    if max_occurrences == 1:
        _bootstrap_single(ctx=ctx, source=source, output=output, auditor=auditor, **tuning)
    else:
        _bootstrap_multi(
            ctx=ctx,
            source=source,
            output=output,
            max_occurrences=max_occurrences,
            auditor=auditor,
            **tuning,
        )
