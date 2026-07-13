"""CLI for interactively bootstrapping a jingle seed clip.

Cold-start discovery: when no reference sample exists yet, this walks the user
through a hinted region of an episode -- auditioning clips via ffplay and
asking yes/no questions -- until the jingle's onset and offset are pinned down,
then writes a canonical seed clip. The seed feeds ``audio_locate`` /
``find_best_sample_match`` to batch the remaining episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from part_io.adapters.audio.clips import extract_audio_clip
from part_io.app.audio_bootstrap import locate_jingle_span, locate_jingle_spans
from part_io.cli import handle_cli_error
from part_io.cli.audio_review import build_interactive_auditor
from part_io.cli.output import seed_written
from part_io.cli.registry import command
from part_io.core.ports.audio import AuditorFn  # noqa: TC001


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
        handle_cli_error(exc)

    print(seed_written(output, onset, offset))


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
    source: Path,
    output: Path | None,
    auditor: AuditorFn,
    **tuning: float,
) -> None:
    try:
        span = locate_jingle_span(auditor=auditor, **tuning)
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if span is None:
        print("No jingle found in the search region.")
        sys.exit(1)

    onset, offset = span
    dest = output or Path("static") / "jingles" / f"{source.stem}_seed.mp3"
    _write_seed(source, dest, onset, offset)


def _bootstrap_multi(
    *,
    source: Path,
    output: Path | None,
    max_occurrences: int,
    auditor: AuditorFn,
    **tuning: float,
) -> None:
    try:
        spans = locate_jingle_spans(auditor=auditor, max_occurrences=max_occurrences, **tuning)
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if not spans:
        print("No jingle found in the search region.")
        sys.exit(1)

    output_dir = output or Path("static") / "jingles"
    for index, (onset, offset) in enumerate(spans, start=1):
        dest = output_dir / f"{source.stem}_seed_{index:02d}.mp3"
        _write_seed(source, dest, onset, offset)


@command("bootstrap-audio", help="Interactively locate a jingle and write a seed clip.")
def bootstrap(
    source: Annotated[Path, typer.Argument(help="Audio file to search for the jingle.")],
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
        handle_cli_error(exc)

    tuning = _tuning_kwargs(
        region_start=region_start,
        region_end=region_end,
        tile_seconds=tile_seconds,
        probe_seconds=probe_seconds,
        resolution=resolution,
    )

    if max_occurrences == 1:
        _bootstrap_single(source=source, output=output, auditor=auditor, **tuning)
    else:
        _bootstrap_multi(
            source=source,
            output=output,
            max_occurrences=max_occurrences,
            auditor=auditor,
            **tuning,
        )


def main() -> None:
    """Run as a standalone script."""
    typer.run(bootstrap)
