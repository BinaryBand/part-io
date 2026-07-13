"""CLI for interactively bootstrapping a jingle seed clip.

Cold-start discovery: when no reference sample exists yet, this walks the user
through a hinted region of an episode -- auditioning clips via ffplay and
asking yes/no questions -- until the jingle's onset and offset are pinned down,
then writes a canonical seed clip. The seed feeds ``audio_locate`` /
``find_best_sample_match`` to batch the remaining episodes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from part_io.adapters.audio.clips import extract_audio_clip
from part_io.app.audio_bootstrap import locate_jingle_span, locate_jingle_spans
from part_io.cli import handle_cli_error
from part_io.cli.audio_review import build_interactive_auditor
from part_io.core.ports.audio import AuditorFn  # noqa: TC001


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively locate a jingle in an episode and write a seed clip."
    )
    parser.add_argument("source", type=Path, help="Audio file to search for the jingle")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Seed clip destination (default: static/jingles/<source stem>_seed.mp3); "
            "with --max-occurrences > 1 this is a directory for the numbered seed clips"
        ),
    )
    parser.add_argument(
        "--max-occurrences",
        type=int,
        default=1,
        help="Maximum number of jingle occurrences to locate in the region",
    )
    float_flags = [
        ("--region-start", 0.0, "Search region start in seconds"),
        ("--region-end", 120.0, "Search region end in seconds"),
        ("--tile-seconds", 10.0, "Discovery tile width in seconds"),
        ("--probe-seconds", 1.5, "Tuning probe clip length in seconds"),
        ("--resolution", 0.5, "Stop bisecting below this bracket width"),
    ]
    for flag, default, help_text in float_flags:
        parser.add_argument(flag, type=float, default=default, help=help_text)
    return parser


def _tuning_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "region_start": args.region_start,
        "region_end": args.region_end,
        "tile_seconds": args.tile_seconds,
        "probe_seconds": args.probe_seconds,
        "resolution": args.resolution,
    }


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

    print(f"jingle {onset:.3f}s -> {offset:.3f}s written to {output}")


def _bootstrap_single(args: argparse.Namespace, auditor: AuditorFn) -> None:
    try:
        span = locate_jingle_span(auditor=auditor, **_tuning_kwargs(args))
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if span is None:
        print("No jingle found in the search region.")
        sys.exit(1)

    onset, offset = span
    output = args.output or Path("static") / "jingles" / f"{args.source.stem}_seed.mp3"
    _write_seed(args.source, output, onset, offset)


def _bootstrap_multi(args: argparse.Namespace, auditor: AuditorFn) -> None:
    try:
        spans = locate_jingle_spans(
            auditor=auditor, max_occurrences=args.max_occurrences, **_tuning_kwargs(args)
        )
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if not spans:
        print("No jingle found in the search region.")
        sys.exit(1)

    output_dir = args.output or Path("static") / "jingles"
    for index, (onset, offset) in enumerate(spans, start=1):
        output = output_dir / f"{args.source.stem}_seed_{index:02d}.mp3"
        _write_seed(args.source, output, onset, offset)


def main() -> None:
    args = _build_parser().parse_args()

    try:
        if not args.source.exists():
            raise FileNotFoundError(f"Source not found: {args.source}")  # noqa: TRY301
        auditor = build_interactive_auditor(source_path=args.source)
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if args.max_occurrences == 1:
        _bootstrap_single(args, auditor)
    else:
        _bootstrap_multi(args, auditor)


if __name__ == "__main__":
    main()
