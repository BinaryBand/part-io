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
from part_io.cli import handle_cli_error
from part_io.cli.audio_review import build_interactive_auditor
from part_io.services.audio_bootstrap import locate_jingle_span


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively locate a jingle in an episode and write a seed clip."
    )
    parser.add_argument("source", type=Path, help="Audio file to search for the jingle")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Seed clip destination (default: static/jingles/<source stem>_seed.mp3)",
    )
    parser.add_argument(
        "--region-start", type=float, default=0.0, help="Search region start in seconds"
    )
    parser.add_argument(
        "--region-end", type=float, default=120.0, help="Search region end in seconds"
    )
    parser.add_argument(
        "--tile-seconds", type=float, default=10.0, help="Discovery tile width in seconds"
    )
    parser.add_argument(
        "--probe-seconds", type=float, default=1.5, help="Tuning probe clip length in seconds"
    )
    parser.add_argument(
        "--resolution", type=float, default=0.5, help="Stop bisecting below this bracket width"
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _validate_args(args)
        auditor = build_interactive_auditor(source_path=args.source)
        span = locate_jingle_span(
            auditor=auditor,
            region_start=args.region_start,
            region_end=args.region_end,
            tile_seconds=args.tile_seconds,
            probe_seconds=args.probe_seconds,
            resolution=args.resolution,
        )
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if span is None:
        print("No jingle found in the search region.")
        sys.exit(1)

    onset, offset = span
    output = args.output or Path("static") / "jingles" / f"{args.source.stem}_seed.mp3"

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        extract_audio_clip(
            source_path=args.source,
            destination_path=output,
            start_seconds=onset,
            duration_seconds=offset - onset,
        )
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    print(f"jingle {onset:.3f}s -> {offset:.3f}s written to {output}")


if __name__ == "__main__":
    main()
