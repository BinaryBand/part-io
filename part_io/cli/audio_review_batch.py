"""Batch CLI for generating audio review bundles across media files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from part_io.adapters.process.runner import run_resolved
from part_io.utils.cli import add_review_export_arguments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run audio review generation for every media file using close/open snippets."
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=Path("downloads") / "media",
        help="Directory containing source media .mp3 files",
    )
    parser.add_argument(
        "--snippets-dir",
        type=Path,
        default=Path("downloads") / "snippets",
        help="Directory containing reference snippet files",
    )
    parser.add_argument(
        "--close-sample", type=str, default="close.mp3", help="Close sample filename"
    )
    parser.add_argument("--open-sample", type=str, default="open.mp3", help="Open sample filename")
    parser.add_argument("--threshold", type=float, default=0.8, help="Match score threshold")
    parser.add_argument(
        "--step-seconds", type=float, default=0.1, help="Sliding-window step in seconds"
    )
    add_review_export_arguments(parser)
    parser.add_argument(
        "--bundle-pattern",
        type=str,
        default="{base}/{kind}_high_points",
        help="Bundle naming pattern, e.g. '{base}/{kind}_high_points'",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine coarse matches via finer-grained local search",
    )
    return parser


def _iter_media_files(media_dir: Path) -> list[Path]:
    return sorted(path for path in media_dir.glob("*.mp3") if path.is_file())


def _run_one(
    *,
    source_file: Path,
    sample_path: Path,
    threshold: float,
    step_seconds: float,
    max_clips: int,
    output_root: Path,
    bundle_name: str,
    overwrite: bool,
    refine: bool = False,
) -> int:
    command = [
        sys.executable,
        "-m",
        "part_io.cli.audio_review",
        str(source_file),
        str(sample_path),
        "--threshold",
        str(threshold),
        "--step-seconds",
        str(step_seconds),
        "--max-clips",
        str(max_clips),
        "--output-root",
        str(output_root),
        "--bundle-name",
        bundle_name,
    ]
    if overwrite:
        command.append("--overwrite")
    if refine:
        command.append("--refine")

    result = run_resolved(command)
    return int(result.returncode)


def main() -> None:
    """Run batch audio review generation across all source files in media dir."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.max_clips < 0:
        parser.exit(2, "--max-clips must be >= 0\n")

    if not args.media_dir.exists():
        parser.exit(2, f"Media directory not found: {args.media_dir}\n")

    close_sample_path = args.snippets_dir / args.close_sample
    open_sample_path = args.snippets_dir / args.open_sample

    if not close_sample_path.exists():
        parser.exit(2, f"Close sample not found: {close_sample_path}\n")
    if not open_sample_path.exists():
        parser.exit(2, f"Open sample not found: {open_sample_path}\n")

    media_files = _iter_media_files(args.media_dir)
    if not media_files:
        parser.exit(2, f"No .mp3 files found in media directory: {args.media_dir}\n")

    for source_file in media_files:
        base_name = source_file.stem

        close_bundle = args.bundle_pattern.format(base=base_name, kind="close")
        close_exit = _run_one(
            source_file=source_file,
            sample_path=close_sample_path,
            threshold=args.threshold,
            step_seconds=args.step_seconds,
            max_clips=args.max_clips,
            output_root=args.output_root,
            bundle_name=close_bundle,
            overwrite=args.overwrite,
            refine=args.refine,
        )
        if close_exit != 0:
            raise SystemExit(close_exit)

        open_bundle = args.bundle_pattern.format(base=base_name, kind="open")
        open_exit = _run_one(
            source_file=source_file,
            sample_path=open_sample_path,
            threshold=args.threshold,
            step_seconds=args.step_seconds,
            max_clips=args.max_clips,
            output_root=args.output_root,
            bundle_name=open_bundle,
            overwrite=args.overwrite,
            refine=args.refine,
        )
        if open_exit != 0:
            raise SystemExit(open_exit)


if __name__ == "__main__":
    main()
