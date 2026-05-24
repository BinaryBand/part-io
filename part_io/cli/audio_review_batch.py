"""Batch CLI for generating audio review bundles across media files."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from part_io.adapters.process.runner import run_resolved
from part_io.utils.cli import add_alignment_refinement_arguments, add_review_export_arguments


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
    add_alignment_refinement_arguments(parser)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(2, os.cpu_count() or 1),
        help="Number of parallel subprocess workers (default: min(2, cpu_count))",
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
    onset_anchor: bool = False,
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
    if onset_anchor:
        command.append("--onset-anchor")

    result = run_resolved(command, capture_output=True)
    if result.returncode != 0 and result.stderr:
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.flush()
    return int(result.returncode)


def _validate_batch_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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


def _build_batch_jobs(
    *,
    media_files: list[Path],
    close_sample_path: Path,
    open_sample_path: Path,
    bundle_pattern: str,
) -> list[tuple[Path, Path, str]]:
    jobs: list[tuple[Path, Path, str]] = []
    for source_file in media_files:
        base_name = source_file.stem
        jobs.append(
            (
                source_file,
                close_sample_path,
                bundle_pattern.format(base=base_name, kind="close"),
            )
        )
        jobs.append(
            (
                source_file,
                open_sample_path,
                bundle_pattern.format(base=base_name, kind="open"),
            )
        )
    return jobs


def _emit_progress(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _run_batch_jobs(
    *,
    jobs: list[tuple[Path, Path, str]],
    workers: int,
    threshold: float,
    step_seconds: float,
    max_clips: int,
    output_root: Path,
    overwrite: bool,
    onset_anchor: bool,
) -> None:
    total = len(jobs)
    _emit_progress(f"Processing {total} bundles across {workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                source_file=sf,
                sample_path=sp,
                bundle_name=bn,
                threshold=threshold,
                step_seconds=step_seconds,
                max_clips=max_clips,
                output_root=output_root,
                overwrite=overwrite,
                onset_anchor=onset_anchor,
            ): bn
            for sf, sp, bn in jobs
        }
        done = 0
        for future in as_completed(futures):
            bundle_name = futures[future]
            exit_code = future.result()
            done += 1
            if exit_code != 0:
                _emit_progress(f"[{done}/{total}] FAILED {bundle_name}")
                for pending in futures:
                    pending.cancel()
                raise SystemExit(exit_code)
            _emit_progress(f"[{done}/{total}] done  {bundle_name}")


def main() -> None:
    """Run batch audio review generation across all source files in media dir."""
    parser = _build_parser()
    args = parser.parse_args()

    _validate_batch_args(parser, args)

    close_sample_path = args.snippets_dir / args.close_sample
    open_sample_path = args.snippets_dir / args.open_sample

    media_files = _iter_media_files(args.media_dir)
    if not media_files:
        parser.exit(2, f"No .mp3 files found in media directory: {args.media_dir}\n")

    jobs = _build_batch_jobs(
        media_files=media_files,
        close_sample_path=close_sample_path,
        open_sample_path=open_sample_path,
        bundle_pattern=args.bundle_pattern,
    )

    _run_batch_jobs(
        jobs=jobs,
        workers=args.workers,
        threshold=args.threshold,
        step_seconds=args.step_seconds,
        max_clips=args.max_clips,
        output_root=args.output_root,
        overwrite=args.overwrite,
        onset_anchor=args.onset_anchor,
    )


if __name__ == "__main__":
    main()
