"""CLI for generating manual audio review bundles.

Given a source episode and a reference sample, this tool finds candidate
matches, extracts MP3 clips, and writes a manifest plus labels template under
``downloads/review`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch, find_audio_sample_matches
from part_io.adapters.process.runner import run_resolved


def _format_clip_name(index: int, match: AudioMatch) -> str:
    score = f"{match.score:.4f}".replace(".", "_")
    start = f"{match.start_seconds:.3f}".replace(".", "_")
    return f"{index:03d}_score-{score}_start-{start}.mp3"


def _render_manifest_clip_path(clip_path: Path) -> str:
    try:
        return str(clip_path.relative_to(Path.cwd()))
    except ValueError:
        return str(clip_path)


def _extract_clip(*, source_path: Path, destination_path: Path, match: AudioMatch) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{match.start_seconds:.3f}",
        "-t",
        f"{match.duration_seconds:.3f}",
        "-i",
        str(source_path),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(destination_path),
    ]
    result = run_resolved(command, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg failed to write clip: {destination_path}")


def _write_manifest(*, bundle_dir: Path, source_path: Path, matches: list[AudioMatch]) -> Path:
    manifest_path = bundle_dir / "matches_manifest.csv"
    fieldnames = [
        "index",
        "score",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "clip_path",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, match in enumerate(matches, start=1):
            clip_path = bundle_dir / _format_clip_name(index, match)
            _extract_clip(source_path=source_path, destination_path=clip_path, match=match)
            writer.writerow(
                {
                    "index": index,
                    "score": round(match.score, 4),
                    "start_seconds": round(match.start_seconds, 3),
                    "end_seconds": round(match.end_seconds, 3),
                    "duration_seconds": round(match.duration_seconds, 3),
                    "clip_path": _render_manifest_clip_path(clip_path),
                }
            )

    return manifest_path


def _write_labels_template(
    *, bundle_dir: Path, source_path: Path, sample_path: Path, threshold: float
) -> Path:
    labels_path = bundle_dir / "match_labels.json"
    payload = {
        "source_path": str(source_path),
        "sample_path": str(sample_path),
        "true_positive_indices": [],
        "false_positive_indices": [],
        "threshold": threshold,
        "notes": "Fill true_positive_indices / false_positive_indices after manual listening.",
    }
    labels_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return labels_path


def _resolve_bundle_dir(
    *, output_root: Path, source_path: Path, sample_path: Path, bundle_name: str | None
) -> Path:
    if bundle_name:
        return output_root / bundle_name
    return output_root / source_path.stem / sample_path.stem


def main() -> None:
    """Generate review clips + manifest for manual labeling."""
    parser = argparse.ArgumentParser(
        description="Generate manual review material for audio matches."
    )
    parser.add_argument("source", type=Path, help="Longer audio file to scan")
    parser.add_argument("sample", type=Path, help="Reference sample to search for")
    parser.add_argument("--threshold", type=float, default=0.8, help="Match score threshold")
    parser.add_argument(
        "--step-seconds", type=float, default=0.1, help="Sliding-window step in seconds"
    )
    parser.add_argument(
        "--dedupe-overlap",
        type=float,
        default=0.5,
        help="Suppress overlapping matches above this ratio",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=25,
        help="Maximum number of top-scored matches to export (0 means all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("downloads") / "review",
        help="Root folder where review bundles are written",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default=None,
        help="Optional bundle directory name under output root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing bundle directory",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        sys.exit(2)
    if not args.sample.exists():
        print(f"Sample not found: {args.sample}", file=sys.stderr)
        sys.exit(2)
    if args.max_clips < 0:
        print("--max-clips must be >= 0", file=sys.stderr)
        sys.exit(2)

    bundle_dir = _resolve_bundle_dir(
        output_root=args.output_root,
        source_path=args.source,
        sample_path=args.sample,
        bundle_name=args.bundle_name,
    )
    if bundle_dir.exists() and not args.overwrite:
        print(f"Bundle already exists: {bundle_dir} (use --overwrite)", file=sys.stderr)
        sys.exit(2)

    bundle_dir.mkdir(parents=True, exist_ok=True)

    try:
        matches = find_audio_sample_matches(
            source_path=args.source,
            sample_path=args.sample,
            score_threshold=args.threshold,
            step_seconds=args.step_seconds,
            dedupe_overlap=args.dedupe_overlap,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    selected_matches = matches if args.max_clips == 0 else matches[: args.max_clips]

    try:
        manifest_path = _write_manifest(
            bundle_dir=bundle_dir,
            source_path=args.source,
            matches=selected_matches,
        )
        labels_path = _write_labels_template(
            bundle_dir=bundle_dir,
            source_path=args.source,
            sample_path=args.sample,
            threshold=args.threshold,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    print(f"Bundle: {bundle_dir}")
    print(f"Exported clips: {len(selected_matches)} (from {len(matches)} total matches)")
    print(f"Manifest: {manifest_path}")
    print(f"Labels template: {labels_path}")


if __name__ == "__main__":
    main()
