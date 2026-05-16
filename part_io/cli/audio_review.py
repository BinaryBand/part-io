"""CLI for generating manual audio review bundles.

Given a source episode and a reference sample, this tool finds candidate
matches, extracts MP3 clips, and writes a manifest plus labels template under
``downloads/review`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from part_io.adapters.audio.matcher import (
    AudioMatch,
    _suppress_overlapping,
    anchor_to_onset,
    cross_correlate_align,
    find_audio_sample_matches,
)
from part_io.adapters.process.runner import run_resolved
from part_io.utils.cli import (
    add_alignment_refinement_arguments,
    add_audio_sample_arguments,
    add_review_export_arguments,
)


def _refine_match(
    *,
    coarse_match: AudioMatch,
    source_path: Path,
    sample_path: Path,
    threshold: float,
    refine_step_seconds: float = 0.01,
    refine_window_seconds: float = 5.0,
) -> AudioMatch:
    """Refine a coarse match via finer-grained local search.

    Searches around the coarse winner within ±refine_window_seconds at finer
    resolution (refine_step_seconds). Returns best refined match from finer scan.
    """
    window_start = max(0.0, coarse_match.start_seconds - refine_window_seconds)
    window_end = coarse_match.start_seconds + coarse_match.duration_seconds + refine_window_seconds

    candidates = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=min(threshold, coarse_match.score * 0.9),
        step_seconds=refine_step_seconds,
        dedupe_overlap=0.5,
        search_start_seconds=window_start,
        search_end_seconds=window_end,
    )

    if not candidates:
        return coarse_match

    return max(candidates, key=lambda m: m.score)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate manual review material for audio matches."
    )
    add_audio_sample_arguments(parser)
    parser.add_argument(
        "--step-seconds", type=float, default=0.1, help="Sliding-window step in seconds"
    )
    parser.add_argument(
        "--dedupe-overlap",
        type=float,
        default=0.5,
        help="Suppress overlapping matches above this ratio",
    )
    add_review_export_arguments(parser)
    parser.add_argument(
        "--bundle-name",
        type=str,
        default=None,
        help="Optional bundle directory name under output root",
    )
    add_alignment_refinement_arguments(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")
    if not args.sample.exists():
        raise FileNotFoundError(f"Sample not found: {args.sample}")
    if args.max_clips < 0:
        raise ValueError("--max-clips must be >= 0")


def _find_and_refine_matches(
    *,
    source_path: Path,
    sample_path: Path,
    threshold: float,
    step_seconds: float,
    dedupe_overlap: float,
    refine: bool = False,
    onset_anchor: bool = False,
    precise: bool = False,
) -> list[AudioMatch]:
    """Find matches and optionally refine them."""
    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=threshold,
        step_seconds=step_seconds,
        dedupe_overlap=dedupe_overlap,
    )
    if refine:
        matches = [
            _refine_match(
                coarse_match=match,
                source_path=source_path,
                sample_path=sample_path,
                threshold=threshold,
            )
            for match in matches
        ]
        matches = _suppress_overlapping(matches, min_overlap=dedupe_overlap)
    if onset_anchor:
        matches = [anchor_to_onset(match=match, source_path=source_path) for match in matches]
        matches = _suppress_overlapping(matches, min_overlap=dedupe_overlap)
    if precise:
        matches = [
            cross_correlate_align(
                match=match,
                source_path=source_path,
                sample_path=sample_path,
            )
            for match in matches
        ]
        matches = _suppress_overlapping(matches, min_overlap=dedupe_overlap)
    return matches


def _generate_bundle(
    *,
    source_path: Path,
    sample_path: Path,
    threshold: float,
    step_seconds: float,
    dedupe_overlap: float,
    max_clips: int,
    output_root: Path,
    bundle_name: str | None,
    overwrite: bool,
    refine: bool = False,
    onset_anchor: bool = False,
    precise: bool = False,
) -> tuple[Path, Path, Path, int, int]:
    _validate_args(
        argparse.Namespace(
            source=source_path,
            sample=sample_path,
            max_clips=max_clips,
        )
    )

    bundle_dir = _resolve_bundle_dir(
        output_root=output_root,
        source_path=source_path,
        sample_path=sample_path,
        bundle_name=bundle_name,
    )
    if bundle_dir.exists() and not overwrite:
        raise ValueError(f"Bundle already exists: {bundle_dir} (use --overwrite)")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    matches = _find_and_refine_matches(
        source_path=source_path,
        sample_path=sample_path,
        threshold=threshold,
        step_seconds=step_seconds,
        dedupe_overlap=dedupe_overlap,
        refine=refine,
        onset_anchor=onset_anchor,
        precise=precise,
    )

    ranked_matches = sorted(matches, key=lambda match: match.score, reverse=True)
    selected_matches = ranked_matches if max_clips == 0 else ranked_matches[:max_clips]
    manifest_path = _write_manifest(
        bundle_dir=bundle_dir,
        source_path=source_path,
        matches=selected_matches,
    )
    labels_path = _write_labels_template(
        bundle_dir=bundle_dir,
        source_path=source_path,
        sample_path=sample_path,
        threshold=threshold,
    )
    return bundle_dir, manifest_path, labels_path, len(matches), len(selected_matches)


def main() -> None:
    """Generate review clips + manifest for manual labeling."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _validate_args(args)
        bundle_dir, manifest_path, labels_path, total_matches, selected_count = _generate_bundle(
            source_path=args.source,
            sample_path=args.sample,
            threshold=args.threshold,
            step_seconds=args.step_seconds,
            dedupe_overlap=args.dedupe_overlap,
            max_clips=args.max_clips,
            output_root=args.output_root,
            bundle_name=args.bundle_name,
            overwrite=args.overwrite,
            refine=args.refine,
            onset_anchor=args.onset_anchor,
            precise=args.precise,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")

    print(f"Bundle: {bundle_dir}")
    print(f"Exported clips: {selected_count} (from {total_matches} total matches)")
    print(f"Manifest: {manifest_path}")
    print(f"Labels template: {labels_path}")


if __name__ == "__main__":
    main()
