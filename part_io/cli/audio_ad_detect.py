"""CLI: pair detected ad opens with closes and write ad_segments.json.

Reads the open and close match manifests produced by audio-review-batch for a
single episode, pairs each open with the nearest following eligible close, and
writes a JSON file that audio-ad-remove can consume to cut the ads.

When --use-labels is given only manifest rows whose index appears in
true_positive_indices (from match_labels.json) are considered.  This is the
recommended mode once you have manually reviewed the clips.  Without the flag
all manifest rows are used with a warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from part_io.adapters.audio.ad_segments import (
    AdSegment,
    load_manifest_matches,
    pair_ad_segments,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pair ad-open and ad-close detections and write ad_segments.json."
    )
    parser.add_argument(
        "--episode",
        required=True,
        help="Episode stem (e.g. ep_ce79a6d1) or path to episode dir under review-root",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("downloads") / "review",
        help="Root directory of review bundles",
    )
    parser.add_argument(
        "--open-bundle",
        default="open_high_points",
        help="Bundle directory name for open matches",
    )
    parser.add_argument(
        "--close-bundle",
        default="close_high_points",
        help="Bundle directory name for close matches",
    )
    parser.add_argument(
        "--use-labels",
        action="store_true",
        help="Filter matches to true_positive_indices from match_labels.json",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=10.0,
        help="Minimum seconds between open end and close start (default: 10)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=600.0,
        help="Maximum seconds between open end and close start (default: 600)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for ad_segments.json (default: {review_root}/{episode}/ad_segments.json)",
    )
    return parser


def _resolve_episode_dir(review_root: Path, episode: str) -> Path:
    candidate = Path(episode)
    if candidate.is_dir():
        return candidate
    return review_root / episode


def _load_bundle(
    episode_dir: Path,
    bundle_name: str,
    *,
    use_labels: bool,
) -> tuple[list, list[str]]:
    """Load matches from a bundle, return (matches, warnings)."""
    warnings: list[str] = []
    manifest = episode_dir / bundle_name / "matches_manifest.csv"
    labels = episode_dir / bundle_name / "match_labels.json"

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    labels_path: Path | None = None
    if use_labels:
        if not labels.exists():
            warnings.append(f"--use-labels set but no labels file at {labels}; using all rows")
        else:
            data = json.loads(labels.read_text(encoding="utf-8"))
            if not data.get("true_positive_indices"):
                warnings.append(
                    f"Labels file {labels} has empty true_positive_indices; using all rows"
                )
            labels_path = labels

    matches = load_manifest_matches(manifest, labels_path)
    return matches, warnings


def _segment_to_dict(seg: AdSegment) -> dict:
    return {
        "open_start": seg.open_start,
        "open_end": seg.open_end,
        "close_start": seg.close_start,
        "close_end": seg.close_end,
        "open_score": seg.open_score,
        "close_score": seg.close_score,
        "gap_seconds": seg.gap_seconds,
    }


def _match_to_dict(m) -> dict:
    return {
        "start_seconds": m.start_seconds,
        "end_seconds": m.end_seconds,
        "score": m.score,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    episode_dir = _resolve_episode_dir(args.review_root, args.episode)
    if not episode_dir.exists():
        parser.exit(2, f"Episode directory not found: {episode_dir}\n")

    all_warnings: list[str] = []

    try:
        opens, open_warnings = _load_bundle(episode_dir, args.open_bundle, use_labels=args.use_labels)
        closes, close_warnings = _load_bundle(episode_dir, args.close_bundle, use_labels=args.use_labels)
    except FileNotFoundError as exc:
        parser.exit(2, f"{exc}\n")
        return

    all_warnings.extend(open_warnings)
    all_warnings.extend(close_warnings)

    if not opens:
        all_warnings.append("No open matches found — no segments will be detected")
    if not closes:
        all_warnings.append("No close matches found — no segments will be detected")

    segments, unpaired_opens, unpaired_closes = pair_ad_segments(
        opens,
        closes,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
    )

    for m in unpaired_opens:
        all_warnings.append(
            f"Unpaired open at {m.start_seconds}s (score={m.score}) — no close within"
            f" [{args.min_gap}, {args.max_gap}]s; skipping"
        )
    for m in unpaired_closes:
        all_warnings.append(
            f"Unpaired close at {m.start_seconds}s (score={m.score}) — no preceding open; skipping"
        )

    for w in all_warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    output_path = args.output or (episode_dir / "ad_segments.json")
    payload = {
        "episode": args.episode,
        "segments": [_segment_to_dict(s) for s in segments],
        "unpaired_opens": [_match_to_dict(m) for m in unpaired_opens],
        "unpaired_closes": [_match_to_dict(m) for m in unpaired_closes],
        "warnings": all_warnings,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Detected {len(segments)} ad segment(s)")
    for i, seg in enumerate(segments, 1):
        print(
            f"  {i}. {seg.open_start:.3f}s → {seg.close_end:.3f}s"
            f"  (gap {seg.gap_seconds:.1f}s,"
            f" open={seg.open_score:.4f}, close={seg.close_score:.4f})"
        )
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
