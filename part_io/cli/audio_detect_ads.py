"""CLI for detecting ad breaks in an audio file.

Finds occurrences of an opening-tag sample and a closing-tag sample, then
pairs them up where the content between them is between ``--min-gap`` and
``--max-gap`` seconds.  Each tag occurrence is used at most once (greedy,
earliest-open-first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch, find_audio_sample_matches


_DEFAULT_MIN_GAP = 25.0   # seconds — shorter than this is unlikely to be a real ad
_DEFAULT_MAX_GAP = 300.0  # seconds — 5 minutes


def _pair_matches(
    opens: list[AudioMatch],
    closes: list[AudioMatch],
    *,
    min_gap: float,
    max_gap: float,
) -> list[tuple[AudioMatch, AudioMatch]]:
    """Pair each open match with the earliest unused close match in the gap window.

    After pairing, overlapping pairs (open timestamps within ``min_gap / 2``
    seconds of each other) are deduplicated, keeping the highest combined score.
    """
    used_closes: set[int] = set()
    pairs: list[tuple[AudioMatch, AudioMatch]] = []

    for open_match in opens:
        for idx, close_match in enumerate(closes):
            if idx in used_closes:
                continue
            if close_match.start_seconds <= open_match.end_seconds:
                continue
            gap = close_match.start_seconds - open_match.end_seconds
            if gap < min_gap:
                continue
            if gap > max_gap:
                break  # closes are time-ordered; no later one can be closer
            used_closes.add(idx)
            pairs.append((open_match, close_match))
            break

    # Deduplicate pairs whose overall [open.start, close.end] intervals overlap.
    # Among a cluster, keep the pair with the highest combined score.
    kept: list[tuple[AudioMatch, AudioMatch]] = []
    for pair in pairs:
        combined = pair[0].score + pair[1].score
        p_start, p_end = pair[0].start_seconds, pair[1].end_seconds
        merged = False
        for i, existing in enumerate(kept):
            e_start, e_end = existing[0].start_seconds, existing[1].end_seconds
            # Overlapping if intervals share any time
            if p_start < e_end and p_end > e_start:
                if combined > existing[0].score + existing[1].score:
                    kept[i] = pair
                merged = True
                break
        if not merged:
            kept.append(pair)

    return kept


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect ad breaks bounded by opening and closing audio tags."
    )
    parser.add_argument("source", type=Path, help="Audio file to scan")
    parser.add_argument("open_sample", type=Path, help="Opening-tag reference sample")
    parser.add_argument("close_sample", type=Path, help="Closing-tag reference sample")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Match score threshold for both samples (default: 0.8)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=_DEFAULT_MIN_GAP,
        help="Minimum seconds between open-end and close-start (default: 25)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=_DEFAULT_MAX_GAP,
        help="Maximum seconds between open-end and close-start (default: 300)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.5,
        help="Search step in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score cutoff: keep only matches >= mean + N*std (default: 3.0)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    for path in (args.source, args.open_sample, args.close_sample):
        if not path.exists():
            parser.exit(2, f"File not found: {path}\n")

    try:
        opens = find_audio_sample_matches(
            source_path=args.source,
            sample_path=args.open_sample,
            score_threshold=args.threshold,
            step_seconds=args.step,
            z_threshold=args.z_threshold,
        )
        closes = find_audio_sample_matches(
            source_path=args.source,
            sample_path=args.close_sample,
            score_threshold=args.threshold,
            step_seconds=args.step,
            z_threshold=args.z_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
        return

    print(f"Open tags found:  {len(opens)}")
    print(f"Close tags found: {len(closes)}")

    pairs = _pair_matches(opens, closes, min_gap=args.min_gap, max_gap=args.max_gap)

    if not pairs:
        print("No ad breaks detected.")
        sys.exit(1)

    print(f"\nDetected {len(pairs)} ad break(s):\n")
    for i, (open_match, close_match) in enumerate(pairs, 1):
        ad_start = open_match.start_seconds
        ad_end = close_match.end_seconds
        content_gap = close_match.start_seconds - open_match.end_seconds
        print(
            f"  Ad {i}: {ad_start:.1f}s → {ad_end:.1f}s"
            f"  (gap {content_gap:.1f}s,"
            f" open score {open_match.score:.3f},"
            f" close score {close_match.score:.3f})"
        )


if __name__ == "__main__":
    main()
