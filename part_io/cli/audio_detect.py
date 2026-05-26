"""Audio sample detection — finds matches and prints coordinates as JSON to stdout.

No files are written. Used as a subprocess by the remote pipeline so detection
jobs can run in parallel without sharing process state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from part_io.adapters.audio.matcher import find_audio_sample_matches
from part_io.services.audio_detection import detect_top_matches, matches_to_cli_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find audio sample matches and print {index,score,start,end} as JSON."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("sample", type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--step-seconds", type=float, default=0.1)
    parser.add_argument(
        "--correlation-mode",
        choices=("gcc-phat", "dot"),
        default="dot",
    )
    parser.add_argument(
        "--refine-peaks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=10,
        help="Return at most N top-scoring matches (default: 10; 0 = unlimited)",
    )
    args = parser.parse_args()

    def _detector(**kwargs):
        return find_audio_sample_matches(
            **kwargs,
            correlation_mode=args.correlation_mode,
            refine_peaks=args.refine_peaks,
        )

    matches = detect_top_matches(
        detector=_detector,
        source_path=args.source,
        sample_path=args.sample,
        score_threshold=args.threshold,
        step_seconds=args.step_seconds,
        max_matches=args.max_matches,
    )

    result = matches_to_cli_rows(matches)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
