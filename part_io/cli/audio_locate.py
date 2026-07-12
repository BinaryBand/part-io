"""CLI for locating the single best occurrence of an audio sample.

This picks the global peak of the similarity curve and reports its prominence
(a z-score against the source's own baseline), which is more robust than a fixed
threshold when scores are compressed -- e.g. finding a jingle in a speech-heavy
episode. Use ``--search-seconds`` to limit the scan to an intro/outro region and
``--min-prominence`` to reject weak peaks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from part_io.adapters.audio.matcher import find_best_sample_match
from part_io.cli import handle_cli_error


def main() -> None:
    """Parse args, locate the best sample occurrence, and print it."""
    parser = argparse.ArgumentParser(
        description="Locate the single best occurrence of an audio sample."
    )
    parser.add_argument("source", type=Path, help="Longer audio file to scan")
    parser.add_argument("sample", type=Path, help="Reference sample to search for")
    parser.add_argument("--step-seconds", type=float, default=0.1, help="Sliding-window step")
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=None,
        help="Restrict the scan to the first N seconds of the source",
    )
    parser.add_argument(
        "--min-prominence",
        type=float,
        default=0.0,
        help="Reject peaks whose prominence z-score is below this value",
    )
    args = parser.parse_args()

    try:
        match = find_best_sample_match(
            source_path=args.source,
            sample_path=args.sample,
            step_seconds=args.step_seconds,
            search_seconds=args.search_seconds,
        )
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if match is None or match.prominence < args.min_prominence:
        print("No confident match found.")
        sys.exit(1)

    print(
        f"{match.start_seconds:.3f}s -> {match.end_seconds:.3f}s "
        f"(score={match.score:.4f}, prominence={match.prominence:.2f})"
    )


if __name__ == "__main__":
    main()
