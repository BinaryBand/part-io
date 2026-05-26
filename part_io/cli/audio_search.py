"""CLI for finding a reference audio sample inside a longer MP3."""

from __future__ import annotations

import argparse
import sys

from part_io.adapters.audio.matcher import find_audio_sample_matches
from part_io.utils.cli import add_audio_sample_arguments


def main() -> None:
    """Parse args, search for sample matches, and print timestamps."""
    parser = argparse.ArgumentParser(description="Find repeated occurrences of an audio sample.")
    add_audio_sample_arguments(parser)
    args = parser.parse_args()

    try:
        matches = find_audio_sample_matches(
            source_path=args.source,
            sample_path=args.sample,
            score_threshold=args.threshold,
            correlation_mode=args.correlation_mode,
            refine_peaks=args.refine_peaks,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")

    if not matches:
        print("No matches found.")
        sys.exit(1)

    for match in matches:
        print(f"{match.start_seconds:.3f}s -> {match.end_seconds:.3f}s (score={match.score:.4f})")


if __name__ == "__main__":
    main()
