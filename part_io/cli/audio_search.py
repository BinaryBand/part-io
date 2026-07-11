"""CLI for finding a reference audio sample inside a longer MP3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from part_io.adapters.audio.matcher import find_audio_sample_matches
from part_io.cli import handle_cli_error


def main() -> None:
    """Parse args, search for sample matches, and print timestamps."""
    parser = argparse.ArgumentParser(description="Find repeated occurrences of an audio sample.")
    parser.add_argument("source", type=Path, help="Longer audio file to scan")
    parser.add_argument("sample", type=Path, help="Reference sample to search for")
    parser.add_argument("--threshold", type=float, default=0.8, help="Match score threshold")
    args = parser.parse_args()

    try:
        matches = find_audio_sample_matches(
            source_path=args.source,
            sample_path=args.sample,
            score_threshold=args.threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if not matches:
        print("No matches found.")
        sys.exit(1)

    for match in matches:
        print(f"{match.start_seconds:.3f}s -> {match.end_seconds:.3f}s (score={match.score:.4f})")


if __name__ == "__main__":
    main()
