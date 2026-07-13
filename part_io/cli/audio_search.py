"""CLI for finding a reference audio sample inside a longer MP3."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from part_io.adapters.audio.matcher import find_audio_sample_matches
from part_io.cli import handle_cli_error


def search(
    source: Annotated[Path, typer.Argument(help="Longer audio file to scan.")],
    sample: Annotated[Path, typer.Argument(help="Reference sample to search for.")],
    threshold: Annotated[float, typer.Option(help="Match score threshold.")] = 0.8,
) -> None:
    """Find repeated occurrences of an audio sample."""
    try:
        matches = find_audio_sample_matches(
            source_path=source,
            sample_path=sample,
            score_threshold=threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if not matches:
        print("No matches found.")
        sys.exit(1)

    for match in matches:
        print(f"{match.start_seconds:.3f}s -> {match.end_seconds:.3f}s (score={match.score:.4f})")


def main() -> None:
    """Run as a standalone script."""
    typer.run(search)
