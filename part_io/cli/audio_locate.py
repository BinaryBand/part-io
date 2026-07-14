"""CLI for locating the single best occurrence of an audio sample.

This picks the global peak of the similarity curve and reports its prominence
(a z-score against the source's own baseline), which is more robust than a fixed
threshold when scores are compressed -- e.g. finding a jingle in a speech-heavy
episode. Use ``--search-seconds`` to limit the scan to an intro/outro region and
``--min-prominence`` to reject weak peaks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from part_io.adapters.audio.matcher import find_best_sample_match
from part_io.cli import handle_cli_error
from part_io.cli.output import locate_result
from part_io.cli.registry import command


@command("locate-audio", help="Locate the single best occurrence of an audio sample.")
def locate(
    source: Annotated[Path, typer.Argument(help="Longer audio file to scan.")],
    sample: Annotated[Path, typer.Argument(help="Reference sample to search for.")],
    step_seconds: Annotated[float, typer.Option(help="Sliding-window step.")] = 0.1,
    search_seconds: Annotated[
        float | None,
        typer.Option(help="Restrict the scan to the first N seconds of the source."),
    ] = None,
    min_prominence: Annotated[
        float, typer.Option(help="Reject peaks whose prominence z-score is below this value.")
    ] = 0.0,
) -> None:
    """Locate the single best occurrence of an audio sample."""
    try:
        match = find_best_sample_match(
            source_path=source,
            sample_path=sample,
            step_seconds=step_seconds,
            search_seconds=search_seconds,
        )
    except (FileNotFoundError, ValueError) as exc:
        handle_cli_error(exc)

    if match is None or match.prominence < min_prominence:
        print("No confident match found.")
        sys.exit(1)

    print(locate_result(match.start_seconds, match.end_seconds, match.score, match.prominence))


def main() -> None:
    """Run as a standalone script."""
    typer.run(locate)


if __name__ == "__main__":
    main()
