"""Standardized output formatting for CLI commands.

Pure formatting functions — no ``print()`` or ``sys.exit()`` calls.
Entry points are responsible for I/O and process exit.
"""

from __future__ import annotations

from pathlib import Path


def no_match(label: str) -> str:
    """Return a standard "no results" message.

    >>> no_match("matches")
    'No matches found.'
    """
    return f"No {label} found."


def match_line(start_seconds: float, end_seconds: float, score: float) -> str:
    """Format a single audio-match result line."""
    return f"{start_seconds:.3f}s -> {end_seconds:.3f}s (score={score:.4f})"


def locate_result(
    start_seconds: float,
    end_seconds: float,
    score: float,
    prominence: float,
) -> str:
    """Format the ``locate-audio`` peak result line."""
    return (
        f"{start_seconds:.3f}s -> {end_seconds:.3f}s "
        f"(score={score:.4f}, prominence={prominence:.2f})"
    )


def bundle_summary(
    *,
    bundle_dir: Path,
    selected_count: int,
    total_matches: int,
    manifest_path: Path,
    labels_path: Path,
) -> list[str]:
    """Format the multi-line review-bundle summary."""
    return [
        f"Bundle: {bundle_dir}",
        f"Exported clips: {selected_count} (from {total_matches} total matches)",
        f"Manifest: {manifest_path}",
        f"Labels: {labels_path}",
    ]


def seed_written(output: Path, onset: float, offset: float) -> str:
    """Format the seed-clip write confirmation."""
    return f"jingle {onset:.3f}s -> {offset:.3f}s written to {output}"
