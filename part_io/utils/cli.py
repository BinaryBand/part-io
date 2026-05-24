"""Shared CLI utilities for audio tools."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_audio_sample_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common audio sample search arguments to a parser.

    Adds:
    - source (positional): Longer audio file to scan
    - sample (positional): Reference sample to search for
    - --threshold (optional): Match score threshold (default 0.8)
    """
    parser.add_argument("source", type=Path, help="Longer audio file to scan")
    parser.add_argument("sample", type=Path, help="Reference sample to search for")
    parser.add_argument("--threshold", type=float, default=0.8, help="Match score threshold")


def add_review_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common review export arguments used by review CLIs."""
    parser.add_argument(
        "--max-clips",
        type=int,
        default=25,
        help="Maximum number of top-scored matches to export (0 means all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("downloads") / "review",
        help="Root folder where review bundles are written",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing bundle directory",
    )


def add_alignment_refinement_arguments(parser: argparse.ArgumentParser) -> None:
    """Add alignment post-processing flags shared by audio review CLIs."""
    parser.add_argument(
        "--onset-anchor",
        action="store_true",
        help="Shift match start to first significant energy onset",
    )
