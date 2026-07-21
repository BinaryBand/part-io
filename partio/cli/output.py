"""Standardized output formatting for CLI commands.

Provides :class:`ExitCode`, :func:`emit` (human-readable or JSON), and
:func:`fail` (error-to-stderr + exit).  Entry points should call these
instead of ``print()`` / ``sys.exit()`` directly.
"""

from __future__ import annotations

import json
import sys
from enum import IntEnum
from pathlib import Path
from typing import NoReturn

import typer


class ExitCode(IntEnum):
    """Standard exit codes for CLI commands."""

    OK = 0
    NO_RESULT = 1
    USER_ERROR = 2
    INTERNAL = 70


def _json_flag(ctx: typer.Context | None) -> bool:
    """Safely extract the ``--json`` flag from a Typer context."""
    if ctx is None or ctx.obj is None:
        return False
    return ctx.obj.get("json", False)


def emit(payload: str | list[str] | dict, *, as_json: bool = False) -> None:
    """Print human-readable output or JSON.

    When *as_json* is ``True``, the payload is serialized with
    :func:`json.dumps` and printed on a single line.  Otherwise each
    string (or list element) is printed on its own line.
    """
    if as_json:
        if isinstance(payload, str):
            payload = {"message": payload}
        print(json.dumps(payload, default=str))
    elif isinstance(payload, list):
        for line in payload:
            print(line)
    else:
        print(payload)


def fail(exc: BaseException) -> NoReturn:
    """Print *exc* to stderr and exit with :attr:`ExitCode.USER_ERROR`."""
    print(str(exc), file=sys.stderr)
    raise SystemExit(ExitCode.USER_ERROR)


# -- existing formatters ---------------------------------------------------


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
