"""Run ty type checks for the project."""

from __future__ import annotations

import sys

from part_io.cli.lint.entrypoints import run_single_tool_entrypoint
from part_io.cli.lint.execution import run_registered_tool


def main() -> None:
    """Run ty check and propagate exit code."""
    sys.exit(run_single_tool_entrypoint(run_registered_tool, "ty"))


if __name__ == "__main__":
    main()
