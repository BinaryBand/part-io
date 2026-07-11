"""Run vulture dead-code checks using project lint config."""

from __future__ import annotations

import sys

from part_io.cli.lint.execution import run_registered_tool


def main() -> None:
    """Run vulture with settings from config/lint.toml."""
    sys.exit(run_registered_tool("vulture"))


if __name__ == "__main__":
    main()
