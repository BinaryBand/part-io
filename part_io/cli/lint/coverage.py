"""Run pytest coverage check using project lint config."""

from __future__ import annotations

import sys

from part_io.cli.lint.entrypoints import run_single_tool_entrypoint
from part_io.cli.lint.execution import run_registered_tool
from part_io.cli.lint.registry import build_tool_cmd
from part_io.utils.coverage import cleanup_coverage_temp_files


def _build_cmd(cfg: dict) -> list[str]:
    """Build pytest command with coverage floor from centralized registry."""
    return build_tool_cmd("coverage", cfg)


def main() -> None:
    """Run pytest --cov with the floor defined in config/lint.toml."""
    # Start clean to avoid stale temp files piling up between runs.
    cleanup_coverage_temp_files()

    exit_code = run_single_tool_entrypoint(run_registered_tool, "coverage")

    # Clean up any temp files created by this run.
    cleanup_coverage_temp_files()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
