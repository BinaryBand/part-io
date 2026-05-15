"""Run copy-paste detection checks using jscpd config."""

from __future__ import annotations

import sys

from part_io.cli.lint.entrypoints import run_single_tool_entrypoint
from part_io.cli.lint.execution import run_registered_tool
from part_io.cli.lint.registry import build_tool_cmd


def _build_cmd(cfg: dict) -> list[str]:
    """Build jscpd command from centralized registry."""
    return build_tool_cmd("cpd", cfg)


def main() -> None:
    """Run jscpd duplicate detection and propagate exit code."""
    sys.exit(run_single_tool_entrypoint(run_registered_tool, "cpd"))


if __name__ == "__main__":
    main()
