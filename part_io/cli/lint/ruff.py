"""Run ruff format and check using project lint config."""

from __future__ import annotations

import sys

from part_io.cli.lint.entrypoints import run_single_tool_entrypoint
from part_io.cli.lint.execution import run_registered_tool
from part_io.cli.lint.registry import build_tool_cmd


def _build_cmd(cfg: dict) -> list[str]:
    """Build ruff command from centralized registry."""
    return build_tool_cmd("ruff", cfg)


def main() -> None:
    """Run ruff check or format with settings from config/lint.toml."""
    sys.exit(run_single_tool_entrypoint(run_registered_tool, "ruff"))
