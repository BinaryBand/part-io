"""Execution helpers for registered lint tools."""

from __future__ import annotations

from part_io.adapters.errors import LintBoundaryError
from part_io.adapters.lint.runner import run_linter_adapter
from part_io.cli.lint.registry import build_tool_cmd, get_tool_spec


def run_registered_tool(tool_key: str) -> int:
    """Run one registered tool through the unified adapter."""
    try:
        spec = get_tool_spec(tool_key)
        return run_linter_adapter(
            spec.executable,
            lambda cfg: build_tool_cmd(tool_key, cfg),
            config_section=spec.config_section,
        )
    except LintBoundaryError:
        return 2


__all__ = ["run_registered_tool"]
