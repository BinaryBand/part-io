"""Unified lint runner adapter for tool commands.

Callable ports implemented here are defined in part_io.models.ports.
"""

from __future__ import annotations

from collections.abc import Callable

from part_io.adapters.config.lint_config_loader import (
    DEFAULT_LINT_CONFIG_PATH,
    load_lint_config,
)
from part_io.adapters.errors import LintProcessError
from part_io.adapters.process.runner import run_resolved

_CONFIG = DEFAULT_LINT_CONFIG_PATH


def run_linter_command(tool_name: str, cmd: list[str]) -> int:
    """Execute a lint command and translate process errors to boundary errors."""
    try:
        result = run_resolved(cmd)
    except (OSError, ValueError) as exc:
        raise LintProcessError(f"Error running {tool_name}: {exc}") from exc

    return result.returncode


def run_linter_adapter(
    tool_name: str,
    build_cmd: Callable[[dict], list[str]],
    config_section: str | None = None,
) -> int:
    """Load lint config, build command args, and execute the tool."""
    cfg = load_lint_config(tool_name, config_section, config_path=_CONFIG)
    cmd = build_cmd(cfg)
    return run_linter_command(tool_name, cmd)
