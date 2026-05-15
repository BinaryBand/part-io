"""Shared wrapper helpers for lint entrypoints."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def run_single_tool_entrypoint(run_tool_fn: Callable[[str], int], tool_key: str) -> int:
    """Run one registered tool through a single-tool entrypoint contract."""
    return run_tool_fn(tool_key)


def run_multi_tool_entrypoint(
    run_tool_fn: Callable[[str], int],
    tool_keys: Sequence[str],
) -> int:
    """Run registered tools in order and stop at the first failure."""
    exit_code = 0
    for tool_key in tool_keys:
        exit_code = run_tool_fn(tool_key)
        if exit_code != 0:
            break
    return exit_code
