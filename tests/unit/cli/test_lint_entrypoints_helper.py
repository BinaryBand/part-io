"""Tests for shared lint entrypoint helpers."""

from __future__ import annotations

from part_io.cli.lint import entrypoints as lint_entrypoints


def test_run_single_tool_entrypoint_returns_delegate_code() -> None:
    """Single-tool helper should return the delegated exit code."""
    assert lint_entrypoints.run_single_tool_entrypoint(lambda _tool: 7, "ruff") == 7


def test_run_multi_tool_entrypoint_stops_on_first_failure() -> None:
    """Multi-tool helper should stop after the first non-zero return code."""
    calls: list[str] = []

    def fake_runner(tool_key: str) -> int:
        calls.append(tool_key)
        return 3 if tool_key == "second" else 0

    rc = lint_entrypoints.run_multi_tool_entrypoint(fake_runner, ("first", "second", "third"))

    assert rc == 3
    assert calls == ["first", "second"]
