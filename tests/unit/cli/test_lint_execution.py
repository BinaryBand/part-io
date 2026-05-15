"""Tests for registered lint tool execution glue."""

from __future__ import annotations

from part_io.adapters.errors import LintBoundaryError
from part_io.cli.lint import execution as lint_execution


def test_run_registered_tool_delegates_to_adapter(monkeypatch) -> None:
    """Registered tool execution should route through unified adapter once."""
    calls: list[tuple[str, str | None, list[str]]] = []

    def fake_adapter(tool_name, build_cmd, config_section=None):
        calls.append((tool_name, config_section, build_cmd({})))
        return 0

    monkeypatch.setattr(lint_execution, "run_linter_adapter", fake_adapter)

    rc = lint_execution.run_registered_tool("ty")

    assert rc == 0
    assert calls == [("ty", None, ["ty", "check"])]


def test_run_registered_tool_maps_boundary_errors_to_two(monkeypatch) -> None:
    """Execution wrapper should map boundary failures to exit code 2."""
    monkeypatch.setattr(
        lint_execution,
        "run_linter_adapter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LintBoundaryError("boom")),
    )

    assert lint_execution.run_registered_tool("ty") == 2


