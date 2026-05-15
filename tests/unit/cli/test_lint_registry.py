"""Tests for centralized lint tool registry and command builders."""

from __future__ import annotations

import pytest

from part_io.cli.lint import registry as lint_registry
from part_io.models.lint import ToolSpec


def test_tool_specs_are_typed() -> None:
    """Central tool registry should store typed ToolSpec entries."""
    assert isinstance(lint_registry.TOOL_SPECS["ruff"], ToolSpec)
    assert lint_registry.TOOL_SPECS["ruff"].config_section == "ruff"
    assert lint_registry.TOOL_SPECS["cpd"].executable == "npx"


def test_build_tool_cmd_ruff_coverage_and_cpd() -> None:
    """Central builders should render expected command args for key tools."""
    ruff_cmd = lint_registry.build_tool_cmd("ruff", {"mode": "check", "paths": ["part_io"]})
    cov_cmd = lint_registry.build_tool_cmd("coverage", {"floor": 90})
    cpd_cmd = lint_registry.build_tool_cmd("cpd", {})

    assert ruff_cmd[:2] == ["ruff", "check"]
    assert ruff_cmd[-1] == "part_io"
    assert "--cov-fail-under=90" in cov_cmd
    assert cpd_cmd == ["npx", "--yes", "jscpd@4.0.5", "--config", "config/jscpd.json"]


def test_build_tool_cmd_unknown_key_fails_closed() -> None:
    """Unknown tool keys should fail closed instead of silently defaulting."""
    with pytest.raises(ValueError, match="Unknown lint tool key"):
        lint_registry.build_tool_cmd("missing", {})


def test_build_tool_cmd_is_still_registry_owned() -> None:
    """Registry should continue to own tool command builders."""
    assert lint_registry.build_tool_cmd("ty", {}) == ["ty", "check"]
