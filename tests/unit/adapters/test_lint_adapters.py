"""Unit tests for lint adapter contract and implementations."""

from __future__ import annotations

import pytest

from part_io.adapters.errors import LintConfigError, LintProcessError
from part_io.adapters.lint import runner as adapter_module
from part_io.cli.lint.registry import build_tool_cmd


def test_adapter_coverage_build_cmd() -> None:
    """Coverage adapter should build pytest command with configurable floor."""
    cfg = {"floor": 90}
    cmd = build_tool_cmd("coverage", cfg)

    assert cmd[:3] == ["python", "-m", "pytest"]
    assert "--cov-fail-under=90" in cmd


def test_adapter_coverage_build_cmd_uses_default_floor() -> None:
    """Coverage should use default floor if not in config."""
    cmd = build_tool_cmd("coverage", {})

    assert "--cov-fail-under=80" in cmd


def test_adapter_cpd_build_cmd() -> None:
    """CPD adapter should build jscpd command with default config path."""
    cmd = build_tool_cmd("cpd", {})

    assert cmd[:4] == ["npx", "--yes", "jscpd@4.0.5", "--config"]
    assert cmd[-1] == "config/jscpd.json"


def test_adapter_semgrep_build_cmd() -> None:
    """Semgrep adapter should build hardcoded semgrep command."""
    cmd = build_tool_cmd("semgrep", {})

    assert cmd[0] == "semgrep"
    assert "scan" in cmd
    assert "--error" in cmd


def test_adapter_ty_build_cmd() -> None:
    """Ty adapter should build minimal ty command."""
    cmd = build_tool_cmd("ty", {})

    assert cmd == ["ty", "check"]


def test_adapter_vulture_build_cmd() -> None:
    """Vulture adapter should build command with optional flags from config."""
    cfg = {"min_confidence": 85, "paths": ["src"]}
    cmd = build_tool_cmd("vulture", cfg)

    assert "vulture" in cmd
    assert "--min-confidence" in cmd
    assert "85" in cmd
    assert "src" in cmd


def test_adapter_vulture_build_cmd_uses_defaults() -> None:
    """Vulture should use default paths if not in config."""
    cmd = build_tool_cmd("vulture", {})

    assert "part_io" in cmd
    assert "tests" in cmd


def test_adapter_lizard_build_cmd() -> None:
    """Lizard adapter should build command with optional flags from config."""
    cfg = {"ccn": 12, "length": 50, "warnings_only": True, "paths": ["app"]}
    cmd = build_tool_cmd("lizard", cfg)

    assert "lizard" in cmd
    assert "--CCN" in cmd
    assert "12" in cmd
    assert "--length" in cmd
    assert "--warnings_only" in cmd
    assert "app" in cmd


def test_adapter_lizard_build_cmd_uses_defaults() -> None:
    """Lizard should use default paths if not in config."""
    cmd = build_tool_cmd("lizard", {})

    assert "part_io" in cmd


def test_unified_adapter_handles_missing_config_section(monkeypatch, tmp_path) -> None:
    """Adapter should gracefully handle missing config sections."""
    config_path = tmp_path / "lint.toml"
    config_path.write_text("# empty config\n", encoding="utf-8")

    monkeypatch.setattr(adapter_module, "_CONFIG", config_path)

    # Mock run_resolved to avoid needing actual executables
    mock_result = type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(adapter_module, "run_resolved", lambda _cmd: mock_result)

    def fake_build(_cfg):
        return ["fake_tool"]

    # Should not raise; returns 0 for mocked command
    rc = adapter_module.run_linter_adapter("fake_tool", fake_build, config_section="missing")
    assert rc == 0


def test_unified_adapter_error_handling_on_corrupt_config(monkeypatch, tmp_path) -> None:
    """Adapter should fail gracefully on corrupt TOML."""
    config_path = tmp_path / "lint.toml"
    config_path.write_text("[invalid toml\n", encoding="utf-8")

    monkeypatch.setattr(adapter_module, "_CONFIG", config_path)

    def fake_build(_cfg):
        return ["fake_tool"]

    with pytest.raises(LintConfigError, match="Error loading fake_tool config"):
        adapter_module.run_linter_adapter("fake_tool", fake_build, config_section="any")


def test_unified_adapter_handles_command_errors(monkeypatch, tmp_path) -> None:
    """Adapter should map command construction/execution errors to exit code 2."""
    config_path = tmp_path / "lint.toml"
    config_path.write_text("[demo]\n", encoding="utf-8")
    monkeypatch.setattr(adapter_module, "_CONFIG", config_path)

    monkeypatch.setattr(
        adapter_module,
        "run_resolved",
        lambda _cmd: (_ for _ in ()).throw(FileNotFoundError("missing tool")),
    )

    with pytest.raises(LintProcessError, match="Error running demo"):
        adapter_module.run_linter_adapter("demo", lambda _cfg: ["demo"], config_section="demo")
