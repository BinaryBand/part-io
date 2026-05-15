"""Linting tests that enforce repository quality and architecture gates."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from part_io.adapters.process.runner import run_resolved

ROOT = Path(__file__).resolve().parents[2]


class TestRuff:
    """Ensure the codebase passes ruff linting and formatting checks."""

    PATHS = ["part_io/", "tests/"]

    def test_ruff_check(self):
        """Fail if ruff reports any lint violations."""
        result = run_resolved(
            ["poetry", "run", "ruff", "check", *self.PATHS],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_ruff_format(self):
        """Fail if ruff reports any formatting violations."""
        result = run_resolved(
            ["poetry", "run", "ruff", "format", "--check", *self.PATHS],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestTy:
    """Ensure the codebase passes the current ty gate."""

    def test_ty(self):
        """Fail if ty reports any type-checking violations."""
        result = run_resolved(
            ["poetry", "run", "ty", "check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestSemgrep:
    """Ensure the codebase passes the current Semgrep architecture gate."""

    def test_semgrep(self):
        """Fail if Semgrep reports any architecture or process violations."""
        result = run_resolved(
            [
                "poetry",
                "run",
                "semgrep",
                "scan",
                "--config",
                "config/semgrep/",
                "--error",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestVulture:
    """Ensure the codebase passes the current Vulture dead-code gate."""

    def test_vulture(self):
        """Fail if Vulture reports unused code above the configured confidence threshold."""
        result = run_resolved(["poetry", "run", "python", "-m", "part_io.cli.lint.vulture"])
        assert result.returncode == 0


# Vault keys whose names end in a recognised secret suffix are already caught
# by the generic host-vars-no-inline-secret-keys pattern-regex.
_GENERIC_SECRET_SUFFIX = re.compile(
    r".*(?:password|secret|token|api_key|private_key)$",
    re.IGNORECASE,
)


class TestLizard:
    """Enforce function complexity and length caps via Lizard."""

    def test_function_complexity(self):
        """Production functions must not exceed the configured CCN and length limits."""
        result = run_resolved(["poetry", "run", "python", "-m", "part_io.cli.lint.lizard"])
        assert result.returncode == 0


class TestCoverage:
    """Enforce the coverage floor defined in config/lint.toml."""

    def test_coverage_floor(self) -> None:
        """Fail if total coverage of part_io/ is below the configured floor."""
        result = run_resolved(
            ["poetry", "run", "python", "-m", "part_io.cli.lint.coverage"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("npx") is None, reason="npx executable is not installed")
class TestCpd:
    """Enforce duplicate-code gate via jscpd when executable is available."""

    def test_cpd(self) -> None:
        """Fail if jscpd reports duplicate code above configured threshold."""
        result = run_resolved(
            ["poetry", "run", "python", "-m", "part_io.cli.lint.cpd"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr
