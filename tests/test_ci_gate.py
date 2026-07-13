"""CI/CD gate: fail the suite when any linter, type checker, or dead-code scanner reports issues."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)


def test_ruff_check() -> None:
    """Ruff check must produce zero diagnostics after auto-fix."""
    result = _run(["ruff", "check", str(ROOT)])
    assert result.returncode == 0, (
        f"ruff check failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


def test_ruff_format() -> None:
    """Ruff format --check must report no reformats needed."""
    result = _run(["ruff", "format", "--check", str(ROOT)])
    assert result.returncode == 0, (
        f"ruff format --check found unformatted files "
        f"(exit {result.returncode}):\n\n{result.stdout}"
    )


def test_ty_check() -> None:
    """Ty check must produce zero diagnostics."""
    result = _run(["ty", "check", str(ROOT)])
    assert result.returncode == 0, (
        f"ty check failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


def test_import_linter() -> None:
    """import-linter contracts must all pass."""
    result = _run(["lint-imports", "--config", str(ROOT / "pyproject.toml")])
    assert result.returncode == 0, (
        f"import-linter failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


def test_vulture() -> None:
    """Vulture must report no dead code above the confidence threshold."""
    result = _run(["vulture"])
    assert result.returncode == 0, (
        f"vulture found dead code (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.skipif(shutil.which("ast-grep") is None, reason="ast-grep executable is not installed")
def test_astgrep() -> None:
    """ast-grep architectural rules must all pass.

    Requires the `sg` CLI -- install via `cargo install ast-grep` or
    `brew install ast-grep`. It is not a Python package and cannot be
    declared in pyproject.toml.
    """
    result = _run(["ast-grep", "scan", "--config", str(ROOT / "sgconfig.yml"), str(ROOT)])
    assert result.returncode == 0, (
        f"ast-grep found violations (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )
