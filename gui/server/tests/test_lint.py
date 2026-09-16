"""Gate: fail the suite when ruff, ty, or the module-length cap reports issues.

Scoped to this project only (``gui/server``), independent of the partio repo
root's own gate. Mirrors ``tests/test_lint.py`` at the repo root.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAX_MODULE_LINES = 400


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)


def test_ruff_check() -> None:
    """ruff check must produce zero diagnostics after auto-fix."""
    result = _run(["ruff", "check", str(ROOT)])
    assert result.returncode == 0, (
        f"ruff check failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


def test_ruff_format() -> None:
    """ruff format --check must report no reformats needed."""
    result = _run(["ruff", "format", "--check", str(ROOT)])
    assert result.returncode == 0, (
        f"ruff format --check found unformatted files (exit {result.returncode}):\n\n"
        f"{result.stdout}"
    )


def test_ty_check() -> None:
    """ty check must produce zero diagnostics."""
    result = _run(["ty", "check", str(ROOT)])
    assert result.returncode == 0, (
        f"ty check failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


def test_module_length() -> None:
    """No source module may exceed MAX_MODULE_LINES lines."""
    offenders: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).parts
        if any(part.startswith(".") for part in parts) or "tests" in parts:
            continue
        line_count = path.read_text().count("\n") + 1
        if line_count > MAX_MODULE_LINES:
            offenders.append(f"{path.relative_to(ROOT)}: {line_count} lines")
    listing = "\n".join(offenders)
    assert not offenders, f"modules exceed {MAX_MODULE_LINES} lines; split them:\n\n{listing}"
