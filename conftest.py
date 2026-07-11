"""Project-root conftest: auto-format and auto-fix before each test session."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pytest_configure(config) -> None:
    subprocess.run(["ruff", "format", str(ROOT)], cwd=ROOT)
    subprocess.run(["ruff", "check", "--fix", str(ROOT)], cwd=ROOT)
