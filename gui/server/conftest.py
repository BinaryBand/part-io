"""Project-root conftest: auto-format and auto-fix before each test session.

Also defaults ``PARTIO_ROOT`` to the partio repo root two directories up
(``gui/server`` -> ``gui`` -> repo root) so the test suite exercises
``app.settings`` against the real checkout without requiring every developer
to export it by hand. Set explicitly in the environment to override.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTIO_REPO_ROOT = ROOT.parents[1]

os.environ.setdefault("PARTIO_ROOT", str(PARTIO_REPO_ROOT))


def pytest_configure(config) -> None:
    subprocess.run(["ruff", "format", str(ROOT)], cwd=ROOT, check=False)
    subprocess.run(["ruff", "check", "--fix", str(ROOT)], cwd=ROOT, check=False)
