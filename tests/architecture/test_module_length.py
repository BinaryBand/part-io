"""Verify no production module exceeds the length cap."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_LINES = 400
SKIP_DIRS: list[Path] = []


def _python_files(base: Path) -> list[Path]:
    return [path for path in sorted(base.rglob("*.py")) if path.name != "__init__.py"]


def test_module_length() -> None:
    """Every .py file in part_io/ must stay under MAX_LINES lines."""
    for path in _python_files(ROOT / "part_io"):
        if any(skip in [*path.parents, path] for skip in SKIP_DIRS):
            continue
        text = path.read_text(encoding="utf-8")
        lines = len(text.splitlines())
        assert lines <= MAX_LINES, (
            f"{path.relative_to(ROOT)} has {lines} lines (cap is {MAX_LINES})"
        )
