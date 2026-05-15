"""Callable ports for lint tool configuration loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

LoadLintConfigFn = Callable[[str, str | None, Path | None], dict]


__all__ = ["LoadLintConfigFn"]
