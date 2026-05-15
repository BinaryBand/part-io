"""Callable ports for lint report writing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from part_io.models.results import LintRunReport

WriteLintReportFn = Callable[[Path, LintRunReport], None]


__all__ = ["WriteLintReportFn"]
