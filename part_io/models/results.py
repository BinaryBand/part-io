"""Backward-compat re-export — prefer ``part_io.models.tasks.results``."""

from part_io.models.tasks.results import LintRunReport, TaskRunResult

__all__ = ["TaskRunResult", "LintRunReport"]
