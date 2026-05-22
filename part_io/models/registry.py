"""Backward-compat re-export — prefer ``part_io.models.tasks.registry``."""

from part_io.models.tasks.registry import TaskRegistry, TaskSpec

__all__ = ["TaskSpec", "TaskRegistry"]
