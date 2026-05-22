"""Callable ports for task registry loading and selection."""

from __future__ import annotations

from collections.abc import Callable

from part_io.models.tasks.registry import TaskRegistry, TaskSpec

LoadRegistryFn = Callable[[], TaskRegistry]
SelectTasksFn = Callable[[TaskRegistry, list[str]], list[TaskSpec]]
TasksForProfileFn = Callable[[TaskRegistry, str | None], list[TaskSpec]]


__all__ = ["LoadRegistryFn", "SelectTasksFn", "TasksForProfileFn"]
