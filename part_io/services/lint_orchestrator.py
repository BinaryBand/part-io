"""Lint task orchestration service used by CLI entrypoints.

This module expects callable ports from part_io.models.ports.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

from part_io.models.ports import (
    LoadRegistryFn,
    SelectTasksFn,
    TasksForProfileFn,
    WriteLintReportFn,
)
from part_io.models.registry import TaskRegistry, TaskSpec
from part_io.models.results import LintRunReport, TaskRunResult


def select_lint_tasks(
    selected: list[str] | None,
    *,
    profile: str | None,
    load_registry_fn: LoadRegistryFn,
    select_tasks_fn: SelectTasksFn,
    tasks_for_profile_fn: TasksForProfileFn,
) -> tuple[TaskRegistry, list[TaskSpec]]:
    """Load registry and resolve selected lint tasks."""
    registry = load_registry_fn()
    chosen = (
        select_tasks_fn(registry, selected) if selected else tasks_for_profile_fn(registry, profile)
    )
    return registry, chosen


def execute_lint_tasks(
    chosen: list[TaskSpec],
    *,
    run_module_fn: Callable[[str], int],
    on_task_start: Callable[[TaskSpec], None] | None = None,
) -> tuple[list[TaskRunResult], int]:
    """Execute lint tasks in order and stop at first failure."""
    results: list[TaskRunResult] = []
    exit_code = 0

    for task in chosen:
        if on_task_start is not None:
            on_task_start(task)

        start = perf_counter()
        rc = run_module_fn(task.module)
        duration_ms = round((perf_counter() - start) * 1000, 2)

        results.append(
            TaskRunResult(
                target=task.target,
                module=task.module,
                required=task.required,
                exit_code=rc,
                duration_ms=duration_ms,
            )
        )

        if rc != 0:
            exit_code = rc
            break

    return results, exit_code


def write_lint_report_if_requested(
    report_json: Path | None,
    *,
    selected: list[str] | None,
    profile: str | None,
    default_profile: str,
    results: list[TaskRunResult],
    exit_code: int,
    write_lint_report_fn: WriteLintReportFn,
) -> None:
    """Write lint report when a report output path is provided."""
    if report_json is None:
        return

    selected_profile = None if selected else profile or default_profile
    write_lint_report_fn(
        report_json,
        LintRunReport(
            selected_profile=selected_profile,
            results=results,
            exit_code=exit_code,
        ),
    )


