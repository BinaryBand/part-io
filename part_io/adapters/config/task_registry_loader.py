"""Declarative task registry loading and validation helpers.

Callable ports implemented here are defined in part_io.models.ports.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from part_io.models.registry import TaskRegistry, TaskSpec

DEFAULT_REGISTRY_PATH = Path("config/tasks.toml")


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML from path."""
    if not path.exists():
        raise ValueError(f"Registry file is missing: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_tasks(raw_tasks: object) -> dict[str, TaskSpec]:
    """Parse and validate task list from TOML."""
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Invalid tasks: expected a non-empty [[tasks]] array")

    parsed: dict[str, TaskSpec] = {}
    for index, entry in enumerate(raw_tasks, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid tasks[{index}]: expected table")
        entry_map = cast(dict[str, object], entry)

        try:
            task = TaskSpec.model_validate(entry_map)
        except ValidationError as exc:
            errors = exc.errors()
            if errors:
                field = errors[0]["loc"][0]
                msg = errors[0]["msg"]
                raise ValueError(f"Invalid tasks[{index}].{field}: {msg}") from exc
            raise ValueError(f"Invalid tasks[{index}]: validation failed") from exc

        if task.target in parsed:
            raise ValueError(f"Duplicate task target: {task.target}")

        parsed[task.target] = task

    return parsed


def _parse_profiles(raw_profiles: object, known_targets: set[str]) -> dict[str, tuple[str, ...]]:
    """Parse and validate profile definitions from TOML."""
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("Invalid [profiles]: expected a non-empty table")

    parsed: dict[str, tuple[str, ...]] = {}
    for profile_name, value in raw_profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError("Invalid profile name: expected non-empty string")
        if not isinstance(value, list) or not value:
            raise ValueError(f"Invalid profile '{profile_name}': expected non-empty array")

        targets: list[str] = []
        for index, task_id in enumerate(value, start=1):
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(
                    f"Invalid profile '{profile_name}' entry {index}: expected non-empty string"
                )
            task_id = task_id.strip()
            if task_id not in known_targets:
                raise ValueError(f"Unknown task '{task_id}' in profile '{profile_name}'")
            targets.append(task_id)

        parsed[profile_name.strip()] = tuple(targets)

    return parsed


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> TaskRegistry:
    """Load and validate task registry TOML from *path*."""
    raw = _load_toml(path)
    tasks = _parse_tasks(raw.get("tasks"))
    profiles = _parse_profiles(raw.get("profiles"), set(tasks))

    default_profile = raw.get("default_profile", "strict")
    if not isinstance(default_profile, str) or not default_profile.strip():
        raise ValueError("Invalid default_profile: expected non-empty string")
    default_profile = default_profile.strip()

    try:
        return TaskRegistry(
            tasks=tasks,
            profiles=profiles,
            default_profile=default_profile,
        )
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            msg = errors[0]["msg"]
            raise ValueError(f"Invalid registry: {msg}") from exc
        raise ValueError("Invalid registry: validation failed") from exc


def tasks_for_profile(registry: TaskRegistry, profile: str | None = None) -> list[TaskSpec]:
    """Return ordered tasks for *profile* or the registry default profile."""
    selected_profile = profile or registry.default_profile
    if selected_profile not in registry.profiles:
        raise ValueError(f"Unknown profile '{selected_profile}'")
    return [registry.tasks[target] for target in registry.profiles[selected_profile]]


def select_tasks(registry: TaskRegistry, task_ids: list[str]) -> list[TaskSpec]:
    """Return ordered tasks by explicit IDs and fail on unknown IDs."""
    selected: list[TaskSpec] = []
    unknown: list[str] = []
    for task_id in task_ids:
        if task_id in registry.tasks:
            selected.append(registry.tasks[task_id])
        else:
            unknown.append(task_id)

    if unknown:
        missing = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unknown lint task(s): {missing}")

    return selected


__all__ = ["DEFAULT_REGISTRY_PATH", "load_registry", "tasks_for_profile", "select_tasks"]
