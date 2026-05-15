"""Tests for declarative lint task registry behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from part_io.adapters.config.task_registry_loader import (
    load_registry,
    select_tasks,
    tasks_for_profile,
)


def test_load_registry_default_profile() -> None:
    """The default project registry should load and expose strict profile tasks."""
    registry = load_registry()
    tasks = tasks_for_profile(registry)
    assert tasks
    assert tasks[0].target == "lint.coverage"


def test_load_registry_rejects_unknown_default_profile(tmp_path: Path) -> None:
    """Loading should fail when default_profile does not exist in profiles."""
    path = tmp_path / "tasks.toml"
    path.write_text(
        """
        default_profile = "strict"

        [[tasks]]
        target = "lint.demo"
        module = "part_io.cli.lint.demo"
        description = "demo"

        [profiles]
        ci = ["lint.demo"]
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown default_profile"):
        load_registry(path)


def test_load_registry_rejects_unknown_profile_task(tmp_path: Path) -> None:
    """Loading should fail when profiles reference unknown task IDs."""
    path = tmp_path / "tasks.toml"
    path.write_text(
        """
        default_profile = "strict"

        [[tasks]]
        target = "lint.demo"
        module = "part_io.cli.lint.demo"
        description = "demo"

        [profiles]
        strict = ["lint.unknown"]
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown task"):
        load_registry(path)


def test_select_tasks_rejects_unknown_task_ids() -> None:
    """Explicit selection should fail-closed for unknown lint IDs."""
    registry = load_registry()

    with pytest.raises(ValueError, match="Unknown lint task"):
        select_tasks(registry, ["lint.missing"])


