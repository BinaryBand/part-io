"""Tests for profile-aware generated task target behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from part_io.cli.generate import tasks as generate_tasks
from part_io.models.registry import TaskSpec


def _single_task() -> list[TaskSpec]:
    """Return one deterministic task for generation tests."""
    return [
        TaskSpec(
            target="lint.alpha",
            module="part_io.cli.lint.alpha",
            description="Alpha",
            required=True,
        )
    ]


def _configure_main_inputs(monkeypatch, output: Path, argv: list[str]) -> None:
    """Patch output path, task discovery, and argv for main() tests."""
    monkeypatch.setattr(generate_tasks, "_OUTPUT", output)
    monkeypatch.setattr(generate_tasks, "discover_tasks", lambda profile=None: _single_task())
    monkeypatch.setattr("sys.argv", argv)


def test_discover_tasks_uses_default_profile(monkeypatch, test_registry) -> None:
    """Default discovery should return tasks from default_profile in order."""
    monkeypatch.setattr(generate_tasks, "load_registry", lambda: test_registry)

    tasks = generate_tasks.discover_tasks()

    assert [task.target for task in tasks] == ["lint.alpha", "lint.beta"]


def test_discover_tasks_uses_explicit_profile(monkeypatch, test_registry) -> None:
    """Discovery should return only tasks in the requested profile."""
    monkeypatch.setattr(generate_tasks, "load_registry", lambda: test_registry)

    tasks = generate_tasks.discover_tasks(profile="ci")

    assert [task.target for task in tasks] == ["lint.beta"]


def test_discover_tasks_rejects_unknown_profile(monkeypatch, test_registry) -> None:
    """Discovery should fail-closed for unknown profile names."""
    monkeypatch.setattr(generate_tasks, "load_registry", lambda: test_registry)

    with pytest.raises(ValueError, match="Unknown profile"):
        generate_tasks.discover_tasks(profile="missing")


def test_render_contains_targets_and_recipes() -> None:
    """Rendered makefile should include all declared targets and module recipes."""
    tasks = [
        TaskSpec(
            target="lint.alpha",
            module="part_io.cli.lint.alpha",
            description="Alpha",
            required=True,
        ),
        TaskSpec(
            target="lint.beta",
            module="part_io.cli.lint.beta",
            description="Beta",
            required=True,
        ),
    ]

    content = generate_tasks._render(tasks)

    assert "TASK_TARGETS :=" in content
    assert "lint.alpha" in content
    assert "lint.beta" in content
    assert "poetry run python -m part_io.cli.lint.alpha" in content
    assert ".PHONY: lint.alpha lint.beta" in content


@pytest.mark.parametrize("preexisting", [None, "stale\n"])
def test_main_check_reports_missing_or_stale_file(
    monkeypatch,
    tmp_path: Path,
    preexisting: str | None,
) -> None:
    """Check mode should fail when output is missing or stale."""
    output = tmp_path / "generated.mk"
    if preexisting is not None:
        output.write_text(preexisting, encoding="utf-8")

    _configure_main_inputs(monkeypatch, output, ["generate", "--check"])

    with pytest.raises(SystemExit) as exc:
        generate_tasks.main()

    assert exc.value.code == 1


def test_main_writes_output(monkeypatch, tmp_path: Path) -> None:
    """Default mode should write rendered output to configured destination."""
    output = tmp_path / "generated.mk"
    _configure_main_inputs(monkeypatch, output, ["generate"])

    generate_tasks.main()

    assert output.exists()
    assert "lint.alpha" in output.read_text(encoding="utf-8")
