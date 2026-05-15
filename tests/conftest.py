"""Pytest configuration and fixtures."""

import pytest

from part_io.models.registry import TaskRegistry, TaskSpec
from part_io.utils.coverage import cleanup_coverage_temp_files


def pytest_sessionstart(session):
    """Remove stale coverage temp files before tests start."""
    _ = session
    cleanup_coverage_temp_files()


def pytest_sessionfinish(session, exitstatus):
    """Clean up coverage temp files after test session completes."""
    _ = session
    _ = exitstatus
    cleanup_coverage_temp_files()


@pytest.fixture
def test_registry() -> TaskRegistry:
    """Provide a simple test task registry with alpha and beta tasks."""
    tasks: dict[str, TaskSpec] = {
        "lint.alpha": TaskSpec(
            target="lint.alpha",
            module="part_io.cli.lint.alpha",
            description="alpha",
            required=True,
        ),
        "lint.beta": TaskSpec(
            target="lint.beta",
            module="part_io.cli.lint.beta",
            description="beta",
            required=True,
        ),
    }
    profiles: dict[str, tuple[str, ...]] = {
        "strict": ("lint.alpha", "lint.beta"),
        "ci": ("lint.beta",),
    }
    return TaskRegistry(tasks=tasks, profiles=profiles, default_profile="strict")


