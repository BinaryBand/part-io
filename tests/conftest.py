"""Pytest configuration and fixtures."""

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
