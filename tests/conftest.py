"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest

from partio.cli.library import _cache as cache_module
from partio.cli.library import _feeds as feeds_module
from partio.cli.library import refresh

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"


def _cleanup_coverage_temp_files(root: Path | None = None) -> int:
    """Delete top-level ``.coverage.*`` files from *root* and return count.

    This intentionally avoids recursive traversal to prevent runaway cleanup
    behavior in large workspaces.
    """
    workspace_root = root or Path.cwd()
    removed = 0

    for candidate in workspace_root.iterdir():
        if not candidate.name.startswith(".coverage."):
            continue

        # Only remove files/symlinks; never recurse into directories.
        if not (candidate.is_file() or candidate.is_symlink()):
            continue

        try:
            candidate.unlink()
            removed += 1
        except OSError:
            continue

    return removed


def pytest_sessionstart(session):
    """Remove stale coverage temp files before tests start."""
    _ = session
    _cleanup_coverage_temp_files()


def pytest_sessionfinish(session, exitstatus):
    """Clean up coverage temp files after test session completes."""
    _ = session
    _ = exitstatus
    _cleanup_coverage_temp_files()


@pytest.fixture(autouse=True)
def _isolate_stores(tmp_path, monkeypatch):
    """Point every persisted store at a per-test scratch file.

    The store paths default to ``static/*.json`` *relative to the working
    directory*, so a command under test would otherwise write into the real
    repository library.  Applied globally rather than per module because
    forgetting it fails silently: a suite run once left 16 junk entries in the
    committed ``static/library.json`` before this existed.

    Feed fetches are memoized for the life of the process, so the memo is
    dropped here too -- otherwise one test's stubbed episodes would leak into
    the next test's library.
    """
    monkeypatch.setattr(cache_module, "DEFAULT_LIBRARY_PATH", tmp_path / "library.json")
    monkeypatch.setattr(feeds_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")
    refresh()
    yield
    refresh()


def _static_snapshot() -> dict[str, tuple[int, int]]:
    """Fingerprint every file under ``static/`` by size and mtime."""
    if not STATIC_DIR.is_dir():
        return {}
    snapshot: dict[str, tuple[int, int]] = {}
    for path in STATIC_DIR.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


@pytest.fixture(autouse=True)
def _guard_static_dir():
    """Fail the offending test if it writes into the repository's ``static/``.

    :func:`_isolate_stores` covers the JSON stores, but commands also write
    audio (seed clips, review bundles).  This catches any path that escapes
    ``tmp_path`` and names the test that did it, rather than leaving a stray
    file to be noticed later.
    """
    before = _static_snapshot()
    yield
    after = _static_snapshot()
    # A changed file appears on both sides of the symmetric difference, so
    # collect names into a set before reporting.
    changed = sorted({name for name, _stat in set(before.items()) ^ set(after.items())})
    assert not changed, (
        "test wrote into the repository's static/ directory; redirect it at tmp_path:\n  "
        + "\n  ".join(changed)
    )
