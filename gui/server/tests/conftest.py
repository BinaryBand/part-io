"""Test fixtures: keep every test from touching the real partio library.

Mirrors the equivalent fixture in the partio repo root's own
``tests/conftest.py``. Store paths default to ``static/*.json`` *relative to
the working directory* (see ``app.settings.resolve_partio_root``'s
``os.chdir``), so without this, a test that adds/removes a feed or seed would
write into the real repository's ``static/`` instead of a scratch directory.
"""

from __future__ import annotations

import pytest
from partio.cli.library import _cache as cache_module
from partio.cli.library import _cut_rules as cut_rules_module
from partio.cli.library import _feeds as feeds_module
from partio.cli.library import refresh


@pytest.fixture(autouse=True)
def _isolate_stores(tmp_path, monkeypatch):
    """Point every persisted store at a per-test scratch file."""
    monkeypatch.setattr(cache_module, "DEFAULT_LIBRARY_PATH", tmp_path / "library.json")
    monkeypatch.setattr(feeds_module, "DEFAULT_FEEDS_PATH", tmp_path / "feeds.json")
    monkeypatch.setattr(cut_rules_module, "DEFAULT_CUT_RULES_PATH", tmp_path / "cut_rules.json")
    refresh()
    yield
    refresh()
