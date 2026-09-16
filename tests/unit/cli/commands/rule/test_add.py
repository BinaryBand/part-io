"""Tests for the `rule add` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from partio.cli.commands.rule import add as rule_add
from partio.cli.library import _cut_rules as store_module


@pytest.fixture(autouse=True)
def _cut_rules_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_CUT_RULES_PATH", tmp_path / "cut_rules.json")


def test_add_saves_a_rule(capsys):
    """A saved rule round-trips through the store."""
    rule_add.add(ctx=None, label="The Daily", opening=Path("open.mp3"), closing=Path("close.mp3"))

    entries = store_module.cut_rule_store().list_items()
    assert [(e.label, e.opening_path, e.closing_path) for e in entries] == [
        ("The Daily", Path("open.mp3"), Path("close.mp3"))
    ]
    assert "The Daily" in capsys.readouterr().out
