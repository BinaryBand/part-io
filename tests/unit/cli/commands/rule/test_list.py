"""Tests for the `rule list` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from partio.cli.commands.rule.list import list_rules
from partio.cli.library import _cut_rules as store_module
from partio.cli.output import ExitCode
from partio.core.ports import CutRuleEntry


@pytest.fixture(autouse=True)
def _cut_rules_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_CUT_RULES_PATH", tmp_path / "cut_rules.json")


def test_list_shows_saved_rules(capsys):
    """Every saved rule appears with its label and jingle paths."""
    store_module.cut_rule_store().add_item(
        CutRuleEntry(
            id="r1", label="Show A", opening_path=Path("a1.mp3"), closing_path=Path("a2.mp3")
        )
    )

    list_rules(ctx=None)

    out = capsys.readouterr().out
    assert "Show A" in out
    assert "a1.mp3" in out
    assert "a2.mp3" in out


def test_list_empty_store_points_at_rule_add(capsys):
    """With nothing saved, say how to start."""
    with pytest.raises(SystemExit) as exc_info:
        list_rules(ctx=None)

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert "rule add" in capsys.readouterr().out
