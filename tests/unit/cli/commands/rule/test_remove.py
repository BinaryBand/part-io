"""Tests for the `rule remove` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from partio.cli.commands.rule import remove as remove_module
from partio.cli.commands.rule.remove import remove
from partio.cli.library import _cut_rules as store_module
from partio.cli.output import ExitCode
from partio.cli.select import GO_BACK
from partio.core.ports import CutRuleEntry


@pytest.fixture(autouse=True)
def _cut_rules_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DEFAULT_CUT_RULES_PATH", tmp_path / "cut_rules.json")


def test_remove_forgets_the_rule(capsys):
    """Removing an existing id drops it from the store."""
    store = store_module.cut_rule_store()
    store.add_item(
        CutRuleEntry(id="r1", label="A", opening_path=Path("a1.mp3"), closing_path=Path("a2.mp3"))
    )
    store.add_item(
        CutRuleEntry(id="r2", label="B", opening_path=Path("b1.mp3"), closing_path=Path("b2.mp3"))
    )

    remove(ctx=None, rule_id="r1")

    assert [e.id for e in store_module.cut_rule_store().list_items()] == ["r2"]
    assert "Removed r1" in capsys.readouterr().out


def test_remove_unknown_id_fails():
    """An unknown id is a user error, not a silent no-op."""
    with pytest.raises(SystemExit) as exc_info:
        remove(ctx=None, rule_id="nope")

    assert exc_info.value.code == ExitCode.USER_ERROR


def test_remove_without_an_id_picks_a_rule_by_name(monkeypatch):
    """Nobody should have to look up an id, so the default is a picker."""
    store_module.cut_rule_store().add_item(
        CutRuleEntry(id="r1", label="A", opening_path=Path("a1.mp3"), closing_path=Path("a2.mp3"))
    )
    captured = {}

    def _pick(_message, options, **_kwargs):
        captured["titles"] = [option.title for option in options]
        return "r1"

    monkeypatch.setattr(remove_module, "select_one", _pick)

    remove(ctx=None)

    assert captured["titles"] == ["A"]
    assert store_module.cut_rule_store().list_items() == []


def test_remove_without_rules_explains_how_to_add_one(capsys):
    """An empty store points at `rule add` instead of opening an empty picker."""
    with pytest.raises(SystemExit) as exc_info:
        remove(ctx=None)

    assert exc_info.value.code == ExitCode.NO_RESULT
    assert "rule add" in capsys.readouterr().out


def test_esc_in_the_picker_exits_cleanly(monkeypatch):
    """The picker is the command's first screen, so esc backs out of it."""
    store_module.cut_rule_store().add_item(
        CutRuleEntry(id="r1", label="A", opening_path=Path("a1.mp3"), closing_path=Path("a2.mp3"))
    )
    monkeypatch.setattr(remove_module, "select_one", lambda *_a, **_k: GO_BACK)

    with pytest.raises(SystemExit) as exc_info:
        remove(ctx=None)

    assert exc_info.value.code == ExitCode.OK
    assert len(store_module.cut_rule_store().list_items()) == 1
