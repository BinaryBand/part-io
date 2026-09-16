"""Tests for the cli.library._cut_rules module."""

from __future__ import annotations

from pathlib import Path

import pytest

from partio.cli.library import _cut_rules
from partio.core.ports import CutRuleEntry


@pytest.fixture(autouse=True)
def _cut_rules_path(tmp_path, monkeypatch):
    monkeypatch.setattr(_cut_rules, "DEFAULT_CUT_RULES_PATH", tmp_path / "cut_rules.json")


def test_cut_rule_store_round_trips_an_entry() -> None:
    """A saved rule survives a store round trip unchanged."""
    entry = CutRuleEntry(
        id="r1", label="The Daily", opening_path=Path("open.mp3"), closing_path=Path("close.mp3")
    )

    _cut_rules.cut_rule_store().add_item(entry)

    assert _cut_rules.cut_rule_store().list_items() == [entry]


def test_cut_rules_returns_every_saved_rule() -> None:
    """cut_rules() reads straight through to the store."""
    _cut_rules.cut_rule_store().add_item(
        CutRuleEntry(id="r1", label="A", opening_path=Path("a1.mp3"), closing_path=Path("a2.mp3"))
    )
    _cut_rules.cut_rule_store().add_item(
        CutRuleEntry(id="r2", label="B", opening_path=Path("b1.mp3"), closing_path=Path("b2.mp3"))
    )

    assert [entry.label for entry in _cut_rules.cut_rules()] == ["A", "B"]


def test_cut_rules_is_empty_when_nothing_is_saved() -> None:
    """An absent store reads as no rules rather than as an error."""
    assert _cut_rules.cut_rules() == []


def test_cut_rules_tolerates_a_corrupt_store(monkeypatch, tmp_path) -> None:
    """A broken rules file degrades to "no rules" instead of blocking a cut."""
    broken = tmp_path / "cut_rules.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(_cut_rules, "DEFAULT_CUT_RULES_PATH", broken)

    assert _cut_rules.cut_rules() == []
