"""Tests for the adapters.store.json_store module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from partio.adapters.store.json_store import JsonItemStore


@dataclass(frozen=True)
class _Widget:
    id: str
    name: str


def _make_store(path: Path) -> JsonItemStore[_Widget]:
    return JsonItemStore(
        path=path,
        to_dict=lambda w: {"id": w.id, "name": w.name},
        from_dict=lambda raw: _Widget(id=raw["id"], name=raw["name"]),
        item_id=lambda w: w.id,
    )


def test_list_items_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """Listing a store backed by a nonexistent file should return an empty list."""
    store = _make_store(tmp_path / "missing.json")
    assert store.list_items() == []


def test_add_then_list_round_trips(tmp_path: Path) -> None:
    """An added item should reappear via list_items with equal fields."""
    store = _make_store(tmp_path / "items.json")
    store.add_item(_Widget(id="a", name="Alpha"))
    assert store.list_items() == [_Widget(id="a", name="Alpha")]


def test_add_item_rejects_duplicate_id(tmp_path: Path) -> None:
    """Adding an item whose id already exists must raise ValueError."""
    store = _make_store(tmp_path / "items.json")
    store.add_item(_Widget(id="a", name="Alpha"))
    with pytest.raises(ValueError, match="already exists"):
        store.add_item(_Widget(id="a", name="Duplicate"))


def test_get_item_returns_none_when_absent(tmp_path: Path) -> None:
    """get_item should return None for an id that was never added."""
    store = _make_store(tmp_path / "items.json")
    assert store.get_item("missing") is None


def test_get_item_returns_matching_item(tmp_path: Path) -> None:
    """get_item should return the item whose id matches."""
    store = _make_store(tmp_path / "items.json")
    store.add_item(_Widget(id="a", name="Alpha"))
    store.add_item(_Widget(id="b", name="Beta"))
    assert store.get_item("b") == _Widget(id="b", name="Beta")


def test_remove_item_drops_only_the_matching_entry(tmp_path: Path) -> None:
    """remove_item should delete exactly the entry with the given id."""
    store = _make_store(tmp_path / "items.json")
    store.add_item(_Widget(id="a", name="Alpha"))
    store.add_item(_Widget(id="b", name="Beta"))
    store.remove_item("a")
    assert store.list_items() == [_Widget(id="b", name="Beta")]


def test_remove_item_is_a_noop_when_absent(tmp_path: Path) -> None:
    """Removing a nonexistent id should not raise or change the store."""
    store = _make_store(tmp_path / "items.json")
    store.add_item(_Widget(id="a", name="Alpha"))
    store.remove_item("missing")
    assert store.list_items() == [_Widget(id="a", name="Alpha")]
