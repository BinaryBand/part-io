"""JSON-file-backed implementation of the ItemStore CRUD protocol."""

from __future__ import annotations

import json
from collections.abc import Callable  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Generic, TypeVar

T = TypeVar("T")


class JsonItemStore(Generic[T]):
    """Persist a list of items as JSON, keyed by an id.

    Serialization is delegated to *to_dict*/*from_dict* callables so the same
    store can back different item kinds (audio paths today; snippets or cut
    rules later) without changing the file-I/O plumbing.
    """

    def __init__(
        self,
        *,
        path: Path,
        to_dict: Callable[[T], dict],
        from_dict: Callable[[dict], T],
        item_id: Callable[[T], str],
    ) -> None:
        """Bind this store to a JSON file at *path* with the given (de)serializers."""
        self._path = path
        self._to_dict = to_dict
        self._from_dict = from_dict
        self._item_id = item_id

    def list_items(self) -> list[T]:
        """Return every stored item."""
        return [self._from_dict(raw) for raw in self._read_all()]

    def add_item(self, item: T) -> None:
        """Append *item*, raising ``ValueError`` if its id is already in use."""
        new_id = self._item_id(item)
        items = self.list_items()
        if any(self._item_id(existing) == new_id for existing in items):
            raise ValueError(f"Item with id {new_id!r} already exists")
        self._write_all([*items, item])

    def get_item(self, item_id: str) -> T | None:
        """Return the item with *item_id*, or ``None`` if not found."""
        for item in self.list_items():
            if self._item_id(item) == item_id:
                return item
        return None

    def remove_item(self, item_id: str) -> None:
        """Remove the item with *item_id*, if present."""
        remaining = [item for item in self.list_items() if self._item_id(item) != item_id]
        self._write_all(remaining)

    def _read_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write_all(self, items: list[T]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._to_dict(item) for item in items]
        self._path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


__all__ = ["JsonItemStore"]
