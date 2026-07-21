"""Generic CRUD port for persisted, user-managed items."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypeVar

T = TypeVar("T")


class AudioPathKind(StrEnum):
    """Whether a remembered audio path is a long recording or a short reference clip."""

    SOURCE = "source"
    SAMPLE = "sample"


@dataclass(frozen=True)
class AudioPathEntry:
    """A remembered audio file path (a source recording or a reference sample)."""

    id: str
    path: Path
    label: str
    kind: AudioPathKind


@dataclass(frozen=True)
class FeedEntry:
    """A remembered podcast feed, addressed by its RSS/Atom *url*."""

    id: str
    url: str
    label: str


class ItemStore(Protocol[T]):
    """CRUD protocol for a collection of remembered items."""

    def list_items(self) -> list[T]:
        """Return every stored item."""
        ...

    def add_item(self, item: T) -> None:
        """Add a new item, raising if its id is already in use."""
        ...

    def get_item(self, item_id: str) -> T | None:
        """Return the item with *item_id*, or ``None`` if not found."""
        ...

    def remove_item(self, item_id: str) -> None:
        """Remove the item with *item_id*, if present."""
        ...


__all__ = ["AudioPathEntry", "AudioPathKind", "FeedEntry", "ItemStore"]
