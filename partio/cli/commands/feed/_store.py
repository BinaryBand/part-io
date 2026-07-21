"""Shared JSON-backed store wiring for the feed command group."""

from __future__ import annotations

from pathlib import Path

from partio.adapters.store import JsonItemStore
from partio.core.ports import FeedEntry

DEFAULT_FEEDS_PATH = Path("static") / "feeds.json"


def _to_dict(entry: FeedEntry) -> dict:
    return {"id": entry.id, "url": entry.url, "label": entry.label}


def _from_dict(raw: dict) -> FeedEntry:
    return FeedEntry(id=raw["id"], url=raw["url"], label=raw["label"])


def default_store() -> JsonItemStore[FeedEntry]:
    """Build the default JSON-backed store for remembered feeds."""
    return JsonItemStore(
        path=DEFAULT_FEEDS_PATH,
        to_dict=_to_dict,
        from_dict=_from_dict,
        item_id=lambda entry: entry.id,
    )
