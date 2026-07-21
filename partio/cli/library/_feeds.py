"""Where remembered podcast feeds are persisted."""

from __future__ import annotations

from pathlib import Path

from partio.adapters.store import JsonItemStore
from partio.core.ports import FeedEntry

DEFAULT_FEEDS_PATH = Path("static") / "feeds.json"


def _to_dict(entry: FeedEntry) -> dict:
    return {"id": entry.id, "url": entry.url, "label": entry.label}


def _from_dict(raw: dict) -> FeedEntry:
    return FeedEntry(id=raw["id"], url=raw["url"], label=raw["label"])


def feed_store() -> JsonItemStore[FeedEntry]:
    """Build the JSON-backed store for remembered feeds."""
    return JsonItemStore(
        path=DEFAULT_FEEDS_PATH,
        to_dict=_to_dict,
        from_dict=_from_dict,
        item_id=lambda entry: entry.id,
    )


def feeds() -> list[FeedEntry]:
    """Every remembered feed, or an empty list if the store is unreadable.

    Degrading to "no feeds" rather than raising keeps a corrupt store from
    blocking a prompt that could still offer what is already on disk.
    """
    try:
        return feed_store().list_items()
    except (OSError, ValueError):
        return []
