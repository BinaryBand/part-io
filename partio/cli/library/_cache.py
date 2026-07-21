"""The on-disk index: audio that has actually been materialized locally.

Written by :func:`partio.cli.library.ensure_local` when it downloads an episode
and by ``audio bootstrap`` when it writes a seed clip.  Nothing user-facing
reads it directly -- it exists so a picker can tell "already here" from "still
only in the feed", and so bootstrapped samples are offered at all.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from partio.adapters.store import JsonItemStore
from partio.core.ports import AudioPathEntry, AudioPathKind

DEFAULT_LIBRARY_PATH = Path("static") / "library.json"


def _to_dict(entry: AudioPathEntry) -> dict:
    return {
        "id": entry.id,
        "path": str(entry.path),
        "label": entry.label,
        "kind": entry.kind.value,
    }


def _from_dict(raw: dict) -> AudioPathEntry:
    return AudioPathEntry(
        id=raw["id"],
        path=Path(raw["path"]),
        label=raw["label"],
        kind=AudioPathKind(raw["kind"]),
    )


def cache_store() -> JsonItemStore[AudioPathEntry]:
    """Build the JSON-backed index of locally available audio."""
    return JsonItemStore(
        path=DEFAULT_LIBRARY_PATH,
        to_dict=_to_dict,
        from_dict=_from_dict,
        item_id=lambda entry: entry.id,
    )


def cached(kind: AudioPathKind | None = None) -> list[AudioPathEntry]:
    """Indexed audio paths of *kind* (all kinds when ``None``).

    A broken or missing index yields an empty list rather than an error: it is
    a cache, and losing it should cost a re-download, not a working prompt.
    """
    try:
        entries = cache_store().list_items()
    except (OSError, ValueError):
        return []
    return [entry for entry in entries if kind is None or entry.kind is kind]


def remember(path: Path, *, label: str, kind: AudioPathKind) -> None:
    """Index *path* as locally available, ignoring a path already indexed."""
    store = cache_store()
    if any(entry.path == path for entry in store.list_items()):
        return
    store.add_item(AudioPathEntry(id=uuid.uuid4().hex[:8], path=path, label=label, kind=kind))
