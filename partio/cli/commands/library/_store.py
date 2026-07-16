"""Shared JSON-backed store wiring for the library command group."""

from __future__ import annotations

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


def default_store(path: Path | None = None) -> JsonItemStore[AudioPathEntry]:
    """Build the default JSON-backed store for remembered audio paths."""
    return JsonItemStore(
        path=path if path is not None else DEFAULT_LIBRARY_PATH,
        to_dict=_to_dict,
        from_dict=_from_dict,
        item_id=lambda entry: entry.id,
    )
