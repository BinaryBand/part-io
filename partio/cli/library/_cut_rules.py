"""Where saved cut rules -- named opening/closing jingle pairs -- are persisted."""

from __future__ import annotations

from pathlib import Path

from partio.adapters.store import JsonItemStore
from partio.core.ports import CutRuleEntry

DEFAULT_CUT_RULES_PATH = Path("static") / "cut_rules.json"


def _to_dict(entry: CutRuleEntry) -> dict:
    return {
        "id": entry.id,
        "label": entry.label,
        "opening_path": str(entry.opening_path),
        "closing_path": str(entry.closing_path),
    }


def _from_dict(raw: dict) -> CutRuleEntry:
    return CutRuleEntry(
        id=raw["id"],
        label=raw["label"],
        opening_path=Path(raw["opening_path"]),
        closing_path=Path(raw["closing_path"]),
    )


def cut_rule_store() -> JsonItemStore[CutRuleEntry]:
    """Build the JSON-backed store for saved cut rules."""
    return JsonItemStore(
        path=DEFAULT_CUT_RULES_PATH,
        to_dict=_to_dict,
        from_dict=_from_dict,
        item_id=lambda entry: entry.id,
    )


def cut_rules() -> list[CutRuleEntry]:
    """Every saved cut rule, or an empty list if the store is unreadable.

    Degrading to "no rules" rather than raising keeps a corrupt store from
    blocking `audio cut` from falling back to manual jingle entry.
    """
    try:
        return cut_rule_store().list_items()
    except (OSError, ValueError):
        return []
