"""Optional refine seam for remote detection flows.

The default behavior is no-op baseline matching. Refinement is only applied when
explicitly enabled via environment and when the optional implementation module
is available.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)
_ENV_ENABLE = "PART_IO_ENABLE_REFINE_PLUGIN"


def refine_plugin_enabled() -> bool:
    raw = os.getenv(_ENV_ENABLE, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _load_refine_impl() -> Callable[..., list[Any]] | None:
    """Return optional refine implementation, or None when unavailable."""
    try:
        from part_io.adapters.audio.refine_impl import refine_matches
    except ModuleNotFoundError:
        _LOG.warning("Refine module not found; using baseline detection.")
        return None
    except Exception:
        _LOG.exception("Failed importing refine module; using baseline detection.")
        return None
    return refine_matches


def apply_optional_refine(
    *,
    matches: Sequence[Any],
    source_path: Path,
    sample_path: Path,
) -> list[Any]:
    """Apply optional refine implementation when enabled, else return baseline."""
    baseline = list(matches)
    if not refine_plugin_enabled():
        return baseline

    refine_matches = _load_refine_impl()
    if refine_matches is None:
        return baseline

    try:
        return list(
            refine_matches(matches=baseline, source_path=source_path, sample_path=sample_path)
        )
    except Exception:
        _LOG.exception("Refine execution failed; using baseline detection.")
        return baseline


__all__ = ["apply_optional_refine", "refine_plugin_enabled"]
