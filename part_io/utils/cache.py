"""Utility helpers for on-disk numpy-backed caches."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from part_io.utils.hash import partial_file_hash


def load_npz_profile(source_path: Path, cache_dir: Path) -> np.ndarray | None:
    """Load a numpy `.npz` profile for *source_path* from *cache_dir*.

    The cache key is a hash of the first 64 KB of *source_path*, so entries
    survive renames and are never confused with a cut/modified version of the
    same file. Returns the stored profile or ``None`` on miss or error.
    """
    cache_path = cache_dir / f"{partial_file_hash(source_path)}.npz"
    if not cache_path.exists():
        return None
    try:
        return np.load(cache_path)["profile"]
    except Exception as exc:  # pragma: no cover - benign cache read failures
        logging.debug("Failed to load cached profile '%s': %s", cache_path, exc, exc_info=True)
    return None


def save_npz_profile(source_path: Path, profile: np.ndarray, cache_dir: Path) -> None:
    """Persist *profile* to *cache_dir* keyed by a hash of *source_path*.

    Failures are debug-logged and otherwise ignored to keep the cache from
    blocking detection on network mounts or read-only filesystems.
    """
    cache_path = cache_dir / f"{partial_file_hash(source_path)}.npz"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, profile=profile)
    except Exception as exc:  # pragma: no cover - benign cache write failures
        logging.debug("Failed to save cached profile '%s': %s", cache_path, exc, exc_info=True)
