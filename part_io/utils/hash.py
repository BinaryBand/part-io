"""File hashing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

_PARTIAL_HASH_BYTES = 65536  # 64 KB


def partial_file_hash(path: Path) -> str:
    """Return a hex SHA-256 digest of the first 64 KB of *path* plus its size.

    Including the file size ensures that a file cut at the end (same header,
    smaller content) always produces a different hash from the original.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(_PARTIAL_HASH_BYTES))
    h.update(path.stat().st_size.to_bytes(8, "little"))
    return h.hexdigest()


__all__ = ["partial_file_hash"]
