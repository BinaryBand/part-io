"""Process-wide configuration.

``partio.cli.library``'s storage paths (``static/feeds.json``,
``static/downloads/``, ...) are resolved relative to the current working
directory, not to the ``partio`` package's own location. This module resolves
and validates ``PARTIO_ROOT`` once at import time so every other module can
assume the working directory is already correct.
"""

from __future__ import annotations

import os
from pathlib import Path


class InvalidPartioRootError(RuntimeError):
    """Raised when ``PARTIO_ROOT`` does not point at a real partio checkout."""


def resolve_partio_root() -> Path:
    """Resolve ``PARTIO_ROOT`` (env var, default: the current directory).

    Also ``chdir``s the process into it, since ``partio.cli.library`` builds
    its storage paths as bare relative ``Path("static")``/``Path("downloads")``
    -- setting the working directory is what actually makes those resolve
    against the real library instead of wherever this process happened to
    start.

    Raises:
        InvalidPartioRootError: When the resolved path has no ``partio``
            package directory, so a misconfigured working directory fails
            loudly instead of silently reading/writing an empty ``static/``.
    """
    raw = os.environ.get("PARTIO_ROOT", ".")
    root = Path(raw).expanduser().resolve()
    if not (root / "partio" / "__init__.py").is_file():
        raise InvalidPartioRootError(
            f"PARTIO_ROOT={raw!r} (resolved to {root}) has no partio/__init__.py. "
            "Set PARTIO_ROOT to the partio repo root, or launch this process "
            "with that directory as its current working directory."
        )
    os.chdir(root)
    return root


PARTIO_ROOT = resolve_partio_root()

# Audio files may only be served from these roots (relative to PARTIO_ROOT),
# matching partio's own library/output locations.
AUDIO_SERVE_ROOTS = ("static/downloads", "static/jingles", "downloads/review")

HOST = os.environ.get("PARTIO_GUI_HOST", "127.0.0.1")
PORT = int(os.environ.get("PARTIO_GUI_PORT", "8756"))
