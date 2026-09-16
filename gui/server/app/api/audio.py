"""Range-capable audio file serving, used by every screen's player.

The GUI never spawns ``ffplay`` for playback (that's how the CLI's terminal
auditor works, but per-scrub process-spawn latency is a poor fit for a
responsive player) -- instead the client's audio player streams straight from
this endpoint over HTTP range requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.settings import AUDIO_SERVE_ROOTS, PARTIO_ROOT

router = APIRouter(prefix="/api/audio", tags=["audio"])


def _is_allowed(path: Path) -> bool:
    """Whether *path* falls under one of the library's own audio directories."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(
        resolved.is_relative_to((PARTIO_ROOT / root).resolve()) for root in AUDIO_SERVE_ROOTS
    )


@router.get("/file")
def get_audio_file(
    path: Annotated[str, Query(description="Absolute path to an audio file.")],
) -> FileResponse:
    """Serve an audio file with HTTP Range support.

    Not a general-purpose local file server: *path* must resolve under one of
    the library's own directories (`static/downloads`, `static/jingles`,
    `downloads/review`), which is all a localhost-only tool needs -- there's
    no user account to protect against, only a guard against this endpoint
    being usable as an arbitrary local-file server by some other process.
    """
    resolved = Path(path)
    if not _is_allowed(resolved):
        raise HTTPException(status_code=403, detail="Path is outside the served audio roots.")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(resolved, media_type="audio/mpeg")
