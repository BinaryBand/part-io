"""FastAPI application factory and root routes."""

from __future__ import annotations

from fastapi import FastAPI
from partio.core.audio_cut import plan_cut

from app.api.audio import router as audio_router
from app.api.library import router as library_router
from app.settings import PARTIO_ROOT

app = FastAPI(title="partio GUI server")
app.include_router(audio_router)
app.include_router(library_router)


@app.get("/health")
def health() -> dict[str, object]:
    """Report readiness, including a real call into partio's core.

    Calling ``plan_cut`` here (not just importing it) is a deliberate smoke
    test: it proves the editable cross-package dependency on ``partio``
    actually resolves and runs, not merely that FastAPI itself boots.
    """
    segments = plan_cut(
        opening_start=10.0,
        opening_end=15.0,
        closing_start=20.0,
        closing_end=25.0,
        total_seconds=30.0,
    )
    return {
        "status": "ok",
        "partio_root": str(PARTIO_ROOT),
        "plan_cut_smoke_test": [
            {"start_seconds": s.start_seconds, "end_seconds": s.end_seconds} for s in segments
        ],
    }
