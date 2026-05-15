"""Process execution adapter that delegates to the shared process helper.

Callable ports implemented here are defined in part_io.models.ports.
"""

from __future__ import annotations

from part_io.utils.exec import resolve_executable, run_resolved

__all__ = ["resolve_executable", "run_resolved"]


