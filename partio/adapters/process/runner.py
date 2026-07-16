"""Process execution adapter that delegates to the shared process helper.

Callable ports implemented here are defined in partio.core.ports.
"""

from __future__ import annotations

from partio.utils.exec import resolve_executable, run_resolved

__all__ = ["resolve_executable", "run_resolved"]
