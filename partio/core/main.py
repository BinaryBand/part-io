"""core.main: pure business logic.

No I/O -- deterministic functions, which makes this the natural home for
property-based tests (hypothesis). `clamp` below is a worked example that the
mirror test exercises with @given -- replace or delete it.
"""

from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    """Constrain value to the inclusive [low, high] range -- worked example."""
    return max(low, min(value, high))
