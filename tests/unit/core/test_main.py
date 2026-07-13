"""Tests for the core layer.

core is pure and deterministic, which makes it the natural home for
property-based tests (hypothesis). `clamp` is exercised with @given below as a
worked example; add more as core grows real logic.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from part_io.core.main import clamp

finite = st.floats(allow_nan=False, allow_infinity=False)


@given(value=finite, low=finite, high=finite)
def test_clamp_stays_within_bounds(value: float, low: float, high: float) -> None:
    if low > high:
        low, high = high, low
    result = clamp(value, low, high)
    assert low <= result <= high
