from __future__ import annotations

import pytest

from part_io.services.cut_planning import build_cut_plan


def test_build_cut_plan_sorts_and_builds_spans() -> None:
    plan = build_cut_plan([(20.0, 30.0), (5.0, 10.0)])

    assert plan.cuts == [(5.0, 10.0), (20.0, 30.0)]
    assert plan.spans == [(0.0, 5.0), (10.0, 20.0), (30.0, None)]


def test_build_cut_plan_adds_intro_trim_first() -> None:
    plan = build_cut_plan([(10.0, 12.0)], intro_trim=3.5)

    assert plan.spans == [(3.5, 10.0), (12.0, None)]


def test_build_cut_plan_allows_empty_cuts() -> None:
    plan = build_cut_plan([])

    assert plan.cuts == []
    assert plan.spans == [(0.0, None)]


def test_build_cut_plan_rejects_overlaps() -> None:
    with pytest.raises(ValueError, match="Overlapping ad segments"):
        build_cut_plan([(5.0, 12.0), (10.0, 14.0)])
