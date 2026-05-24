from __future__ import annotations

from pathlib import Path

from part_io.cli.remote._state import EpisodeState, PipelineState, _Match
from part_io.cli.remote_pipeline import _loop_work_counts


def test_loop_work_counts_reports_remaining_work() -> None:
    state = PipelineState()
    state.episodes["cuttable"] = EpisodeState(
        candidates={
            "open": [_Match(score=0.9, start=10.0, end=20.0, label="positive")],
            "close": [_Match(score=0.9, start=40.0, end=50.0, label="positive")],
            "intro": [],
            "outro": [],
        }
    )
    state.episodes["uncertain"] = EpisodeState(
        candidates={
            "open": [_Match(score=0.5, start=12.0, end=22.0)],
            "close": [],
            "intro": [],
            "outro": [],
        }
    )
    state.episodes["done"] = EpisodeState(
        candidates={
            "open": [_Match(score=0.9, start=10.0, end=20.0, label="positive")],
            "close": [_Match(score=0.9, start=40.0, end=50.0, label="positive")],
            "intro": [],
            "outro": [],
        },
        cut=True,
    )

    n_undetected, n_uncertain, n_cuttable = _loop_work_counts(
        state,
        [Path("cuttable.mp3"), Path("uncertain.mp3"), Path("fresh.mp3")],
    )

    assert (n_undetected, n_uncertain, n_cuttable) == (1, 1, 1)


def test_loop_work_counts_reports_clean_state() -> None:
    state = PipelineState()

    assert _loop_work_counts(state, []) == (0, 0, 0)
