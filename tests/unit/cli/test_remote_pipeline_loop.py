from __future__ import annotations

from pathlib import Path

from part_io.cli.remote._state import EpisodeState, PipelineState
from part_io.cli.remote_pipeline import _loop_work_counts


def test_loop_work_counts_reports_remaining_work() -> None:
    state = PipelineState()
    state.episodes["cuttable"] = EpisodeState(open_class="positive", close_class="positive")
    state.episodes["uncertain"] = EpisodeState(open_class="uncertain")
    state.episodes["done"] = EpisodeState(open_class="positive", close_class="positive", cut=True)

    n_undetected, n_uncertain, n_cuttable = _loop_work_counts(
        state,
        [Path("cuttable.mp3"), Path("uncertain.mp3"), Path("fresh.mp3")],
    )

    assert (n_undetected, n_uncertain, n_cuttable) == (1, 1, 1)


def test_loop_work_counts_reports_clean_state() -> None:
    state = PipelineState()

    assert _loop_work_counts(state, []) == (0, 0, 0)
