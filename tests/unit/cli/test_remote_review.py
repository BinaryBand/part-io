from __future__ import annotations

from pathlib import Path

from part_io.cli.remote._review import (
    ReviewItem,
    _collect_uncertain_candidates,
    _count_uncertain,
    _reclassify_all,
    _run_quiz,
)
from part_io.cli.remote._state import PipelineState, Segment, _Match


def test_count_uncertain_counts_all_audio_kinds() -> None:
    state = PipelineState()
    ep = state.episode("ep001")
    ep.open_candidates = [_Match(score=0.8, start=1.0, end=2.0)]
    ep.close_candidates = [_Match(score=0.7, start=3.0, end=4.0)]
    ep.intro_candidates = [_Match(score=0.9, start=5.0, end=6.0, label="positive")]

    assert _count_uncertain(state) == 2


def test_collect_uncertain_candidates_returns_sorted_candidates() -> None:
    state = PipelineState()

    ep1 = state.episode("ep1")
    ep1.open_candidates = [_Match(score=0.81, start=1.0, end=2.0)]

    ep2 = state.episode("ep2")
    ep2.open_candidates = [
        _Match(score=0.95, start=1.0, end=2.0),
        _Match(score=0.88, start=3.0, end=4.0),
    ]

    items = _collect_uncertain_candidates(state)

    assert [(it.stem, it.kind, it.candidate_idx, round(it.score, 2)) for it in items] == [
        ("ep2", "open", 0, 0.95),
        ("ep1", "open", 0, 0.81),
        ("ep2", "open", 1, 0.88),
    ]


def test_reclassify_all_promotes_uncertain_when_score_clears_threshold() -> None:
    state = PipelineState()

    state.open_target.positives.extend(
        [
            Segment(stem="a", start=0.0, end=1.0, score=0.9),
            Segment(stem="b", start=0.0, end=1.0, score=0.9),
        ]
    )

    ep = state.episode("ep")
    ep.open_candidates = [_Match(score=0.98, start=2.0, end=3.0)]

    _reclassify_all(state)

    assert state.episode("ep").class_for("open") == "positive"


def test_run_quiz_approve_counts_and_saves(tmp_path: Path, monkeypatch) -> None:
    state = PipelineState()
    ep = state.episode("ep001")
    ep.open_candidates = [_Match(score=0.9, start=5.0, end=6.0)]

    items = [ReviewItem(stem="ep001", kind="open", candidate_idx=0, score=0.9)]

    calls: dict[str, int] = {"save": 0}

    def fake_review_candidate(*args, **kwargs):
        _ = args
        _ = kwargs
        return "approved"

    def fake_save(path: Path) -> None:
        _ = path
        calls["save"] += 1

    monkeypatch.setattr("part_io.cli.remote._review._review_candidate", fake_review_candidate)
    monkeypatch.setattr(state, "save", fake_save)

    decisions, interrupted, skipped = _run_quiz(
        state,
        items,
        snippets={
            "open": tmp_path / "open.mp3",
            "close": tmp_path / "close.mp3",
            "intro": tmp_path / "intro.mp3",
            "outro": tmp_path / "outro.mp3",
        },
        state_path=tmp_path / "state.toml",
        remote_dir=tmp_path,
    )

    assert decisions == 1
    assert interrupted is False
    assert skipped == set()
    assert calls["save"] == 1


def test_run_quiz_keyboard_interrupt_returns_interrupted(tmp_path: Path, monkeypatch) -> None:
    state = PipelineState()
    ep = state.episode("ep001")
    ep.open_candidates = [_Match(score=0.9, start=5.0, end=6.0)]

    items = [ReviewItem(stem="ep001", kind="open", candidate_idx=0, score=0.9)]

    def fake_review_candidate(*args, **kwargs):
        _ = args
        _ = kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr("part_io.cli.remote._review._review_candidate", fake_review_candidate)

    decisions, interrupted, skipped = _run_quiz(
        state,
        items,
        snippets={
            "open": tmp_path / "open.mp3",
            "close": tmp_path / "close.mp3",
            "intro": tmp_path / "intro.mp3",
            "outro": tmp_path / "outro.mp3",
        },
        state_path=tmp_path / "state.toml",
        remote_dir=tmp_path,
    )

    assert decisions == 0
    assert interrupted is True
    assert skipped == set()


def test_run_quiz_rejected_uncertain_candidate_marked_skipped(tmp_path: Path, monkeypatch) -> None:
    state = PipelineState()
    ep = state.episode("ep001")
    ep.open_candidates = [_Match(score=0.9, start=5.0, end=6.0)]

    item = ReviewItem(stem="ep001", kind="open", candidate_idx=0, score=0.9)

    def fake_review_candidate(*args, **kwargs):
        _ = args
        _ = kwargs
        return "rejected"

    monkeypatch.setattr("part_io.cli.remote._review._review_candidate", fake_review_candidate)
    monkeypatch.setattr(
        "part_io.cli.remote._review._collect_uncertain_candidates", lambda _state: [item]
    )

    decisions, interrupted, skipped = _run_quiz(
        state,
        [item],
        snippets={
            "open": tmp_path / "open.mp3",
            "close": tmp_path / "close.mp3",
            "intro": tmp_path / "intro.mp3",
            "outro": tmp_path / "outro.mp3",
        },
        state_path=tmp_path / "state.toml",
        remote_dir=tmp_path,
    )

    assert decisions == 1
    assert interrupted is False
    assert skipped == {("ep001", "open", 0)}
