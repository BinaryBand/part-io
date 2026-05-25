from __future__ import annotations

from pathlib import Path

from part_io.cli.remote._review import (
    ReviewItem,
    _collect_uncertain_candidates,
    _count_uncertain,
    _reclassify_all,
    _review_candidate,
    _run_quiz,
)
from part_io.cli.remote._state import PipelineState, Segment, _Match


def test_review_candidate_hides_compare_when_snippet_missing(tmp_path: Path, monkeypatch) -> None:
    state = PipelineState()
    ep = state.episode("ep001")
    ep.open_candidates = [_Match(score=0.9, start=5.0, end=6.0)]
    item = ReviewItem(stem="ep001", kind="open", candidate_idx=0, score=0.9)

    legends: list[str] = []

    monkeypatch.setattr("part_io.cli.remote._review._emit", lambda _msg: None)
    monkeypatch.setattr(
        "part_io.cli.remote._review._write_stderr",
        lambda text, end="\n", flush=False: legends.append(text),
    )
    monkeypatch.setattr("part_io.cli.remote._review._start_audio_segment", lambda *a, **k: None)
    monkeypatch.setattr("part_io.cli.remote._review._stop_audio", lambda _proc: None)
    monkeypatch.setattr("part_io.cli.remote._review._getch", lambda: "s")

    result = _review_candidate(
        state,
        item,
        snippets={},
        history=[],
        remote_dir=tmp_path,
    )

    assert result == "skipped"
    assert legends
    assert "[c]ompare" not in legends[0]


def test_review_candidate_shows_compare_when_snippet_available(tmp_path: Path, monkeypatch) -> None:
    state = PipelineState()
    ep = state.episode("ep001")
    ep.open_candidates = [_Match(score=0.9, start=5.0, end=6.0)]
    item = ReviewItem(stem="ep001", kind="open", candidate_idx=0, score=0.9)

    legends: list[str] = []

    monkeypatch.setattr("part_io.cli.remote._review._emit", lambda _msg: None)
    monkeypatch.setattr(
        "part_io.cli.remote._review._write_stderr",
        lambda text, end="\n", flush=False: legends.append(text),
    )
    monkeypatch.setattr("part_io.cli.remote._review._start_audio_segment", lambda *a, **k: None)
    monkeypatch.setattr("part_io.cli.remote._review._stop_audio", lambda _proc: None)
    monkeypatch.setattr("part_io.cli.remote._review._getch", lambda: "s")

    result = _review_candidate(
        state,
        item,
        snippets={"open": tmp_path / "open.mp3"},
        history=[],
        remote_dir=tmp_path,
    )

    assert result == "skipped"
    assert legends
    assert "[c]ompare" in legends[0]


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


def test_collect_uncertain_candidates_prioritizes_pairing_impact() -> None:
    state = PipelineState()

    # Low-score open that can form a valid pair when approved -> high information gain.
    high_impact = state.episode("high_impact")
    high_impact.open_candidates = [_Match(score=0.20, start=10.0, end=11.0)]
    high_impact.close_candidates = [_Match(score=0.40, start=20.0, end=21.0, label="positive")]

    # High-score open that still cannot pair (close is far out of bounds) -> low information gain.
    low_impact = state.episode("low_impact")
    low_impact.open_candidates = [_Match(score=0.99, start=10.0, end=11.0)]
    low_impact.close_candidates = [_Match(score=0.80, start=1000.0, end=1001.0, label="positive")]

    items = _collect_uncertain_candidates(state)
    open_items = [item for item in items if item.kind == "open"]

    assert [item.stem for item in open_items][:2] == ["high_impact", "low_impact"]


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
