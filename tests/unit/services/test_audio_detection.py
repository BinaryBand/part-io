"""Unit tests for shared audio detection services."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch
from part_io.services import audio_detection


@dataclass
class _FakeCandidate:
    score: float
    start: float
    end: float
    label: str | None = None


@dataclass
class _FakeEpisodeState:
    source: str = ""
    open_candidates: list[_FakeCandidate] = field(default_factory=list)
    close_candidates: list[_FakeCandidate] = field(default_factory=list)
    intro_candidates: list[_FakeCandidate] = field(default_factory=list)
    outro_candidates: list[_FakeCandidate] = field(default_factory=list)


def _fake_factory(match: audio_detection.MatchLike) -> _FakeCandidate:
    return _FakeCandidate(score=match.score, start=match.start_seconds, end=match.end_seconds)


def test_detect_top_matches_sorts_and_limits(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"src")
    sample.write_bytes(b"smp")

    def detector(**_kwargs):
        return [
            AudioMatch(start_seconds=1.0, end_seconds=2.0, duration_seconds=1.0, score=0.3),
            AudioMatch(start_seconds=3.0, end_seconds=4.0, duration_seconds=1.0, score=0.8),
            AudioMatch(start_seconds=5.0, end_seconds=6.0, duration_seconds=1.0, score=0.5),
        ]

    matches = audio_detection.detect_top_matches(
        detector=detector,
        source_path=source,
        sample_path=sample,
        score_threshold=0.1,
        step_seconds=0.1,
        max_matches=2,
    )

    assert [m.score for m in matches] == [0.8, 0.5]


def test_matches_to_cli_rows_rounds_and_indexes() -> None:
    matches = [
        AudioMatch(
            start_seconds=1.23456,
            end_seconds=2.34567,
            duration_seconds=1.11111,
            score=0.987654321,
        )
    ]

    rows = audio_detection.matches_to_cli_rows(matches)

    assert rows == [{"index": 1, "score": 0.987654, "start": 1.235, "end": 2.346}]


def test_run_detection_batch_jobs_collects_results_and_errors(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    sample_ok = tmp_path / "sample_ok.mp3"
    sample_bad = tmp_path / "sample_bad.mp3"
    source.write_bytes(b"src")
    sample_ok.write_bytes(b"ok")
    sample_bad.write_bytes(b"bad")

    def fake_detect_top_matches(**kwargs):
        if kwargs["sample_path"].name == "sample_bad.mp3":
            raise ValueError("boom")
        return [AudioMatch(start_seconds=2.0, end_seconds=3.0, duration_seconds=1.0, score=0.9)]

    monkeypatch.setattr(audio_detection, "detect_top_matches", fake_detect_top_matches)

    jobs = [
        audio_detection.DetectionBatchJob(
            stem="ep1",
            source_path=source,
            sample_path=sample_ok,
            kind="open",
        ),
        audio_detection.DetectionBatchJob(
            stem="ep1",
            source_path=source,
            sample_path=sample_bad,
            kind="close",
        ),
    ]

    results = audio_detection.run_detection_batch_jobs(
        jobs,
        detector=lambda **kwargs: fake_detect_top_matches(**kwargs),
        step_seconds=0.1,
        max_matches=3,
        workers=1,
    )

    assert len(results) == 2
    assert results[0].stem == "ep1"
    assert results[0].kind == "open"
    assert results[0].error is None
    assert results[0].matches[0].score == 0.9

    assert results[1].stem == "ep1"
    assert results[1].kind == "close"
    assert results[1].matches == []
    assert results[1].error == "boom"


def test_build_detection_batch_jobs_includes_intro_when_present(tmp_path: Path) -> None:
    episode_a = tmp_path / "ep_a.mp3"
    episode_b = tmp_path / "ep_b.mp3"
    open_sample = tmp_path / "open.mp3"
    close_sample = tmp_path / "close.mp3"
    intro_sample = tmp_path / "intro.mp3"

    for path in (episode_a, episode_b, open_sample, close_sample, intro_sample):
        path.write_bytes(b"x")

    request = audio_detection.DetectionBatchRequest(
        episodes=[episode_a, episode_b],
        snippets={"open": open_sample, "close": close_sample, "intro": intro_sample},
    )

    jobs = audio_detection.build_detection_batch_jobs(request)

    assert len(jobs) == 6
    assert sum(1 for job in jobs if job.kind == "open") == 2
    assert sum(1 for job in jobs if job.kind == "close") == 2
    assert sum(1 for job in jobs if job.kind == "intro") == 2


def test_build_detection_batch_jobs_includes_optional_outro_when_present(tmp_path: Path) -> None:
    episode_a = tmp_path / "ep_a.mp3"
    open_sample = tmp_path / "open.mp3"
    close_sample = tmp_path / "close.mp3"
    outro_sample = tmp_path / "outro.mp3"
    for path in (episode_a, open_sample, close_sample, outro_sample):
        path.write_bytes(b"x")

    request = audio_detection.DetectionBatchRequest(
        episodes=[episode_a],
        snippets={"open": open_sample, "close": close_sample, "outro": outro_sample},
    )

    jobs = audio_detection.build_detection_batch_jobs(request)

    assert len(jobs) == 3
    assert sum(1 for job in jobs if job.kind == "open") == 1
    assert sum(1 for job in jobs if job.kind == "close") == 1
    assert sum(1 for job in jobs if job.kind == "outro") == 1


def test_run_detection_batch_returns_jobs_and_results(monkeypatch, tmp_path: Path) -> None:
    episode = tmp_path / "ep_a.mp3"
    open_sample = tmp_path / "open.mp3"
    close_sample = tmp_path / "close.mp3"
    for path in (episode, open_sample, close_sample):
        path.write_bytes(b"x")

    monkeypatch.setattr(
        audio_detection,
        "run_detection_batch_jobs",
        lambda jobs, **_kwargs: [
            audio_detection.DetectionBatchResult(
                stem=jobs[0].stem,
                source_path=jobs[0].source_path,
                sample_path=jobs[0].sample_path,
                kind=jobs[0].kind,
                matches=[],
            )
        ],
    )

    request = audio_detection.DetectionBatchRequest(
        episodes=[episode],
        snippets={"open": open_sample, "close": close_sample},
    )

    jobs, results = audio_detection.run_detection_batch(
        request,
        detector=lambda **_kwargs: [],
        step_seconds=0.1,
        max_matches=3,
        workers=1,
    )

    assert len(jobs) == 2
    assert len(results) == 1
    assert results[0].kind in ("open", "close")


def test_apply_batch_result_to_episode_sets_open_fields(tmp_path: Path) -> None:
    source = tmp_path / "ep.mp3"
    sample = tmp_path / "open.mp3"
    source.write_bytes(b"x")
    sample.write_bytes(b"x")
    result = audio_detection.DetectionBatchResult(
        stem="ep",
        source_path=source,
        sample_path=sample,
        kind="open",
        matches=[AudioMatch(start_seconds=1.0, end_seconds=2.0, duration_seconds=1.0, score=0.9)],
    )
    episode = _FakeEpisodeState()

    score_str, error_msg = audio_detection.apply_batch_result_to_episode(
        result,
        episode,
        match_factory=_fake_factory,
    )

    assert error_msg is None
    assert score_str == "0.9000"
    assert len(episode.open_candidates) == 1
    assert episode.open_candidates[0].score == 0.9
    assert episode.open_candidates[0].start == 1.0
    assert episode.open_candidates[0].end == 2.0


def test_apply_batch_result_to_episode_handles_close_no_matches_with_error(tmp_path: Path) -> None:
    source = tmp_path / "ep.mp3"
    sample = tmp_path / "close.mp3"
    source.write_bytes(b"x")
    sample.write_bytes(b"x")
    result = audio_detection.DetectionBatchResult(
        stem="ep",
        source_path=source,
        sample_path=sample,
        kind="close",
        matches=[],
        error="boom",
    )
    episode = _FakeEpisodeState()

    score_str, error_msg = audio_detection.apply_batch_result_to_episode(
        result,
        episode,
        match_factory=_fake_factory,
    )

    assert score_str == "none"
    assert "WARNING: detection failed" in (error_msg or "")
    assert episode.close_candidates == []


def test_apply_batch_result_to_episode_sets_intro_fields(tmp_path: Path) -> None:
    source = tmp_path / "ep.mp3"
    sample = tmp_path / "intro.mp3"
    source.write_bytes(b"x")
    sample.write_bytes(b"x")
    result = audio_detection.DetectionBatchResult(
        stem="ep",
        source_path=source,
        sample_path=sample,
        kind="intro",
        matches=[AudioMatch(start_seconds=4.0, end_seconds=5.0, duration_seconds=1.0, score=1.1)],
    )
    episode = _FakeEpisodeState()

    score_str, _ = audio_detection.apply_batch_result_to_episode(
        result,
        episode,
        match_factory=_fake_factory,
    )

    assert score_str == "1.1000"
    assert len(episode.intro_candidates) == 1
    assert episode.intro_candidates[0].score == 1.1


def test_apply_batch_result_to_episode_sets_outro_fields(tmp_path: Path) -> None:
    source = tmp_path / "ep.mp3"
    sample = tmp_path / "outro.mp3"
    source.write_bytes(b"x")
    sample.write_bytes(b"x")
    result = audio_detection.DetectionBatchResult(
        stem="ep",
        source_path=source,
        sample_path=sample,
        kind="outro",
        matches=[AudioMatch(start_seconds=94.0, end_seconds=95.0, duration_seconds=1.0, score=1.2)],
    )
    episode = _FakeEpisodeState()

    score_str, _ = audio_detection.apply_batch_result_to_episode(
        result,
        episode,
        match_factory=_fake_factory,
    )

    assert score_str == "1.2000"
    assert len(episode.outro_candidates) == 1
    assert episode.outro_candidates[0].score == 1.2


def test_filter_matches_by_position_limits_intro_to_first_quarter() -> None:
    matches = [
        AudioMatch(start_seconds=20.0, end_seconds=21.0, duration_seconds=1.0, score=0.8),
        AudioMatch(start_seconds=30.0, end_seconds=31.0, duration_seconds=1.0, score=0.9),
        AudioMatch(start_seconds=31.0, end_seconds=32.0, duration_seconds=1.0, score=1.0),
    ]

    filtered = audio_detection.filter_matches_by_position(
        matches,
        kind="intro",
        source_duration_seconds=200.0,
    )

    assert [m.start_seconds for m in filtered] == [31.0]


def test_filter_matches_by_position_limits_outro_to_last_quarter() -> None:
    matches = [
        AudioMatch(start_seconds=74.0, end_seconds=75.0, duration_seconds=1.0, score=0.8),
        AudioMatch(start_seconds=75.0, end_seconds=76.0, duration_seconds=1.0, score=0.9),
    ]

    filtered = audio_detection.filter_matches_by_position(
        matches,
        kind="outro",
        source_duration_seconds=100.0,
    )

    assert [m.start_seconds for m in filtered] == [75.0]
