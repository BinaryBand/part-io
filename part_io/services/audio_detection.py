"""Shared audio sample detection service functions.

This module centralizes detection behavior used by CLI wrappers and orchestration
flows so match sorting, limiting, and output shaping stay consistent.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol


class MatchLike(Protocol):
    """Structural match type used by detection services."""

    start_seconds: float
    end_seconds: float
    score: float


class EpisodeStateLike(Protocol):
    """Structural episode state shape needed for detection mutations."""

    source: str
    open_candidates: list[Any]
    open_class: str
    close_candidates: list[Any]
    close_class: str
    intro_candidates: list[Any]
    intro_class: str
    outro_candidates: list[Any]
    outro_class: str


DetectionKind = Literal["open", "close", "intro", "outro"]


@dataclass(frozen=True)
class DetectionBatchJob:
    """One detection task for a specific episode stem and sample type."""

    stem: str
    source_path: Path
    sample_path: Path
    kind: DetectionKind
    floor: float


@dataclass(frozen=True)
class DetectionBatchResult:
    """Detection output for a single batch job."""

    stem: str
    source_path: Path
    sample_path: Path
    kind: DetectionKind
    matches: Sequence[MatchLike]
    error: str | None = None


@dataclass(frozen=True)
class DetectionBatchRequest:
    """Inputs needed to build and run a detection batch for episodes."""

    episodes: list[Path]
    open_sample: Path
    close_sample: Path
    intro_sample: Path | None
    outro_sample: Path | None
    open_floor: float
    close_floor: float


def filter_matches_by_position(
    matches: Sequence[MatchLike],
    *,
    kind: DetectionKind,
    source_duration_seconds: float,
) -> list[MatchLike]:
    """Keep only positionally valid intro/outro matches.

    Intro candidates must start after the first 30 seconds and within the
    first 25% of the source duration.
    Outro candidates must start in the last 25% of the source duration.
    Open/close matches are returned unchanged.
    """
    if source_duration_seconds <= 0:
        return list(matches)
    if kind == "intro":
        max_intro_start = source_duration_seconds * 0.25
        min_intro_start = 30.0
        return [
            match
            for match in matches
            if float(match.start_seconds) > min_intro_start
            and float(match.start_seconds) <= max_intro_start
        ]
    if kind == "outro":
        min_outro_start = source_duration_seconds * 0.75
        return [match for match in matches if float(match.start_seconds) >= min_outro_start]
    return list(matches)


def detect_top_matches(
    *,
    detector: Callable[..., Sequence[MatchLike]],
    source_path: Path,
    sample_path: Path,
    score_threshold: float,
    z_threshold: float | None,
    step_seconds: float,
    max_matches: int,
) -> list[MatchLike]:
    """Return top sample matches sorted by score descending.

    When ``max_matches`` is zero or negative, all matches are returned.
    """
    matches = detector(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=score_threshold,
        z_threshold=z_threshold,
        step_seconds=step_seconds,
    )

    ranked = sorted(matches, key=lambda m: m.score, reverse=True)
    if max_matches > 0:
        ranked = ranked[:max_matches]
    return ranked


def matches_to_cli_rows(matches: Sequence[MatchLike]) -> list[dict[str, float | int]]:
    """Convert matches to the stable JSON row shape used by detect CLI."""
    return [
        {
            "index": index,
            "score": round(float(match.score), 6),
            "start": round(float(match.start_seconds), 3),
            "end": round(float(match.end_seconds), 3),
        }
        for index, match in enumerate(matches, 1)
    ]


def run_detection_batch_jobs(
    jobs: list[DetectionBatchJob],
    *,
    detector: Callable[..., Sequence[MatchLike]],
    z_threshold: float | None,
    step_seconds: float,
    max_matches: int,
    workers: int,
) -> list[DetectionBatchResult]:
    """Execute detection jobs concurrently and return completion-ordered results."""
    results: list[DetectionBatchResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                detect_top_matches,
                detector=detector,
                source_path=job.source_path,
                sample_path=job.sample_path,
                score_threshold=job.floor,
                z_threshold=z_threshold,
                step_seconds=step_seconds,
                max_matches=max_matches,
            ): job
            for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                matches = list(future.result())
                results.append(
                    DetectionBatchResult(
                        stem=job.stem,
                        source_path=job.source_path,
                        sample_path=job.sample_path,
                        kind=job.kind,
                        matches=matches,
                    )
                )
            except (FileNotFoundError, ValueError) as exc:
                results.append(
                    DetectionBatchResult(
                        stem=job.stem,
                        source_path=job.source_path,
                        sample_path=job.sample_path,
                        kind=job.kind,
                        matches=[],
                        error=str(exc),
                    )
                )
    return results


def build_detection_batch_jobs(request: DetectionBatchRequest) -> list[DetectionBatchJob]:
    """Build detection jobs for open/close and optional intro/outro samples."""
    jobs = [
        DetectionBatchJob(
            stem=episode.stem,
            source_path=episode,
            sample_path=request.open_sample,
            kind="open",
            floor=request.open_floor,
        )
        for episode in request.episodes
    ] + [
        DetectionBatchJob(
            stem=episode.stem,
            source_path=episode,
            sample_path=request.close_sample,
            kind="close",
            floor=request.close_floor,
        )
        for episode in request.episodes
    ]

    if request.intro_sample is not None and request.intro_sample.exists():
        jobs += [
            DetectionBatchJob(
                stem=episode.stem,
                source_path=episode,
                sample_path=request.intro_sample,
                kind="intro",
                floor=0.0,
            )
            for episode in request.episodes
        ]

    if request.outro_sample is not None and request.outro_sample.exists():
        jobs += [
            DetectionBatchJob(
                stem=episode.stem,
                source_path=episode,
                sample_path=request.outro_sample,
                kind="outro",
                floor=0.0,
            )
            for episode in request.episodes
        ]

    return jobs


def run_detection_batch(
    request: DetectionBatchRequest,
    *,
    detector: Callable[..., Sequence[MatchLike]],
    z_threshold: float | None,
    step_seconds: float,
    max_matches: int,
    workers: int,
) -> tuple[list[DetectionBatchJob], list[DetectionBatchResult]]:
    """Build and execute a full detection batch for episodes."""
    jobs = build_detection_batch_jobs(request)
    results = run_detection_batch_jobs(
        jobs,
        detector=detector,
        z_threshold=z_threshold,
        step_seconds=step_seconds,
        max_matches=max_matches,
        workers=workers,
    )
    return jobs, results


def apply_batch_result_to_episode(
    result: DetectionBatchResult,
    episode_state: EpisodeStateLike,
    *,
    match_factory: Callable[[MatchLike], Any],
    uncertain_label: str,
    undetected_label: str,
) -> tuple[str, str | None]:
    """Apply one detection result to episode state and return (score_str, error_msg)."""
    episode_state.source = str(result.source_path)
    matches = [match_factory(match) for match in result.matches]

    if result.kind == "open":
        episode_state.open_candidates = matches
        episode_state.open_class = uncertain_label if matches else undetected_label
    elif result.kind == "close":
        episode_state.close_candidates = matches
        episode_state.close_class = uncertain_label if matches else undetected_label
    elif result.kind == "intro":
        episode_state.intro_candidates = matches
        episode_state.intro_class = uncertain_label if matches else undetected_label
    else:  # outro
        episode_state.outro_candidates = matches
        episode_state.outro_class = uncertain_label if matches else undetected_label

    score_str = f"{result.matches[0].score:.4f}" if result.matches else "none"
    error_msg = None
    if result.error:
        error_msg = (
            "  WARNING: detection failed for "
            f"{result.source_path.name} ({result.sample_path.name}): {result.error}"
        )
    return score_str, error_msg
