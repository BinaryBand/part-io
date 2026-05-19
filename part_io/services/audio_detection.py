"""Shared audio sample detection service functions.

This module centralizes detection behavior used by CLI wrappers and orchestration
flows so match sorting, limiting, and output shaping stay consistent.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol


class MatchLike(Protocol):
    """Structural match type used by detection services."""

    start_seconds: float
    end_seconds: float
    score: float


DetectionKind = Literal["open", "close", "intro"]


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
    open_floor: float
    close_floor: float


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
    """Build detection jobs for open/close and optional intro samples."""
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
