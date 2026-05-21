from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

from part_io.adapters.audio.matcher import (
    AudioMatch,
    build_consensus_profile,
    find_audio_sample_matches,
    find_audio_sample_matches_from_profile,
    warm_source_profile,
)
from part_io.adapters.audio.refine_impl import refine_matches as _refine_matches
from part_io.adapters.process.runner import run_resolved
from part_io.cli.remote._state import PipelineState, Segment, _Match
from part_io.services.audio_detection import (
    DetectionBatchRequest,
    apply_batch_result_to_episode,
    filter_matches_by_position,
    run_detection_batch,
)

_LOG = logging.getLogger(__name__)


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _build_consensus_from_segments(positives: list[Segment]) -> np.ndarray | None:
    tuples = [(Path(seg.source), seg.start, seg.end) for seg in positives if seg.source]
    return build_consensus_profile(tuples) if tuples else None


def _process_detection_results(
    results,
    jobs,
    state: PipelineState,
    ep_by_stem: dict[str, Path],
    duration_by_stem: dict[str, float | None],
) -> None:
    for done, result in enumerate(results, start=1):
        stem = result.stem
        kind = result.kind
        filtered_result = result
        if kind in ("intro", "outro"):
            duration = duration_by_stem.get(stem)
            if duration is not None:
                filtered_result = type(result)(
                    stem=result.stem,
                    source_path=result.source_path,
                    sample_path=result.sample_path,
                    kind=result.kind,
                    matches=filter_matches_by_position(
                        result.matches,
                        kind=result.kind,
                        source_duration_seconds=duration,
                    ),
                    error=result.error,
                )
        ep_state = state.episode(stem)
        score_str, error_msg = apply_batch_result_to_episode(
            filtered_result,
            ep_state,
            match_factory=lambda match: _Match(
                score=float(match.score),
                start=float(match.start_seconds),
                end=float(match.end_seconds),
            ),
            uncertain_label="uncertain",
            undetected_label="undetected",
        )
        if error_msg:
            _emit(error_msg)
        ep_state.source = str(ep_by_stem[stem])
        _emit(f"  [{done}/{len(jobs)}] {kind:5}  {stem}  ({score_str})")


def _detect_batch(
    episodes: list[Path],
    state: PipelineState,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path | None = None,
    outro_sample: Path | None = None,
    *,
    step_seconds: float,
    workers: int,
    max_matches: int,
    profile_cache_dir: Path | None = None,
) -> None:
    ep_by_stem = {ep.stem: ep for ep in episodes}
    duration_by_stem = {ep.stem: _probe_audio_duration_seconds(ep) for ep in episodes}

    open_consensus = _build_consensus_from_segments(state.open_target.positives)
    close_consensus = _build_consensus_from_segments(state.close_target.positives)

    consensus_map: dict[Path, np.ndarray] = {}
    if open_consensus is not None:
        consensus_map[open_sample] = open_consensus
        _emit(f"  [consensus] open: averaged {len(state.open_target.positives)} positives")
    if close_consensus is not None:
        consensus_map[close_sample] = close_consensus
        _emit(f"  [consensus] close: averaged {len(state.close_target.positives)} positives")

    def _detector(
        *, source_path: Path, sample_path: Path, score_threshold: float, step_seconds: float
    ) -> list[AudioMatch]:
        _LOG.info("  [detect] %s  %s", sample_path.stem, source_path.stem)
        if sample_path in consensus_map:
            matches = find_audio_sample_matches_from_profile(
                source_path=source_path,
                reference=consensus_map[sample_path],
                score_threshold=score_threshold,
                step_seconds=step_seconds,
                profile_cache_dir=profile_cache_dir,
            )
        else:
            matches = find_audio_sample_matches(
                source_path=source_path,
                sample_path=sample_path,
                score_threshold=score_threshold,
                step_seconds=step_seconds,
                profile_cache_dir=profile_cache_dir,
            )
        return matches

    if profile_cache_dir is not None:
        for ep in episodes:
            warm_source_profile(ep, profile_cache_dir)

    detector = _detector
    jobs, results = run_detection_batch(
        DetectionBatchRequest(
            episodes=episodes,
            open_sample=open_sample,
            close_sample=close_sample,
            intro_sample=intro_sample,
            outro_sample=outro_sample,
        ),
        detector=detector,
        step_seconds=step_seconds,
        max_matches=max_matches,
        workers=workers,
    )

    results = [
        type(result)(
            stem=result.stem,
            source_path=result.source_path,
            sample_path=result.sample_path,
            kind=result.kind,
            matches=_refine_matches(
                matches=list(result.matches),
                source_path=result.source_path,
                sample_path=result.sample_path,
            ),
            error=result.error,
        )
        for result in results
    ]

    _process_detection_results(results, jobs, state, ep_by_stem, duration_by_stem)


def _probe_audio_duration_seconds(source_path: Path) -> float | None:
    result = run_resolved(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    raw_out = result.stdout
    text = raw_out.decode("utf-8", errors="ignore") if isinstance(raw_out, bytes) else str(raw_out)
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None
