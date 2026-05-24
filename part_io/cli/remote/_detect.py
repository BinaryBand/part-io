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
from part_io.utils.hash import partial_file_hash

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
        source_path = ep_by_stem[stem]
        ep_state.source = str(source_path)
        try:
            ep_state.source_hash = partial_file_hash(source_path)
        except OSError:
            ep_state.source_hash = None
        _emit(f"  [{done}/{len(jobs)}] {kind:5}  {stem}  ({score_str})")


def _detect_batch(
    episodes: list[Path],
    state: PipelineState,
    snippets: dict[str, Path | None],
    *,
    snippet_profiles: dict[str, np.ndarray] | None = None,
    step_seconds: float,
    workers: int,
    max_matches: int,
    profile_cache_dir: Path | None = None,
) -> None:
    ep_by_stem = {ep.stem: ep for ep in episodes}
    duration_by_stem = {ep.stem: _probe_audio_duration_seconds(ep) for ep in episodes}

    profile_map = dict(snippet_profiles or {})
    open_consensus = _build_consensus_from_segments(state.open_target.positives)
    if open_consensus is not None and "open" in snippets:
        profile_map["open"] = open_consensus
        _emit(f"  [consensus] open: averaged {len(state.open_target.positives)} positives")
    close_consensus = _build_consensus_from_segments(state.close_target.positives)
    if close_consensus is not None and "close" in snippets:
        profile_map["close"] = close_consensus
        _emit(f"  [consensus] close: averaged {len(state.close_target.positives)} positives")

    path_to_kind = {
        sample_path: kind for kind, sample_path in snippets.items() if sample_path is not None
    }

    def _detector(
        *, source_path: Path, sample_path: Path | None, score_threshold: float, step_seconds: float
    ) -> list[AudioMatch]:
        kind = path_to_kind.get(sample_path, "") if sample_path is not None else ""
        if kind and kind in profile_map:
            matches = find_audio_sample_matches_from_profile(
                source_path=source_path,
                reference=profile_map[kind],
                score_threshold=score_threshold,
                step_seconds=step_seconds,
                profile_cache_dir=profile_cache_dir,
            )
        else:
            if sample_path is None:
                msg = "missing sample path and no embedded profile"
                raise ValueError(msg)
            _LOG.info("  [detect] %s  %s", sample_path.stem, source_path.stem)
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
            snippets=snippets,
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
            matches=(
                _refine_matches(
                    matches=list(result.matches),
                    source_path=result.source_path,
                    sample_path=result.sample_path,
                )
                if result.sample_path is not None and result.sample_path.exists()
                else list(result.matches)
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
