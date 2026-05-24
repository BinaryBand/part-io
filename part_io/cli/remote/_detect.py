from __future__ import annotations

import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from part_io.adapters.audio.matcher import (
    _ANALYSIS_RATE,
    AudioMatch,
    _build_spectral_profile,
    _decode_pcm_mono_16k,
    build_consensus_profile,
    find_audio_sample_matches_from_profile,
    warm_source_profile,
)
from part_io.adapters.audio.refine_impl import align_matches_to_onset as _align_to_onset
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


def _resolve_segment_path(stem: str, remote_dir: Path) -> Path:
    for ext in (".mp3", ".opus"):
        candidate = remote_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return remote_dir / f"{stem}.mp3"


def _build_consensus_from_segments(
    positives: list[Segment],
    *,
    remote_dir: Path,
    episodes: dict,
) -> np.ndarray | None:
    tuples: list[tuple[Path, float, float]] = []
    for seg in positives:
        if not seg.stem:
            continue
        path = _resolve_segment_path(seg.stem, remote_dir)
        ep = episodes.get(seg.stem)
        if ep is not None and ep.source_hash and not ep.source_hash_valid(path):
            continue  # file changed since segment was recorded — skip stale
        tuples.append((path, seg.start, seg.end))
    return build_consensus_profile(tuples) if tuples else None


def _init_profiles(
    state: PipelineState,
    snippets: dict[str, Path | None],
    remote_dir: Path,
    *,
    emit: "Callable[[str], None]",
) -> None:
    """Compute/refresh profiles for all kinds and store on state.profiles.

    Priority:
      1. Fresh consensus from hash-valid positives (most accurate) → update checkpoint.
      2. Existing checkpoint in state.profiles (transplant / no-audio-available).
      3. Bootstrap from snippet file (first run) → save as checkpoint.
      4. Hard fail — no profile, no snippet.
    """
    for kind in snippets:
        target = state.open_target if kind == "open" else state.close_target
        consensus = _build_consensus_from_segments(
            target.positives, remote_dir=remote_dir, episodes=state.episodes
        )
        if consensus is not None:
            state.profiles[kind] = consensus
            emit(f"  [profile:{kind}] consensus from {len(target.positives)} positive(s)")
            continue

        if kind in state.profiles:
            emit(f"  [profile:{kind}] loaded from checkpoint")
            continue

        snippet_path = snippets[kind]
        if snippet_path is not None and snippet_path.exists():
            samples = _decode_pcm_mono_16k(snippet_path)
            if not samples:
                raise RuntimeError(
                    f"Cannot build profile for '{kind}':"
                    f" snippet '{snippet_path}' produced no audio."
                )
            arr = np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)
            state.profiles[kind] = arr
            emit(f"  [profile:{kind}] bootstrapped from snippet '{snippet_path.name}'")
            continue

        raise RuntimeError(
            f"No profile available for kind '{kind}': "
            "provide a snippet file or seed with at least 2 approved positives."
        )


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
        )
        if error_msg:
            _emit(error_msg)
        source_path = ep_by_stem[stem]
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
    remote_dir: Path,
    step_seconds: float,
    workers: int,
    max_matches: int,
    profile_cache_dir: Path | None = None,
) -> None:
    ep_by_stem = {ep.stem: ep for ep in episodes}
    duration_by_stem = {ep.stem: _probe_audio_duration_seconds(ep) for ep in episodes}

    _init_profiles(state, snippets, remote_dir, emit=_emit)

    def _detector(
        *,
        source_path: Path,
        sample_path: Path | None,  # noqa: ARG001 — profile-only; sample_path unused
        kind: str = "",
        score_threshold: float,
        step_seconds: float,
    ) -> list[AudioMatch]:
        if kind not in state.profiles:
            raise RuntimeError(
                f"No profile for kind '{kind}'. This should have been caught by _init_profiles."
            )
        return find_audio_sample_matches_from_profile(
            source_path=source_path,
            reference=state.profiles[kind],
            score_threshold=score_threshold,
            step_seconds=step_seconds,
            profile_cache_dir=profile_cache_dir,
        )

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
            matches=_align_to_onset(
                matches=list(result.matches),
                source_path=result.source_path,
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
