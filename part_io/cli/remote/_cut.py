from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from part_io.adapters.audio.ad_segments import pair_ad_segments
from part_io.adapters.audio.matcher import AudioMatch
from part_io.adapters.process.runner import run_resolved
from part_io.cli.audio_ad_remove import _build_filter_complex, _run_ffmpeg
from part_io.cli.remote._review import _emit, _getch
from part_io.cli.remote._state import _POS, EpisodeState, PipelineState
from part_io.services.cut_planning import build_cut_plan


@dataclass(frozen=True)
class CutSettings:
    min_gap: float
    max_gap: float
    yes: bool
    dry_run: bool
    ad_inclusive: bool = True
    intro_exclusive: bool = True
    fade_dur: float = 0.5
    debug: bool = False
    debug_dir: Path | None = None


def _write_prompt(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def _to_audio_matches(candidates: list[Any]) -> list[AudioMatch]:
    return [
        AudioMatch(
            start_seconds=m.start,
            end_seconds=m.end,
            duration_seconds=m.end - m.start,
            score=m.score,
        )
        for m in candidates
    ]


def _find_best_pair(ep_state: EpisodeState, *, min_gap: float, max_gap: float) -> list[Any] | None:
    """Pass all open/close candidates to pair_ad_segments; return all valid pairs or None."""
    if not ep_state.candidates_for("open") or not ep_state.candidates_for("close"):
        return None
    opens = _to_audio_matches(ep_state.candidates_for("open"))
    closes = _to_audio_matches(ep_state.candidates_for("close"))
    try:
        segs, _, _ = pair_ad_segments(opens, closes, min_gap=min_gap, max_gap=max_gap)
    except (ValueError, KeyError):
        return None
    return segs if segs else None


def _write_debug_clips(
    stem: str,
    source: Path,
    cuts: list[tuple[float, float]],
    output_dir: Path,
    debug_dir: Path | None,
) -> bool:
    """Write debug clip files for each cut segment. Returns True on success."""
    clip_dir = debug_dir or (output_dir / "debug_ads")
    clip_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for idx, (cut_start, cut_end) in enumerate(cuts, start=1):
        clip_path = clip_dir / f"{stem}__ad_{idx:02d}.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ss",
            f"{cut_start:.3f}",
            "-to",
            f"{cut_end:.3f}",
            "-c",
            "copy",
            str(clip_path),
        ]
        result = run_resolved(cmd, capture_output=True)
        if result.returncode != 0:
            _emit(f"  DEBUG FAILED: could not write {clip_path}")
            return False
        wrote += 1
    _emit(f"  Debug clips written: {wrote} -> {clip_dir}")
    return True


def _execute_ffmpeg_cut(source: Path, filter_complex: str, output_path: Path) -> bool:
    """Execute ffmpeg cut and handle file placement. Returns True on success."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        exit_code = _run_ffmpeg(source, filter_complex, temp_path)
        if exit_code != 0:
            temp_path.unlink(missing_ok=True)
            _emit(f"  FAILED: ffmpeg exited {exit_code}")
            return False

        try:
            temp_path.replace(output_path)
        except OSError:
            # Cross-device (e.g. rclone mount): copy then remove local temp.
            shutil.copy2(temp_path, output_path)
            temp_path.unlink(missing_ok=True)

        _emit(f"  Written: {output_path}")
        return True
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        _emit(f"  FAILED: {exc}")
        return False


def _pair_and_cut(
    stem: str,
    source: Path,
    *,
    output_dir: Path,
    ep_state: EpisodeState,
    settings: CutSettings,
) -> str:
    """Pair open/close from ep_state and cut. Returns 'cut', 'skipped', or 'failed'."""
    if not ep_state.is_cuttable():
        _emit(f"  SKIP {stem}: open and close must both be classified as positive.")
        return "skipped"

    segments = _find_best_pair(ep_state, min_gap=settings.min_gap, max_gap=settings.max_gap)
    if segments is None:
        n_o = len(ep_state.candidates_for("open"))
        n_c = len(ep_state.candidates_for("close"))
        _emit(f"  No valid open->close pair ({n_o} open x {n_c} close candidates).")
        return "skipped"

    _emit(f"\n  {len(segments)} ad segment(s) to cut:")
    for i, seg in enumerate(segments, 1):
        cut_s = seg.cut_start if settings.ad_inclusive else seg.open_end
        cut_e = seg.cut_end if settings.ad_inclusive else seg.close_start
        _emit(f"    {i}. [{cut_s:.1f}s -> {cut_e:.1f}s]  ({cut_e - cut_s:.1f}s)")

    if settings.dry_run:
        return "skipped"

    if not settings.yes:
        n = len(segments)
        label = "ad" if n == 1 else f"{n} ads"
        _write_prompt(f"\n  Cut {label} from {stem}? [y]es / [n]o  ")
        key = _getch().lower()
        _emit(key)
        if key != "y":
            _emit("  Skipped.")
            return "skipped"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.mp3"
    if settings.ad_inclusive:
        cuts = [(seg.cut_start, seg.cut_end) for seg in segments]
    else:
        cuts = [(seg.open_end, seg.close_start) for seg in segments]
    intro_trim = None
    intro_candidates = ep_state.candidates_for("intro")
    if ep_state.class_for("intro") == _POS and intro_candidates:
        intro_trim = (
            intro_candidates[0].start if settings.intro_exclusive else intro_candidates[0].end
        )
    plan = build_cut_plan(cuts, intro_trim=intro_trim)

    if settings.debug:
        if not _write_debug_clips(stem, source, plan.cuts, output_dir, settings.debug_dir):
            return "failed"

    filter_complex, _ = _build_filter_complex(plan.spans, fade_dur=settings.fade_dur)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not _execute_ffmpeg_cut(source, filter_complex, output_path):
        return "failed"

    return "cut"


def _cut_cuttable(
    state: PipelineState,
    *,
    remote_dir: Path,
    output_dir: Path,
    settings: CutSettings,
    state_path: Path,
    audio_extensions: tuple[str, ...] = (".mp3", ".opus"),
) -> tuple[int, int, int]:
    """Cut all cuttable episodes. Returns (n_cut, n_skipped, n_failed)."""
    cuttable = [
        (stem, ep) for stem, ep in state.episodes.items() if ep.is_cuttable() and not ep.cut
    ]
    n_cut = n_skipped = n_failed = 0
    for stem, ep_state in sorted(cuttable):
        source = next(
            (
                remote_dir / f"{stem}{ext}"
                for ext in audio_extensions
                if (remote_dir / f"{stem}{ext}").exists()
            ),
            remote_dir / f"{stem}.mp3",
        )
        if not source.exists():
            _emit(f"SKIP {stem}: source not found at {source}")
            n_skipped += 1
            continue
        _emit(f"\n{stem}")
        result = _pair_and_cut(
            stem,
            source,
            output_dir=output_dir,
            ep_state=ep_state,
            settings=settings,
        )
        if result == "cut":
            ep_state.cut = True
            state.save(state_path)
            n_cut += 1
        elif result == "failed":
            n_failed += 1
        else:
            n_skipped += 1
    return n_cut, n_skipped, n_failed


__all__ = ["CutSettings", "_cut_cuttable", "_pair_and_cut"]
