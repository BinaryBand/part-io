"""Remote episode pipeline: detect ad-break positions, review them, cut them out.

Subcommands:
  review  — detect open/close matches per episode, review each interactively
  cut     — use state.toml to pair and cut ad segments
    loop    - detect -> review -> cut one episode at a time until done

State is stored entirely in {review_root}/state.toml.
No clip files or manifest CSVs are written.
Delete state.toml to start fresh.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from part_io.adapters.audio.ad_segments import pair_ad_segments
from part_io.adapters.audio.matcher import AudioMatch, warm_source_profile
from part_io.adapters.process.runner import run_resolved
from part_io.cli.audio_ad_remove import _build_filter_complex, _run_ffmpeg
from part_io.cli.remote._detect import (
    _detect_batch,
)
from part_io.cli.remote._state import (
    _AUDIO_KINDS,
    _POS,
    _UNC,
    EpisodeState,
    PipelineState,
    _replace_target_from_dict_lists,
    _target_to_dict_lists,
)
from part_io.services.cut_planning import build_cut_plan
from part_io.services.review_orchestration import (
    ReviewItem,
    UndoEntry,
    apply_review_decision,
    apply_review_dict_classes,
    collect_uncertain_candidates,
    episode_to_review_dict,
    next_uncertain_episode_kind,
    reclassify_all_episodes,
    undo_review_decision,
)
from part_io.utils.audio_process import AudioProcessManager
from part_io.utils.config import get_profile_cache_dir

if TYPE_CHECKING:  # pragma: no cover - type-only import
    import tomllib

_tomllib_runtime: Any = None
try:  # pragma: no cover - runtime optional deps
    import tomllib as _tomllib_std
except ModuleNotFoundError:
    try:
        import tomli as _tomli_pkg
    except ModuleNotFoundError:
        _tomllib_runtime = None
    else:
        _tomllib_runtime = _tomli_pkg
else:
    _tomllib_runtime = _tomllib_std

tomllib: Any = _tomllib_runtime

_MIN_EPISODE_BYTES = 10 * 1024 * 1024  # skip promos — < 10 MB ≈ < 5 min at 128 kbps
_STATE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "schemas" / "remote_pipeline_state.schema.json"
)
_AUDIO_MGR = AudioProcessManager()
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
# State helpers (moved to part_io.cli.remote._state)


def _reclassify_all(state: PipelineState) -> None:
    """Delegate MOE-based reclassification to the review orchestration service.

    The service operates on a plain dict shape (episodes and target score lists).
    Translate `PipelineState` into that shape, call the service, and write
    resulting class values back into the `PipelineState` in-place.
    """
    # Build plain-serialisable episodes dict expected by the service
    episodes_dict: dict[str, dict[str, Any]] = {}
    for stem, ep in state.episodes.items():
        episodes_dict[stem] = episode_to_review_dict(ep, include_bounds=True)

    open_pos = [{"score": float(s.score)} for s in state.open_target.positives]
    open_neg = [{"score": float(s.score)} for s in state.open_target.negatives]
    close_pos = [{"score": float(s.score)} for s in state.close_target.positives]
    close_neg = [{"score": float(s.score)} for s in state.close_target.negatives]

    # Delegate to service (in-place mutation of episodes_dict)
    reclassify_all_episodes(episodes_dict, open_pos, open_neg, close_pos, close_neg)

    # Write classifications back into PipelineState
    for stem, ep_dict in episodes_dict.items():
        if stem not in state.episodes:
            continue
        apply_review_dict_classes(state.episodes[stem], ep_dict)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_AUDIO_EXTENSIONS = frozenset({".mp3", ".opus"})


def _full_episodes(remote_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in remote_dir.iterdir()
        if p.suffix.lower() in _AUDIO_EXTENSIONS
        and p.is_file()
        and p.stat().st_size >= _MIN_EPISODE_BYTES
    )


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _resolve_opt(cli_value: Any, state_value: Any) -> Any:
    return state_value if cli_value is None else cli_value


def _apply_sticky_review_args(args: argparse.Namespace, state: PipelineState) -> None:
    s = state.settings
    args.snippets_dir = Path(_resolve_opt(args.snippets_dir, s.snippets_dir))
    args.open_sample = str(_resolve_opt(args.open_sample, s.open_sample))
    args.close_sample = str(_resolve_opt(args.close_sample, s.close_sample))
    args.intro_sample = str(_resolve_opt(args.intro_sample, s.intro_sample))
    args.outro_sample = _resolve_opt(args.outro_sample, s.outro_sample)
    args.step_seconds = float(_resolve_opt(args.step_seconds, s.step_seconds))
    args.workers = int(_resolve_opt(args.workers, s.workers))
    args.max_matches = int(_resolve_opt(args.max_matches, s.max_matches))
    # no_interactive is a session flag — never sticky; always use CLI value (default False)
    args.no_interactive = bool(args.no_interactive)
    args.overwrite = bool(_resolve_opt(args.overwrite, s.overwrite))

    s.snippets_dir = str(args.snippets_dir)
    s.open_sample = args.open_sample
    s.close_sample = args.close_sample
    s.intro_sample = args.intro_sample
    s.outro_sample = str(args.outro_sample) if args.outro_sample else None
    s.step_seconds = args.step_seconds
    s.workers = args.workers
    s.max_matches = args.max_matches
    s.overwrite = args.overwrite


def _apply_sticky_cut_args(args: argparse.Namespace, state: PipelineState) -> None:
    s = state.settings
    args.output_dir = Path(_resolve_opt(args.output_dir, s.output_dir))
    args.min_gap = float(_resolve_opt(args.min_gap, s.min_gap))
    args.max_gap = float(_resolve_opt(args.max_gap, s.max_gap))
    # yes/dry_run are session flags — never sticky; always use CLI value (default False)
    args.yes = bool(args.yes)
    args.dry_run = bool(args.dry_run)
    args.inclusive = bool(_resolve_opt(args.inclusive, s.ad_inclusive))
    args.fade = float(_resolve_opt(args.fade, s.fade))

    s.output_dir = str(args.output_dir)
    s.min_gap = args.min_gap
    s.max_gap = args.max_gap
    s.ad_inclusive = args.inclusive
    s.fade = args.fade


def _apply_sticky_loop_args(args: argparse.Namespace, state: PipelineState) -> None:
    _apply_sticky_review_args(args, state)
    _apply_sticky_cut_args(args, state)
    s = state.settings
    args.quiz_size = int(_resolve_opt(args.quiz_size, s.quiz_size))
    args.debug = bool(_resolve_opt(args.debug, s.debug))
    s.quiz_size = args.quiz_size
    s.debug = args.debug


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------


try:
    import termios
    import tty

    _STDIN_FD = sys.stdin.fileno() if sys.stdin.isatty() else None
    _INITIAL_TTY_ATTRS = termios.tcgetattr(_STDIN_FD) if _STDIN_FD is not None else None

    def _restore_tty() -> None:
        if _STDIN_FD is None or _INITIAL_TTY_ATTRS is None:
            return
        try:
            termios.tcsetattr(_STDIN_FD, termios.TCSADRAIN, _INITIAL_TTY_ATTRS)
        except OSError:
            return

    atexit.register(_restore_tty)

    def _getch() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        # In raw mode Ctrl+C arrives as \x03 instead of SIGINT; re-raise it
        # here after the tty is restored so callers see a normal interrupt.
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch

except ImportError:

    def _getch() -> str:  # type: ignore[misc]
        line = input()
        return line[0].lower() if line else ""


def _start_audio(path: Path) -> Any:
    return _AUDIO_MGR.start_player(path)


def _start_audio_segment(source: Path, start: float, end: float) -> Any:
    """Stream a time slice from source directly through ffplay without writing to disk."""
    duration = max(0.0, end - start)
    args = [
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
    ]
    return _AUDIO_MGR.start_player(source, args=args)


def _stop_audio(proc: Any | None) -> None:
    if proc is None:
        return

    _AUDIO_MGR.stop(proc)


def _apply_review_decision(
    *,
    state: PipelineState,
    ep_state: EpisodeState,
    item: ReviewItem,
    source: Path,
    key: Literal["a", "r"],
) -> tuple[str, UndoEntry]:
    ep_dict = episode_to_review_dict(ep_state, include_bounds=True)
    open_pos, open_neg = _target_to_dict_lists(state.open_target)
    close_pos, close_neg = _target_to_dict_lists(state.close_target)

    decision, undo = apply_review_decision(
        episode=ep_dict,
        kind=item.kind,
        candidate_idx=item.candidate_idx,
        action=key,
        source=str(source),
        open_target_positives=open_pos,
        open_target_negatives=open_neg,
        close_target_positives=close_pos,
        close_target_negatives=close_neg,
    )

    apply_review_dict_classes(ep_state, ep_dict)
    _replace_target_from_dict_lists(state.open_target, positives=open_pos, negatives=open_neg)
    _replace_target_from_dict_lists(state.close_target, positives=close_pos, negatives=close_neg)
    return decision.action, undo


def _undo_last_review(state: PipelineState, history: list[UndoEntry]) -> None:
    entry = history.pop()

    prev_ep = state.episodes[entry.stem]
    ep_dict = episode_to_review_dict(prev_ep, include_bounds=True)
    open_pos, open_neg = _target_to_dict_lists(state.open_target)
    close_pos, close_neg = _target_to_dict_lists(state.close_target)
    undo_review_decision(
        episode=ep_dict,
        undo=entry,
        open_target_positives=open_pos,
        open_target_negatives=open_neg,
        close_target_positives=close_pos,
        close_target_negatives=close_neg,
    )

    apply_review_dict_classes(prev_ep, ep_dict)
    _replace_target_from_dict_lists(state.open_target, positives=open_pos, negatives=open_neg)
    _replace_target_from_dict_lists(state.close_target, positives=close_pos, negatives=close_neg)
    _reclassify_all(state)
    print(
        f"\nundone ({entry.action} {entry.kind} for {entry.stem[:16]})",
        file=sys.stderr,
    )


def _review_candidate(
    state: PipelineState,
    item: ReviewItem,
    *,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path,
    outro_sample: Path,
    history: list[UndoEntry],
) -> str:
    """Review one candidate interactively.

    Returns 'approved', 'rejected', 'skipped', or 'undone'.
    """
    ep_state = state.episodes[item.stem]
    source = Path(ep_state.source)
    snippets: dict[str, Path] = {
        "open": open_sample,
        "close": close_sample,
        "intro": intro_sample,
        "outro": outro_sample,
    }
    snippet = snippets[item.kind]
    all_candidates = ep_state.candidates_for(item.kind)
    prev_class = ep_state.class_for(item.kind)
    candidate = all_candidates[item.candidate_idx]
    n_total = len(all_candidates)
    position_label = f" [{item.candidate_idx + 1}/{n_total}]" if n_total > 1 else ""

    undo_hint = "  [u]ndo" if history else ""
    legend = f"  [a]pprove  [r]eject  [p]replay  [c]ompare  [s]kip  [q]uit{undo_hint}  "
    score_str = f"score={candidate.score:.4f}  start={candidate.start:.1f}s"
    print(f"\n  [{item.kind}]{position_label}  {score_str}", file=sys.stderr)
    print(legend, end="", flush=True, file=sys.stderr)
    current_proc: Any | None = _start_audio_segment(source, candidate.start, candidate.end)

    while True:
        key = _getch().lower()
        _stop_audio(current_proc)
        current_proc = None

        if key == "p":
            current_proc = _start_audio_segment(source, candidate.start, candidate.end)
            print(f"\r{legend}", end="", flush=True, file=sys.stderr)
        elif key == "c":
            current_proc = _start_audio(snippet)
            print(f"\r{legend}", end="", flush=True, file=sys.stderr)
        elif key in ("a", "r"):
            action_key = cast(Literal["a", "r"], key)
            action, undo = _apply_review_decision(
                state=state,
                ep_state=ep_state,
                item=item,
                source=source,
                key=action_key,
            )
            undo.stem = item.stem
            undo.prev_class = prev_class
            history.append(
                UndoEntry(
                    stem=undo.stem,
                    kind=undo.kind,
                    action=undo.action,
                    segment_source=undo.segment_source,
                    segment_start=undo.segment_start,
                    segment_end=undo.segment_end,
                    segment_score=undo.segment_score,
                    target_list_was_positive=undo.target_list_was_positive,
                    prev_class=undo.prev_class,
                )
            )
            _reclassify_all(state)
            print(f"\n{action}", file=sys.stderr)
            return action
        elif key == "s":
            print("\nskipped", file=sys.stderr)
            return "skipped"
        elif key == "u" and history:
            _undo_last_review(state, history)
            return "undone"
        elif key == "q":
            print("\nQuitting review.", file=sys.stderr)
            raise KeyboardInterrupt


def _review_one_target(
    state: PipelineState,
    stem: str,
    kind: str,
    *,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path,
    outro_sample: Path,
    history: list[UndoEntry],
) -> str:
    """Review the top candidate for (stem, kind). Returns 'classified', 'skipped', or 'undone'."""
    ep_state = state.episodes[stem]
    all_candidates = ep_state.candidates_for(kind)
    if not all_candidates:
        return "skipped"
    item = ReviewItem(stem=stem, kind=kind, candidate_idx=0, score=all_candidates[0].score)
    result = _review_candidate(
        state,
        item,
        open_sample=open_sample,
        close_sample=close_sample,
        intro_sample=intro_sample,
        outro_sample=outro_sample,
        history=history,
    )
    return "classified" if result in ("approved", "rejected") else result


def _count_uncertain(state: PipelineState) -> int:
    return sum(
        1 for ep in state.episodes.values() for kind in _AUDIO_KINDS if ep.class_for(kind) == _UNC
    )


def _collect_uncertain_candidates(state: PipelineState) -> list[ReviewItem]:
    """Return uncertain candidates using the review orchestration service.

    Translate `PipelineState` into the plain dict/score-list shape the
    service expects, call `collect_uncertain_candidates`, and convert the
    returned `ReviewItem`s for the interactive review queue.
    """
    # Build episodes dict expected by the service
    episodes_dict: dict[str, dict[str, Any]] = {}
    for stem, ep in state.episodes.items():
        episodes_dict[stem] = episode_to_review_dict(ep, include_bounds=False)

    open_pos = [{"score": float(s.score)} for s in state.open_target.positives]
    open_neg = [{"score": float(s.score)} for s in state.open_target.negatives]
    close_pos = [{"score": float(s.score)} for s in state.close_target.positives]
    close_neg = [{"score": float(s.score)} for s in state.close_target.negatives]

    review_items = collect_uncertain_candidates(
        episodes_dict, open_pos, open_neg, close_pos, close_neg
    )

    return list(review_items)


def _run_review_loop(
    state: PipelineState,
    *,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path,
    outro_sample: Path,
    state_path: Path,
    max_decisions: int | None = None,
) -> None:
    """Review uncertain targets until queue empty, user quits, or max_decisions reached."""
    history: list[UndoEntry] = []
    skipped: set[tuple[str, str]] = set()
    decisions = 0
    while max_decisions is None or decisions < max_decisions:
        episodes_dict = {
            stem: episode_to_review_dict(ep, include_bounds=False)
            for stem, ep in state.episodes.items()
        }
        next_t = next_uncertain_episode_kind(episodes_dict, exclude=skipped)
        if next_t is None:
            n_uncertain = _count_uncertain(state)
            if n_uncertain:
                _emit(f"\n{n_uncertain} uncertain target(s) skipped — restart to revisit.")
            break
        stem, kind = next_t
        ep_state = state.episodes[stem]
        n_unc = _count_uncertain(state)
        _emit(f"\n{'=' * 60}")
        _emit(f"Episode: {stem}  ({n_unc} uncertain remaining)")
        _emit("=" * 60)
        result = _review_one_target(
            state,
            stem,
            kind,
            open_sample=open_sample,
            close_sample=close_sample,
            intro_sample=intro_sample,
            outro_sample=outro_sample,
            history=history,
        )
        if result == "classified":
            decisions += 1
            state.save(state_path)
            ep = state.episodes[stem]
            ep_class = ep.class_for(kind)
            if ep_class == _UNC:
                # Rejection didn't auto-classify (conservative MoE); defer until re-run.
                skipped.add((kind, stem))
            else:
                skipped.discard((kind, stem))
        elif result == "skipped":
            skipped.add((kind, stem))
        else:  # undone
            state.save(state_path)
        del ep_state  # avoid unused-variable lint
    if max_decisions is not None and decisions >= max_decisions:
        n_unc = _count_uncertain(state)
        if n_unc:
            _emit(
                f"\nBatch complete ({decisions} decisions)."
                f" {n_unc} uncertain remaining — run again."
            )


def _run_quiz(
    state: PipelineState,
    items: list[ReviewItem],
    *,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path,
    outro_sample: Path,
    state_path: Path,
    pre_skipped: set[tuple[str, str, int]] | None = None,
) -> tuple[int, bool, set[tuple[str, str, int]]]:
    """Review a pre-collected set of uncertain candidates.

    Returns (decisions_made, interrupted_by_quit, skipped_keys).
    skipped_keys includes both explicit skips and rejected-uncertain candidates
    that were not auto-classified, so callers can avoid re-presenting them.
    """
    history: list[UndoEntry] = []
    skipped_keys: set[tuple[str, str, int]] = set(pre_skipped or ())
    decisions = 0
    remaining = list(items)

    while remaining:
        active = next(
            (
                item
                for item in remaining
                if (item.stem, item.kind, item.candidate_idx) not in skipped_keys
                and state.episodes.get(item.stem) is not None
                and state.episodes[item.stem].class_for(item.kind) == _UNC
                and item.candidate_idx < len(state.episodes[item.stem].candidates_for(item.kind))
            ),
            None,
        )
        if active is None:
            break

        n_unc = _count_uncertain(state)
        _emit(f"\n{'=' * 60}")
        _emit(f"Episode: {active.stem}  [{active.kind}]  ({n_unc} uncertain remaining)")
        _emit("=" * 60)

        try:
            result = _review_candidate(
                state,
                active,
                open_sample=open_sample,
                close_sample=close_sample,
                intro_sample=intro_sample,
                outro_sample=outro_sample,
                history=history,
            )
        except KeyboardInterrupt:
            _emit("\nInterrupted. Progress saved.")
            return decisions, True, skipped_keys

        if result in ("approved", "rejected"):
            decisions += 1
            state.save(state_path)
            decided_key = (active.stem, active.kind, active.candidate_idx)
            uncertain_keys = {
                (i.stem, i.kind, i.candidate_idx) for i in _collect_uncertain_candidates(state)
            }
            remaining = [
                i
                for i in remaining
                if (i.stem, i.kind, i.candidate_idx) in uncertain_keys
                and (i.stem, i.kind, i.candidate_idx) != decided_key
            ]
            # Rejection that didn't auto-classify: mark as decided-this-session so
            # _run_loop_until_clean won't re-present it in the next outer pass.
            if result == "rejected" and decided_key in uncertain_keys:
                skipped_keys.add(decided_key)
        elif result == "skipped":
            skipped_keys.add((active.stem, active.kind, active.candidate_idx))
        else:  # undone
            state.save(state_path)
            remaining = [
                i
                for i in _collect_uncertain_candidates(state)
                if (i.stem, i.kind, i.candidate_idx) not in skipped_keys
            ]

    n_skipped = len(skipped_keys)
    if n_skipped:
        _emit(f"\n{n_skipped} candidate(s) skipped — restart to revisit.")
    return decisions, False, skipped_keys


# ---------------------------------------------------------------------------
# Pair and cut
# ---------------------------------------------------------------------------


def _find_best_pair(ep_state: EpisodeState, *, min_gap: float, max_gap: float) -> list[Any] | None:
    """Pass all open/close candidates to pair_ad_segments; return all valid pairs or None."""
    if not ep_state.candidates_for("open") or not ep_state.candidates_for("close"):
        return None
    opens = [
        AudioMatch(
            start_seconds=m.start,
            end_seconds=m.end,
            duration_seconds=m.end - m.start,
            score=m.score,
        )
        for m in ep_state.candidates_for("open")
    ]
    closes = [
        AudioMatch(
            start_seconds=m.start,
            end_seconds=m.end,
            duration_seconds=m.end - m.start,
            score=m.score,
        )
        for m in ep_state.candidates_for("close")
    ]
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
            print(f"  DEBUG FAILED: could not write {clip_path}", file=sys.stderr)
            return False
        wrote += 1
    print(f"  Debug clips written: {wrote} -> {clip_dir}")
    return True


def _execute_ffmpeg_cut(source: Path, filter_complex: str, output_path: Path) -> bool:
    """Execute ffmpeg cut and handle file placement. Returns True on success."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        exit_code = _run_ffmpeg(source, filter_complex, temp_path)
        if exit_code != 0:
            temp_path.unlink(missing_ok=True)
            print(f"  FAILED: ffmpeg exited {exit_code}", file=sys.stderr)
            return False

        try:
            temp_path.replace(output_path)
        except OSError:
            # Cross-device (e.g. rclone mount): copy then remove local temp.
            shutil.copy2(temp_path, output_path)
            temp_path.unlink(missing_ok=True)

        print(f"  Written: {output_path}")
        return True
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        print(f"  FAILED: {exc}", file=sys.stderr)
        return False


def _pair_and_cut(
    stem: str,
    source: Path,
    *,
    output_dir: Path,
    ep_state: EpisodeState,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    ad_inclusive: bool = True,
    intro_exclusive: bool = True,
    fade_dur: float = 0.5,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> str:
    """Pair open/close from ep_state and cut. Returns 'cut', 'skipped', or 'failed'."""
    if not ep_state.is_cuttable():
        print(f"  SKIP {stem}: open and close must both be classified as positive.")
        return "skipped"

    segments = _find_best_pair(ep_state, min_gap=min_gap, max_gap=max_gap)
    if segments is None:
        n_o = len(ep_state.candidates_for("open"))
        n_c = len(ep_state.candidates_for("close"))
        print(f"  No valid open->close pair ({n_o} open x {n_c} close candidates).")
        return "skipped"

    print(f"\n  {len(segments)} ad segment(s) to cut:")
    for i, seg in enumerate(segments, 1):
        cut_s = seg.cut_start if ad_inclusive else seg.open_end
        cut_e = seg.cut_end if ad_inclusive else seg.close_start
        print(f"    {i}. [{cut_s:.1f}s -> {cut_e:.1f}s]  ({cut_e - cut_s:.1f}s)")

    if dry_run:
        return "skipped"

    if not yes:
        n = len(segments)
        label = "ad" if n == 1 else f"{n} ads"
        print(f"\n  Cut {label} from {stem}? [y]es / [n]o  ", end="", flush=True)
        key = _getch().lower()
        print(key, file=sys.stderr)
        if key != "y":
            print("  Skipped.")
            return "skipped"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.mp3"
    if ad_inclusive:
        cuts = [(seg.cut_start, seg.cut_end) for seg in segments]
    else:
        cuts = [(seg.open_end, seg.close_start) for seg in segments]
    intro_trim = None
    intro_candidates = ep_state.candidates_for("intro")
    if ep_state.class_for("intro") == _POS and intro_candidates:
        intro_trim = intro_candidates[0].start if intro_exclusive else intro_candidates[0].end
    plan = build_cut_plan(cuts, intro_trim=intro_trim)

    if debug:
        if not _write_debug_clips(stem, source, plan.cuts, output_dir, debug_dir):
            return "failed"

    filter_complex, _ = _build_filter_complex(plan.spans, fade_dur=fade_dur)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not _execute_ffmpeg_cut(source, filter_complex, output_path):
        return "failed"

    return "cut"


# ---------------------------------------------------------------------------
# Cut helpers
# ---------------------------------------------------------------------------


def _cut_cuttable(
    state: PipelineState,
    *,
    remote_dir: Path,
    output_dir: Path,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    state_path: Path,
    ad_inclusive: bool = True,
    intro_exclusive: bool = True,
    fade_dur: float = 0.5,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> tuple[int, int, int]:
    """Cut all cuttable episodes. Returns (n_cut, n_skipped, n_failed)."""
    cuttable = [
        (stem, ep) for stem, ep in state.episodes.items() if ep.is_cuttable() and not ep.cut
    ]
    n_cut = n_skipped = n_failed = 0
    for stem, ep_state in sorted(cuttable):
        if ep_state.source:
            source = Path(ep_state.source)
        else:
            source = next(
                (
                    remote_dir / f"{stem}{ext}"
                    for ext in _AUDIO_EXTENSIONS
                    if (remote_dir / f"{stem}{ext}").exists()
                ),
                remote_dir / f"{stem}.mp3",
            )
        if not source.exists():
            print(f"SKIP {stem}: source not found at {source}")
            n_skipped += 1
            continue
        print(f"\n{stem}")
        result = _pair_and_cut(
            stem,
            source,
            output_dir=output_dir,
            ep_state=ep_state,
            min_gap=min_gap,
            max_gap=max_gap,
            yes=yes,
            dry_run=dry_run,
            ad_inclusive=ad_inclusive,
            intro_exclusive=intro_exclusive,
            fade_dur=fade_dur,
            debug=debug,
            debug_dir=debug_dir,
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


def _loop_work_counts(state: PipelineState, all_full: list[Path]) -> tuple[int, int, int]:
    """Return remaining undetected, uncertain, and cuttable counts."""
    n_undetected = sum(1 for ep in all_full if not state.episode(ep.stem).is_detected())
    n_uncertain = _count_uncertain(state)
    n_cuttable = sum(1 for ep in state.episodes.values() if ep.is_cuttable() and not ep.cut)
    return n_undetected, n_uncertain, n_cuttable


def _collect_loop_candidates(
    state: PipelineState,
    all_full: list[Path],
    *,
    overwrite: bool,
    quiz_size: int,
    step_seconds: float,
    workers: int,
    max_matches: int,
    state_path: Path,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path,
    outro_sample: Path | None,
    profile_cache_dir: Path | None = None,
    exclude_decided: set[tuple[str, str, int]] | None = None,
) -> tuple[list[ReviewItem], int]:
    """Detect until enough uncertain candidates are available or no work remains.

    *exclude_decided* filters out candidates already reviewed this session so they
    are not re-presented before new detections arrive.
    """
    undetected = [ep for ep in all_full if overwrite or not state.episode(ep.stem).is_detected()]
    n_already = len(all_full) - len(undetected)
    _emit(f"Episodes: {len(all_full)} total, {n_already} detected, {len(undetected)} to detect")

    _excl = exclude_decided or set()
    quiz_items = [
        i
        for i in _collect_uncertain_candidates(state)
        if (i.stem, i.kind, i.candidate_idx) not in _excl
    ]
    while len(quiz_items) < quiz_size and undetected:
        ep_path = undetected.pop(0)
        _emit(f"\nDetecting {ep_path.stem}...")
        _detect_batch(
            [ep_path],
            state,
            open_sample,
            close_sample,
            intro_sample,
            outro_sample,
            step_seconds=step_seconds,
            workers=workers,
            max_matches=max_matches,
            profile_cache_dir=profile_cache_dir,
        )
        _reclassify_all(state)
        state.save(state_path)
        quiz_items = [
            i
            for i in _collect_uncertain_candidates(state)
            if (i.stem, i.kind, i.candidate_idx) not in _excl
        ]

    return quiz_items[:quiz_size], _count_uncertain(state)


def _run_loop_cut_pass(
    state: PipelineState,
    *,
    all_full: list[Path],
    remote_dir: Path,
    output_dir: Path,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    state_path: Path,
    ad_inclusive: bool,
    intro_exclusive: bool,
    fade_dur: float,
    debug: bool,
) -> tuple[int, int, int, int, int, int]:
    """Run a cut pass and return counts plus remaining work summary."""
    n_cut, n_skipped, n_failed = _cut_cuttable(
        state,
        remote_dir=remote_dir,
        output_dir=output_dir,
        min_gap=min_gap,
        max_gap=max_gap,
        yes=yes,
        dry_run=dry_run,
        state_path=state_path,
        ad_inclusive=ad_inclusive,
        intro_exclusive=intro_exclusive,
        fade_dur=fade_dur,
        debug=debug,
    )
    n_remain_undet, n_remain_unc, n_remain_cut = _loop_work_counts(state, all_full)
    return n_cut, n_skipped, n_failed, n_remain_undet, n_remain_unc, n_remain_cut


def _run_loop_once(
    state: PipelineState,
    *,
    all_full: list[Path],
    remote_dir: Path,
    output_dir: Path,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path,
    outro_sample: Path | None,
    quiz_size: int,
    overwrite: bool,
    step_seconds: float,
    workers: int,
    max_matches: int,
    state_path: Path,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    ad_inclusive: bool,
    intro_exclusive: bool,
    fade_dur: float,
    debug: bool,
) -> None:
    """Run one detect/review/cut pass in non-interactive mode."""
    quiz_items, n_unc = _collect_loop_candidates(
        state,
        all_full,
        overwrite=overwrite,
        quiz_size=quiz_size,
        step_seconds=step_seconds,
        workers=workers,
        max_matches=max_matches,
        state_path=state_path,
        open_sample=open_sample,
        close_sample=close_sample,
        intro_sample=intro_sample,
        outro_sample=outro_sample,
        profile_cache_dir=get_profile_cache_dir(remote_dir),
    )
    if quiz_items:
        _emit(f"\n{len(quiz_items)} candidate(s) to review ({n_unc} uncertain total).")
    else:
        msg = (
            f"{n_unc} uncertain remaining — nothing new to review."
            if n_unc
            else "Nothing to review."
        )
        _emit(f"\n{msg}")

    n_cut, n_skipped, n_failed, n_remain_undet, n_remain_unc, n_remain_cut = _run_loop_cut_pass(
        state,
        all_full=all_full,
        remote_dir=remote_dir,
        output_dir=output_dir,
        min_gap=min_gap,
        max_gap=max_gap,
        yes=yes,
        dry_run=dry_run,
        state_path=state_path,
        ad_inclusive=ad_inclusive,
        intro_exclusive=intro_exclusive,
        fade_dur=fade_dur,
        debug=debug,
    )
    _emit(
        f"\nProgress: {n_remain_undet} undetected, {n_remain_unc} uncertain,"
        f" {n_remain_cut} cuttable remaining."
    )
    if n_cut or n_skipped or n_failed:
        print(f"Cut: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        sys.exit(1)


def _run_loop_until_clean(
    state: PipelineState,
    *,
    all_full: list[Path],
    remote_dir: Path,
    output_dir: Path,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path,
    outro_sample: Path | None,
    quiz_size: int,
    overwrite: bool,
    step_seconds: float,
    workers: int,
    max_matches: int,
    state_path: Path,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    ad_inclusive: bool,
    intro_exclusive: bool,
    fade_dur: float,
    debug: bool,
) -> None:
    """Run repeat detect/review/cut passes until no work remains or the user quits."""
    # Accumulate rejected-uncertain candidates across outer passes so the same
    # candidate is never re-presented within the same session.
    session_skipped: set[tuple[str, str, int]] = set()
    while True:
        quiz_items, n_unc = _collect_loop_candidates(
            state,
            all_full,
            overwrite=overwrite,
            quiz_size=quiz_size,
            step_seconds=step_seconds,
            workers=workers,
            max_matches=max_matches,
            state_path=state_path,
            open_sample=open_sample,
            close_sample=close_sample,
            intro_sample=intro_sample,
            outro_sample=outro_sample,
            profile_cache_dir=get_profile_cache_dir(remote_dir),
            exclude_decided=session_skipped,
        )
        n_remain_undet, n_remain_unc, n_remain_cut = _loop_work_counts(state, all_full)
        if not n_remain_undet and not n_remain_unc and (not n_remain_cut or dry_run):
            _emit("\nDirectory is clean.")
            return

        if quiz_items:
            _emit(f"\n{len(quiz_items)} candidate(s) to review ({n_unc} uncertain total).")
            _, interrupted, new_skipped = _run_quiz(
                state,
                quiz_items,
                open_sample=open_sample,
                close_sample=close_sample,
                intro_sample=intro_sample,
                outro_sample=(outro_sample or intro_sample),
                state_path=state_path,
                pre_skipped=session_skipped,
            )
            session_skipped |= new_skipped
            if interrupted:
                return
        else:
            msg = (
                f"{n_unc} uncertain remaining — nothing new to review."
                if n_unc
                else "Nothing to review."
            )
            _emit(f"\n{msg}")

        n_cut, n_skipped, n_failed, n_remain_undet, n_remain_unc, n_remain_cut = _run_loop_cut_pass(
            state,
            all_full=all_full,
            remote_dir=remote_dir,
            output_dir=output_dir,
            min_gap=min_gap,
            max_gap=max_gap,
            yes=yes,
            dry_run=dry_run,
            state_path=state_path,
            ad_inclusive=ad_inclusive,
            intro_exclusive=intro_exclusive,
            fade_dur=fade_dur,
            debug=debug,
        )
        _emit(
            f"\nProgress: {n_remain_undet} undetected, {n_remain_unc} uncertain,"
            f" {n_remain_cut} cuttable remaining."
        )
        if n_cut or n_skipped or n_failed:
            print(f"Cut: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
        if n_failed:
            sys.exit(1)
        if not n_remain_undet and not n_remain_unc and (not n_remain_cut or dry_run):
            _emit("Directory is clean.")
            return


# ---------------------------------------------------------------------------
# review subcommand
# ---------------------------------------------------------------------------


def _cmd_review(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"

    state = PipelineState.load(state_path)
    _apply_sticky_review_args(args, state)
    state.save(state_path)

    open_sample = args.snippets_dir / args.open_sample
    close_sample = args.snippets_dir / args.close_sample
    intro_sample = args.snippets_dir / args.intro_sample
    outro_sample = args.snippets_dir / args.outro_sample if args.outro_sample else None

    for path, label in [
        (remote_dir, "Remote dir"),
        (open_sample, "Open sample"),
        (close_sample, "Close sample"),
    ]:
        if not path.exists():
            sys.exit(f"{label} not found: {path}")

    all_full = _full_episodes(remote_dir)
    if not all_full:
        sys.exit(f"No full-length MP3s (>= 10 MB) found in {remote_dir}")

    to_detect = [
        ep for ep in all_full if args.overwrite or not state.episode(ep.stem).is_detected()
    ]
    n_already = len(all_full) - len(to_detect)
    _emit(f"Episodes: {len(all_full)} total, {n_already} detected, {len(to_detect)} to detect")

    if to_detect:
        _emit(f"\nDetecting {len(to_detect)} episode(s) with {args.workers} worker(s)...")
        _detect_batch(
            to_detect,
            state,
            open_sample,
            close_sample,
            intro_sample,
            outro_sample,
            step_seconds=args.step_seconds,
            workers=args.workers,
            max_matches=args.max_matches,
            profile_cache_dir=get_profile_cache_dir(remote_dir),
        )
        _reclassify_all(state)
        state.save(state_path)

    if args.no_interactive:
        _emit("\nDetection complete. Run remote-review without --no-interactive to label.")
        return

    n_unc = _count_uncertain(state)
    _emit(f"\n{n_unc} uncertain target(s) to review.")
    if not n_unc:
        _emit("Nothing to review — all episodes classified.")
        return

    _run_review_loop(
        state,
        open_sample=open_sample,
        close_sample=close_sample,
        intro_sample=intro_sample,
        outro_sample=(outro_sample or intro_sample),
        state_path=state_path,
    )


# ---------------------------------------------------------------------------
# cut subcommand
# ---------------------------------------------------------------------------


def _cmd_cut(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"

    state = PipelineState.load(state_path)
    _apply_sticky_cut_args(args, state)
    state.save(state_path)
    output_dir: Path = args.output_dir
    n_cuttable = sum(1 for ep in state.episodes.values() if ep.is_cuttable() and not ep.cut)
    if not n_cuttable:
        print("No cuttable episodes in state.toml — need open and close both positive.")
        return

    print(f"Found {n_cuttable} cuttable episode(s).")
    n_cut, n_skipped, n_failed = _cut_cuttable(
        state,
        remote_dir=remote_dir,
        output_dir=output_dir,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        yes=args.yes,
        dry_run=args.dry_run,
        state_path=state_path,
        ad_inclusive=args.inclusive,
        intro_exclusive=state.settings.intro_exclusive,
        fade_dur=args.fade,
    )
    print(f"\nDone: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# loop subcommand - detect -> review -> cut in one pass
# ---------------------------------------------------------------------------


def _cmd_loop(args: argparse.Namespace) -> None:
    """Detect one episode at a time, accumulate uncertain candidates, quiz, then cut."""
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"

    state = PipelineState.load(state_path)
    _apply_sticky_loop_args(args, state)
    state.save(state_path)

    output_dir: Path = args.output_dir
    open_sample = args.snippets_dir / args.open_sample
    close_sample = args.snippets_dir / args.close_sample
    intro_sample = args.snippets_dir / args.intro_sample
    outro_sample = args.snippets_dir / args.outro_sample if args.outro_sample else None

    for path, label in [
        (remote_dir, "Remote dir"),
        (open_sample, "Open sample"),
        (close_sample, "Close sample"),
    ]:
        if not path.exists():
            sys.exit(f"{label} not found: {path}")

    all_full = _full_episodes(remote_dir)
    if not all_full:
        sys.exit(f"No full-length MP3s (>= 10 MB) found in {remote_dir}")
    if args.no_interactive:
        _run_loop_once(
            state,
            all_full=all_full,
            remote_dir=remote_dir,
            output_dir=output_dir,
            open_sample=open_sample,
            close_sample=close_sample,
            intro_sample=intro_sample,
            outro_sample=outro_sample,
            quiz_size=args.quiz_size,
            overwrite=args.overwrite,
            step_seconds=args.step_seconds,
            workers=args.workers,
            max_matches=args.max_matches,
            state_path=state_path,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            yes=args.yes,
            dry_run=args.dry_run,
            ad_inclusive=args.inclusive,
            intro_exclusive=state.settings.intro_exclusive,
            fade_dur=args.fade,
            debug=args.debug,
        )
        return

    _run_loop_until_clean(
        state,
        all_full=all_full,
        remote_dir=remote_dir,
        output_dir=output_dir,
        open_sample=open_sample,
        close_sample=close_sample,
        intro_sample=intro_sample,
        outro_sample=outro_sample,
        quiz_size=args.quiz_size,
        overwrite=args.overwrite,
        step_seconds=args.step_seconds,
        workers=args.workers,
        max_matches=args.max_matches,
        state_path=state_path,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        yes=args.yes,
        dry_run=args.dry_run,
        ad_inclusive=args.inclusive,
        intro_exclusive=state.settings.intro_exclusive,
        fade_dur=args.fade,
        debug=args.debug,
    )


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--step-seconds", type=float, default=None)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for detection",
    )
    p.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Top-N candidate positions to store",
    )


def _add_cut_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--min-gap",
        type=float,
        default=None,
        help="Min gap: open end -> close start",
    )
    p.add_argument(
        "--max-gap",
        type=float,
        default=None,
        help="Max gap in seconds (default 5 min)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        default=None,
        help="Cut without confirmation prompt",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Show cut plan without running ffmpeg",
    )
    p.add_argument(
        "--inclusive",
        action="store_true",
        default=None,
        help="Cut jingle transitions too",
    )
    p.add_argument(
        "--fade",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Fade duration at cut points (default 0.5s, 0 to disable)",
    )


def _add_remote_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "remote_dir",
        type=Path,
        nargs="?",
        default=Path("downloads/remote"),
        metavar="REMOTE_DIR",
        help="Directory of episode MP3s (default: downloads/remote)",
    )


def _add_verbose_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable INFO-level logging (shows ffmpeg decode progress, cache hits, etc.)",
    )


def _build_review_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("review", help="Detect matches and review them interactively")
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument("--snippets-dir", type=Path, default=None)
    p.add_argument("--open-sample", default=None)
    p.add_argument("--close-sample", default=None)
    p.add_argument("--intro-sample", default=None)
    p.add_argument("--outro-sample", default=None)
    _add_detect_args(p)
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument(
        "--no-interactive", action="store_true", default=None, help="Detect only, skip review"
    )


def _build_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("cut", help="Cut ad segments using labels from __state__.toml")
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument("--output-dir", type=Path, default=None)
    _add_cut_args(p)


def _build_loop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "loop",
        help="Detect one episode at a time, quiz uncertain candidates, then cut. Run repeatedly.",
    )
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument("--snippets-dir", type=Path, default=None)
    p.add_argument("--open-sample", default=None)
    p.add_argument("--close-sample", default=None)
    p.add_argument("--intro-sample", default=None)
    p.add_argument("--outro-sample", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    _add_detect_args(p)
    _add_cut_args(p)
    p.add_argument(
        "--quiz-size",
        type=int,
        default=None,
        help="Target number of uncertain candidates to collect before quizzing (default: 10)",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        default=None,
        help="Skip interactive review",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Re-detect and re-cut episodes",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Write each planned cut ad segment to output_dir/debug_ads for review.",
    )


def _build_precache_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "precache",
        help="Pre-warm spectral profile cache for all episodes overnight.",
    )
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Pause between episodes to reduce sustained CPU load (default: 5)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-build cache entries that already exist",
    )


def _cmd_precache(args: argparse.Namespace) -> None:
    import time

    remote_dir: Path = args.remote_dir
    sleep_s: float = args.sleep
    overwrite: bool = args.overwrite
    profile_cache_dir = get_profile_cache_dir(remote_dir)
    profile_cache_dir.mkdir(parents=True, exist_ok=True)

    episodes = _full_episodes(remote_dir)
    if not episodes:
        print("No episodes found.", file=sys.stderr)
        return

    cached = 0
    built = 0
    for i, ep in enumerate(episodes, 1):
        cache_path = profile_cache_dir / f"{ep.stem}.npz"
        if not overwrite and cache_path.exists():
            _LOG.info("[precache] %d/%d  %s  (cached)", i, len(episodes), ep.stem)
            cached += 1
            continue
        _LOG.info("[precache] %d/%d  %s  building…", i, len(episodes), ep.stem)
        print(f"[{i}/{len(episodes)}] {ep.stem}", flush=True)
        try:
            warm_source_profile(ep, profile_cache_dir)
            built += 1
        except Exception as exc:
            print(f"  WARNING: failed — {exc}", file=sys.stderr)
        if i < len(episodes) and sleep_s > 0:
            time.sleep(sleep_s)

    print(f"\nDone. {built} built, {cached} already cached.", flush=True)


def _precache_paths(remote_dir: Path) -> tuple[Path, Path]:
    """Return (pidfile, logfile) for background precache process."""
    base = remote_dir.parent
    return base / ".precache.pid", base / ".precache.log"


def _precache_running(pid_path: Path) -> int | None:
    """Return the running PID if the precache process is alive, else None."""
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _build_precache_start_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("precache-start", help="Start precache as a background process.")
    _add_remote_dir_arg(p)
    p.add_argument(
        "--sleep",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Pause between episodes (default: 10)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-build cache entries that already exist",
    )


def _build_precache_stop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("precache-stop", help="Stop a running background precache process.")
    _add_remote_dir_arg(p)


def _build_precache_status_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("precache-status", help="Show background precache status and recent log.")
    _add_remote_dir_arg(p)
    p.add_argument(
        "--lines",
        type=int,
        default=20,
        metavar="N",
        help="Number of recent log lines to show (default: 20)",
    )


def _cmd_precache_start(args: argparse.Namespace) -> None:
    from part_io.utils.exec import launch_resolved

    remote_dir: Path = args.remote_dir
    pid_path, log_path = _precache_paths(remote_dir)

    existing = _precache_running(pid_path)
    if existing is not None:
        print(f"Already running (pid {existing}). Use precache-stop first.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "part_io.cli.remote_pipeline",
        "precache",
        "--verbose",
        "--sleep",
        str(args.sleep),
        str(remote_dir),
    ]
    if args.overwrite:
        cmd.append("--overwrite")

    log_f = log_path.open("a")
    devnull = open(os.devnull, "rb")
    proc = launch_resolved(
        cmd,
        stdout=log_f,
        stderr=log_f,
        stdin=devnull,
        start_new_session=True,  # detach from the terminal's process group
    )
    pid_path.write_text(str(proc.pid))
    print(f"Started precache (pid {proc.pid}).")
    print(f"Log: {log_path}")
    print("Stop with: poetry run part-io-tasks remote-precache-stop")


def _cmd_precache_stop(args: argparse.Namespace) -> None:
    import signal

    remote_dir: Path = args.remote_dir
    pid_path, _ = _precache_paths(remote_dir)

    pid = _precache_running(pid_path)
    if pid is None:
        print("No precache process is running.")
        pid_path.unlink(missing_ok=True)
        return

    os.kill(pid, signal.SIGTERM)
    pid_path.unlink(missing_ok=True)
    print(f"Stopped precache (pid {pid}).")


def _cmd_precache_status(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    pid_path, log_path = _precache_paths(remote_dir)

    pid = _precache_running(pid_path)
    if pid is None:
        print("Precache: not running.")
        pid_path.unlink(missing_ok=True)
    else:
        print(f"Precache: running (pid {pid})")

    if log_path.exists():
        lines = log_path.read_text().splitlines()
        tail = lines[-args.lines :]
        print(f"\n--- last {len(tail)} lines of {log_path} ---")
        print("\n".join(tail))
    else:
        print("No log file found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote episode review and ad-cut pipeline.")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    _build_review_parser(sub)
    _build_cut_parser(sub)
    _build_loop_parser(sub)
    _build_precache_parser(sub)
    _build_precache_start_parser(sub)
    _build_precache_stop_parser(sub)
    _build_precache_status_parser(sub)
    args = parser.parse_args()

    if args.verbose if hasattr(args, "verbose") else False:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.subcommand == "review":
        _cmd_review(args)
    elif args.subcommand == "cut":
        _cmd_cut(args)
    elif args.subcommand == "precache":
        _cmd_precache(args)
    elif args.subcommand == "precache-start":
        _cmd_precache_start(args)
    elif args.subcommand == "precache-stop":
        _cmd_precache_stop(args)
    elif args.subcommand == "precache-status":
        _cmd_precache_status(args)
    else:
        _cmd_loop(args)


if __name__ == "__main__":
    main()
