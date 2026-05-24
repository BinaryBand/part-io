from __future__ import annotations

import atexit
import sys
from pathlib import Path
from typing import Any, Literal, cast

from part_io.cli.remote._state import (
    _AUDIO_KINDS,
    _UNC,
    EpisodeState,
    PipelineState,
    _replace_target_from_dict_lists,
    _target_to_dict_lists,
)
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

_AUDIO_MGR = AudioProcessManager()


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _write_stderr(text: str, *, end: str = "\n", flush: bool = False) -> None:
    sys.stderr.write(text + end)
    if flush:
        sys.stderr.flush()


def _reclassify_all(state: PipelineState) -> None:
    episodes_dict: dict[str, dict[str, Any]] = {}
    for stem, ep in state.episodes.items():
        episodes_dict[stem] = episode_to_review_dict(ep, include_bounds=True)

    open_pos = [{"score": float(s.score)} for s in state.open_target.positives]
    open_neg = [{"score": float(s.score)} for s in state.open_target.negatives]
    close_pos = [{"score": float(s.score)} for s in state.close_target.positives]
    close_neg = [{"score": float(s.score)} for s in state.close_target.negatives]

    reclassify_all_episodes(episodes_dict, open_pos, open_neg, close_pos, close_neg)

    for stem, ep_dict in episodes_dict.items():
        if stem not in state.episodes:
            continue
        apply_review_dict_classes(state.episodes[stem], ep_dict)


def _count_uncertain(state: PipelineState) -> int:
    return sum(
        1 for ep in state.episodes.values() for kind in _AUDIO_KINDS if ep.class_for(kind) == _UNC
    )


def _collect_uncertain_candidates(state: PipelineState) -> list[ReviewItem]:
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
    duration = max(0.0, end - start)
    args = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"]
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
    _emit(f"\nundone ({entry.action} {entry.kind} for {entry.stem[:16]})")


def _resolve_episode_source(stem: str, remote_dir: Path) -> Path:
    for ext in (".mp3", ".opus"):
        candidate = remote_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return remote_dir / f"{stem}.mp3"


def _review_candidate(
    state: PipelineState,
    item: ReviewItem,
    *,
    snippets: dict[str, Path],
    history: list[UndoEntry],
    remote_dir: Path,
) -> str:
    ep_state = state.episodes[item.stem]
    source = _resolve_episode_source(item.stem, remote_dir)
    snippet = snippets.get(item.kind)
    all_candidates = ep_state.candidates_for(item.kind)
    prev_class = ep_state.class_for(item.kind)
    candidate = all_candidates[item.candidate_idx]
    n_total = len(all_candidates)
    position_label = f" [{item.candidate_idx + 1}/{n_total}]" if n_total > 1 else ""

    undo_hint = "  [u]ndo" if history else ""
    legend = f"  [a]pprove  [r]eject  [p]replay  [c]ompare  [s]kip  [q]uit{undo_hint}  "
    score_str = f"score={candidate.score:.4f}  start={candidate.start:.1f}s"
    _emit(f"\n  [{item.kind}]{position_label}  {score_str}")
    _write_stderr(legend, end="", flush=True)
    current_proc: Any | None = _start_audio_segment(source, candidate.start, candidate.end)

    while True:
        key = _getch().lower()
        _stop_audio(current_proc)
        current_proc = None

        if key == "p":
            current_proc = _start_audio_segment(source, candidate.start, candidate.end)
            _write_stderr(f"\r{legend}", end="", flush=True)
        elif key == "c":
            if snippet is not None:
                current_proc = _start_audio(snippet)
            _write_stderr(f"\r{legend}", end="", flush=True)
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
                    candidate_idx=undo.candidate_idx,
                    target_list_was_positive=undo.target_list_was_positive,
                    prev_class=undo.prev_class,
                    prev_label=undo.prev_label,
                )
            )
            _reclassify_all(state)
            _emit(action)
            return action
        elif key == "s":
            _emit("skipped")
            return "skipped"
        elif key == "u" and history:
            _undo_last_review(state, history)
            return "undone"
        elif key == "q":
            _emit("Quitting review.")
            raise KeyboardInterrupt


def _review_one_target(
    state: PipelineState,
    stem: str,
    kind: str,
    *,
    snippets: dict[str, Path],
    history: list[UndoEntry],
    remote_dir: Path,
) -> str:
    ep_state = state.episodes[stem]
    all_candidates = ep_state.candidates_for(kind)
    if not all_candidates:
        return "skipped"
    item = ReviewItem(stem=stem, kind=kind, candidate_idx=0, score=all_candidates[0].score)
    result = _review_candidate(
        state,
        item,
        snippets=snippets,
        history=history,
        remote_dir=remote_dir,
    )
    return "classified" if result in ("approved", "rejected") else result


def _run_review_loop(
    state: PipelineState,
    *,
    snippets: dict[str, Path],
    state_path: Path,
    remote_dir: Path,
    max_decisions: int | None = None,
) -> None:
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
        n_unc = _count_uncertain(state)
        _emit(f"\n{'=' * 60}")
        _emit(f"Episode: {stem}  ({n_unc} uncertain remaining)")
        _emit("=" * 60)
        result = _review_one_target(
            state,
            stem,
            kind,
            snippets=snippets,
            history=history,
            remote_dir=remote_dir,
        )
        if result == "classified":
            decisions += 1
            state.save(state_path)
            ep = state.episodes[stem]
            ep_class = ep.class_for(kind)
            if ep_class == _UNC:
                skipped.add((kind, stem))
            else:
                skipped.discard((kind, stem))
        elif result == "skipped":
            skipped.add((kind, stem))
        else:
            state.save(state_path)

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
    snippets: dict[str, Path],
    state_path: Path,
    remote_dir: Path,
    pre_skipped: set[tuple[str, str, int]] | None = None,
) -> tuple[int, bool, set[tuple[str, str, int]]]:
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
                snippets=snippets,
                history=history,
                remote_dir=remote_dir,
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
            if result == "rejected" and decided_key in uncertain_keys:
                skipped_keys.add(decided_key)
        elif result == "skipped":
            skipped_keys.add((active.stem, active.kind, active.candidate_idx))
        else:
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


__all__ = [
    "ReviewItem",
    "UndoEntry",
    "_collect_uncertain_candidates",
    "_count_uncertain",
    "_getch",
    "_reclassify_all",
    "_run_quiz",
    "_run_review_loop",
]
