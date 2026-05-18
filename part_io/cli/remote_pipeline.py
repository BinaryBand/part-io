"""Two-phase remote episode pipeline: review clip generation and ad cutting.

Subcommands:
  review  — scan downloads/remote/*.mp3, generate open/close review bundles,
             then present each clip for interactive [p]lay / [a]pprove / [r]eject
  cut     — use labeled bundles to pair ad segments and write cleaned MP3s
  loop    — generate → review → cut one episode at a time until done

All quiz state (labels, adaptive thresholds) is stored in {review-root}/state.toml.
Delete that file to start fresh.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess  # nosemgrep: no-direct-subprocess-import-except-in-utils

import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from part_io.adapters.audio.ad_segments import load_manifest_matches, pair_ad_segments
from part_io.adapters.process.runner import run_resolved
from part_io.cli.audio_ad_remove import (
    _build_filter_complex,
    _build_keep_spans,
    _run_ffmpeg,
    _validate_segments,
)
from part_io.utils.exec import launch_resolved

_MIN_EPISODE_BYTES = 10 * 1024 * 1024  # skip promos — < 10 MB ≈ < 5 min at 128 kbps


# ---------------------------------------------------------------------------
# Pipeline state — persisted to {review_root}/state.toml
# ---------------------------------------------------------------------------


@dataclass
class EpisodeState:
    open_approved: list[int] = field(default_factory=list)
    open_rejected: list[int] = field(default_factory=list)
    close_approved: list[int] = field(default_factory=list)
    close_rejected: list[int] = field(default_factory=list)
    cut: bool = False

    def is_labeled(self) -> bool:
        return bool(
            self.open_approved or self.open_rejected or self.close_approved or self.close_rejected
        )


@dataclass
class PipelineState:
    """All persistent pipeline state. Load from / save to state.toml."""

    open_threshold: float = 0.8
    close_threshold: float = 0.8
    episodes: dict[str, EpisodeState] = field(default_factory=dict)

    def episode(self, stem: str) -> EpisodeState:
        if stem not in self.episodes:
            self.episodes[stem] = EpisodeState()
        return self.episodes[stem]

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        if not path.exists():
            return cls()
        if tomllib is None:
            print(
                "Warning: TOML support unavailable (Python < 3.11 and tomli not installed). "
                "Starting with empty state.",
                file=sys.stderr,
            )
            return cls()
        with path.open("rb") as f:
            data = tomllib.load(f)
        thresholds = data.get("thresholds", {})
        state = cls(
            open_threshold=float(thresholds.get("open", 0.8)),
            close_threshold=float(thresholds.get("close", 0.8)),
        )
        for stem, ep in data.get("episodes", {}).items():
            state.episodes[stem] = EpisodeState(
                open_approved=list(ep.get("open_approved", [])),
                open_rejected=list(ep.get("open_rejected", [])),
                close_approved=list(ep.get("close_approved", [])),
                close_rejected=list(ep.get("close_rejected", [])),
                cut=bool(ep.get("cut", False)),
            )
        return state

    def save(self, path: Path) -> None:
        lines = [
            "# Remote episode pipeline state.\n",
            "# Edit freely — delete this file to reset to a first-time run.\n",
            "\n",
            "[thresholds]\n",
            f"open  = {self.open_threshold:.6g}\n",
            f"close = {self.close_threshold:.6g}\n",
        ]
        for stem, ep in sorted(self.episodes.items()):
            lines.append(f"\n[episodes.{stem}]\n")
            lines.append(f"open_approved  = {ep.open_approved}\n")
            lines.append(f"open_rejected  = {ep.open_rejected}\n")
            lines.append(f"close_approved = {ep.close_approved}\n")
            lines.append(f"close_rejected = {ep.close_rejected}\n")
            lines.append(f"cut = {str(ep.cut).lower()}\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _full_episodes(remote_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in remote_dir.glob("*.mp3")
        if p.is_file() and p.stat().st_size >= _MIN_EPISODE_BYTES
    )


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _read_manifest_csv(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Interactive review helpers
# ---------------------------------------------------------------------------


try:
    import termios
    import tty

    def _getch() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

except ImportError:

    def _getch() -> str:  # type: ignore[misc]
        line = input()
        return line[0].lower() if line else ""


def _start_audio(path: Path) -> "subprocess.Popen[Any]":
    """Start ffplay non-blocking; returns the process so the caller can stop it."""
    return launch_resolved(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        stdout=open(os.devnull, "wb"),
        stderr=open(os.devnull, "wb"),
    )


def _stop_audio(proc: "subprocess.Popen[Any] | None") -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except OSError:
            proc.kill()


@dataclass
class SessionScores:
    """In-memory score accumulator for raising adaptive thresholds within a run."""

    approved_open: list[float] = field(default_factory=list)
    rejected_open: list[float] = field(default_factory=list)
    approved_close: list[float] = field(default_factory=list)
    rejected_close: list[float] = field(default_factory=list)

    def open_floor(self) -> float | None:
        return min(self.approved_open) * 0.995 if self.approved_open else None

    def close_floor(self) -> float | None:
        return min(self.approved_close) * 0.995 if self.approved_close else None


def _scores_list_for(session_scores: SessionScores, kind: str, action: str) -> list[float]:
    if action == "approved":
        return session_scores.approved_open if kind == "open" else session_scores.approved_close
    return session_scores.rejected_open if kind == "open" else session_scores.rejected_close


def _undo_action(
    history: list[tuple[str, int, float]],
    approved: list[int],
    rejected: list[int],
    session_scores: SessionScores,
    kind: str,
) -> tuple[int, str]:
    """Pop history, undo last action. Returns (i_delta, prev_action)."""
    prev_action, prev_index, prev_score = history.pop()
    if prev_action in ("approved", "rejected"):
        lst = approved if prev_action == "approved" else rejected
        lst.remove(prev_index)
        _scores_list_for(session_scores, kind, prev_action).remove(prev_score)
    print(f"↩ undone ({prev_action})")
    return -1, prev_action


def _handle_review_key(
    key: str,
    *,
    index: int,
    score: float,
    clip_path: Path,
    snippet_path: Path,
    kind: str,
    approved: list[int],
    rejected: list[int],
    history: list[tuple[str, int, float]],
    session_scores: SessionScores,
) -> tuple[int, "subprocess.Popen[Any] | None"]:
    """Handle one keypress. Returns (i_delta, new_proc); i_delta!=0 means advance row."""
    if key in ("p", "c"):
        proc = _start_audio(clip_path if key == "p" else snippet_path)
        return 0, proc
    if key == "a":
        approved.append(index)
        _scores_list_for(session_scores, kind, "approved").append(score)
        history.append(("approved", index, score))
        print("✓ approved")
        return 1, None
    if key == "r":
        rejected.append(index)
        _scores_list_for(session_scores, kind, "rejected").append(score)
        history.append(("rejected", index, score))
        print("✗ rejected")
        return 1, None
    if key == "s":
        history.append(("skipped", index, score))
        print("— skipped")
        return 1, None
    if key == "u" and history:
        i_delta, _ = _undo_action(history, approved, rejected, session_scores, kind)
        return i_delta, None
    if key == "q":
        print("\nQuitting review.")
        raise KeyboardInterrupt
    return 0, None


def _review_clip(
    *,
    row: dict,
    kind: str,
    n_rows: int,
    row_num: int,
    snippet_path: Path,
    approved: list[int],
    rejected: list[int],
    history: list[tuple[str, int, float]],
    session_scores: SessionScores,
    n_skipped: int,
) -> tuple[int, int]:
    """Review a single clip; returns (i_delta, n_skipped_delta)."""
    index = int(row["index"])
    score = float(row["score"])
    start = float(row["start_seconds"])
    clip_path = Path(row["clip_path"])

    print(f"\n  [{kind}] Clip {row_num}/{n_rows}  score={score:.4f}  start={start:.1f}s")

    if not clip_path.exists():
        print("  (clip file missing) — skipped")
        history.append(("skipped", index, score))
        return 1, n_skipped + 1

    undo_hint = "  [u]ndo" if history else ""
    legend = f"  [a]pprove  [r]eject  [p]replay  [c]ompare  [s]kip  [q]uit{undo_hint}  "
    current_proc: subprocess.Popen[Any] | None = _start_audio(clip_path)
    print(legend, end="", flush=True)

    while True:
        key = _getch().lower()
        _stop_audio(current_proc)
        current_proc = None
        i_delta, current_proc = _handle_review_key(
            key,
            index=index,
            score=score,
            clip_path=clip_path,
            snippet_path=snippet_path,
            kind=kind,
            approved=approved,
            rejected=rejected,
            history=history,
            session_scores=session_scores,
        )
        if key == "s":
            n_skipped += 1
        if i_delta != 0 or key in ("s",):
            return i_delta, n_skipped


def _review_bundle(
    bundle_dir: Path,
    kind: str,
    session_scores: SessionScores,
    *,
    snippet_path: Path,
    ep_state: EpisodeState,
) -> None:
    """Present each clip for interactive review; update ep_state and session_scores in-place."""
    manifest_path = bundle_dir / "matches_manifest.csv"

    if not manifest_path.exists():
        print(f"  [{kind}] No manifest — skipping.")
        return

    rows = _read_manifest_csv(manifest_path)
    if not rows:
        print(f"  [{kind}] Empty manifest.")
        return

    approved: list[int] = []
    rejected: list[int] = []
    n_skipped = 0
    history: list[tuple[str, int, float]] = []

    i = 0
    while i < len(rows):
        i_delta, n_skipped = _review_clip(
            row=rows[i],
            kind=kind,
            n_rows=len(rows),
            row_num=i + 1,
            snippet_path=snippet_path,
            approved=approved,
            rejected=rejected,
            history=history,
            session_scores=session_scores,
            n_skipped=n_skipped,
        )
        i += i_delta

    if kind == "open":
        ep_state.open_approved = approved
        ep_state.open_rejected = rejected
    else:
        ep_state.close_approved = approved
        ep_state.close_rejected = rejected

    print(f"\n  [{kind}] {len(approved)} approved  {len(rejected)} rejected  {n_skipped} skipped")


def _interactive_review_batch(
    batch: list[Path],
    review_root: Path,
    session_scores: SessionScores,
    state: PipelineState,
    *,
    open_sample: Path,
    close_sample: Path,
) -> None:
    for ep in batch:
        print(f"\n{'=' * 60}")
        print(f"Episode: {ep.stem}")
        print("=" * 60)
        ep_state = state.episode(ep.stem)
        _review_bundle(
            review_root / "open" / ep.stem,
            "open",
            session_scores,
            snippet_path=open_sample,
            ep_state=ep_state,
        )
        _review_bundle(
            review_root / "close" / ep.stem,
            "close",
            session_scores,
            snippet_path=close_sample,
            ep_state=ep_state,
        )


def _print_batch_summary(
    batch_num: int,
    session_scores: SessionScores,
    open_threshold: float,
    close_threshold: float,
) -> None:
    apo = session_scores.approved_open
    rpo = session_scores.rejected_open
    apc = session_scores.approved_close
    rpc = session_scores.rejected_close
    print(f"\nBatch {batch_num} summary:")
    min_open = f"  min score {min(apo):.4f}" if apo else ""
    min_close = f"  min score {min(apc):.4f}" if apc else ""
    print(f"  open:   {len(apo)} approved{min_open}  {len(rpo)} rejected")
    print(f"  close:  {len(apc)} approved{min_close}  {len(rpc)} rejected")
    print(f"  Thresholds → open: {open_threshold:.4f}  close: {close_threshold:.4f}")


# ---------------------------------------------------------------------------
# review subcommand
# ---------------------------------------------------------------------------


def _review_one(
    *,
    source: Path,
    sample: Path,
    bundle_name: str,
    review_root: Path,
    threshold: float,
    z_threshold: float,
    step_seconds: float,
    max_clips: int,
    refine: bool,
    overwrite: bool,
) -> int:
    command = [
        sys.executable,
        "-m",
        "part_io.cli.audio_review",
        str(source),
        str(sample),
        "--threshold",
        str(threshold),
        "--z-threshold",
        str(z_threshold),
        "--step-seconds",
        str(step_seconds),
        "--max-clips",
        str(max_clips),
        "--output-root",
        str(review_root),
        "--bundle-name",
        bundle_name,
    ]
    if refine:
        command.append("--refine")
    if overwrite:
        command.append("--overwrite")
    result = run_resolved(command, capture_output=True)
    if result.returncode != 0 and result.stderr:
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.flush()
    return int(result.returncode)


def _clips_exist(review_root: Path, stem: str) -> bool:
    manifest = review_root / "open" / stem / "matches_manifest.csv"
    return manifest.exists() and any(manifest.parent.glob("*.mp3"))


def _filter_unlabeled(
    all_full: list[Path],
    state: "PipelineState",
    overwrite: bool,
) -> tuple[list[Path], int]:
    """Return (episodes_to_process, n_already_done)."""
    episodes: list[Path] = []
    n_already_done = 0
    for ep in all_full:
        if not overwrite and state.episode(ep.stem).is_labeled():
            n_already_done += 1
        else:
            episodes.append(ep)
    return episodes, n_already_done


def _validate_remote_inputs(remote_dir: Path, open_sample: Path, close_sample: Path) -> None:
    """Validate required paths for remote review/cut workflows."""
    for path, label in [
        (remote_dir, "Remote dir"),
        (open_sample, "Open sample"),
        (close_sample, "Close sample"),
    ]:
        if not path.exists():
            sys.exit(f"{label} not found: {path}")


def _process_batch(
    batch_num: int,
    batch: list[Path],
    episodes: list[Path],
    *,
    args: argparse.Namespace,
    review_root: Path,
    open_sample: Path,
    close_sample: Path,
    open_threshold: float,
    close_threshold: float,
    session_scores: "SessionScores",
    state: "PipelineState",
    state_path: Path,
) -> tuple[float, float]:
    """Generate clips + optional interactive review for one batch. Returns updated thresholds."""
    start_idx = (batch_num - 1) * args.batch_size + 1
    end_idx = min(batch_num * args.batch_size, len(episodes))
    _emit(f"\nBatch {batch_num}: episodes {start_idx}–{end_idx} of {len(episodes)}")

    to_generate = [ep for ep in batch if args.overwrite or not _clips_exist(review_root, ep.stem)]
    if to_generate:
        _generate_batch_clips(
            to_generate,
            open_sample=open_sample,
            close_sample=close_sample,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
            review_root=review_root,
            args=args,
        )
    else:
        _emit(f"  All {len(batch)} episode(s) have existing clips — going straight to review.")

    if not args.no_interactive:
        _interactive_review_batch(
            batch,
            review_root,
            session_scores,
            state,
            open_sample=open_sample,
            close_sample=close_sample,
        )
        return _update_thresholds(
            session_scores, state, state_path, batch_num, open_threshold, close_threshold
        )
    print("\nClips generated. Run remote-review without --no-interactive to label them.")
    return open_threshold, close_threshold


def _cmd_review(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    review_root: Path = args.review_root
    open_sample = args.snippets_dir / args.open_sample
    close_sample = args.snippets_dir / args.close_sample
    state_path = review_root / "state.toml"

    _validate_remote_inputs(remote_dir, open_sample, close_sample)

    all_full = _full_episodes(remote_dir)
    if not all_full:
        sys.exit(f"No full-length MP3s (>= 10 MB) found in {remote_dir}")

    state = PipelineState.load(state_path)
    open_threshold = state.open_threshold
    close_threshold = state.close_threshold

    episodes, n_already_done = _filter_unlabeled(all_full, state, args.overwrite)
    print(f"Episodes to process: {len(episodes)}  ({n_already_done} already labeled, skipped)")
    if not episodes:
        print("All episodes already labeled. Use --overwrite to re-run.")
        return

    session_scores = SessionScores()
    for batch_num, batch in enumerate(_chunks(episodes, args.batch_size), 1):
        open_threshold, close_threshold = _process_batch(
            batch_num,
            batch,
            episodes,
            args=args,
            review_root=review_root,
            open_sample=open_sample,
            close_sample=close_sample,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
            session_scores=session_scores,
            state=state,
            state_path=state_path,
        )


def _update_thresholds(
    session_scores: SessionScores,
    state: PipelineState,
    state_path: Path,
    batch_num: int,
    open_threshold: float,
    close_threshold: float,
) -> tuple[float, float]:
    """Raise thresholds based on session scores and persist state. Returns updated thresholds."""
    floor_open = session_scores.open_floor()
    floor_close = session_scores.close_floor()
    if floor_open is not None:
        open_threshold = max(open_threshold, floor_open)
    if floor_close is not None:
        close_threshold = max(close_threshold, floor_close)
    state.open_threshold = open_threshold
    state.close_threshold = close_threshold
    state.save(state_path)
    _print_batch_summary(batch_num, session_scores, open_threshold, close_threshold)
    return open_threshold, close_threshold


def _generate_batch_clips(
    to_generate: list[Path],
    *,
    open_sample: Path,
    close_sample: Path,
    open_threshold: float,
    close_threshold: float,
    review_root: Path,
    args: argparse.Namespace,
) -> None:
    """Generate review clips for a batch of episodes using a thread pool."""
    jobs: list[tuple[Path, Path, str, float]] = [
        (ep, open_sample, f"open/{ep.stem}", open_threshold) for ep in to_generate
    ] + [(ep, close_sample, f"close/{ep.stem}", close_threshold) for ep in to_generate]
    _run_clip_pool(jobs, review_root=review_root, args=args)


def _run_clip_pool(
    jobs: list[tuple[Path, Path, str, float]],
    *,
    review_root: Path,
    args: argparse.Namespace,
) -> None:
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _review_one,
                source=src,
                sample=smp,
                bundle_name=bn,
                review_root=review_root,
                threshold=thresh,
                z_threshold=args.z_threshold,
                step_seconds=args.step_seconds,
                max_clips=args.max_clips,
                refine=args.refine,
                overwrite=args.overwrite,
            ): bn
            for src, smp, bn, thresh in jobs
        }
        n_jobs = len(futures)
        for future in as_completed(futures):
            bundle = futures[future]
            code = future.result()
            done += 1
            if code != 0:
                failed += 1
            _emit(f"  [{done}/{n_jobs}] {'FAILED' if code != 0 else 'done  '}  {bundle}")
    if failed:
        _emit(f"Warning: {failed} job(s) failed in this batch.")


# ---------------------------------------------------------------------------
# cut subcommand
# ---------------------------------------------------------------------------


def _cleanup_clips(bundle_dir: Path) -> None:
    for clip in bundle_dir.glob("*.mp3"):
        clip.unlink(missing_ok=True)


def _load_ad_segments(
    open_manifest: Path,
    close_manifest: Path,
    ep_state: "EpisodeState",
    min_gap: float,
    max_gap: float,
) -> "list[Any] | str":
    """Load and pair ad segments. Returns segment list or a skip-reason string."""
    try:
        open_indices = frozenset(ep_state.open_approved) or None
        close_indices = frozenset(ep_state.close_approved) or None
        opens = load_manifest_matches(open_manifest, approved_indices=open_indices)
        closes = load_manifest_matches(close_manifest, approved_indices=close_indices)
        segments, unpaired_opens, unpaired_closes = pair_ad_segments(
            opens, closes, min_gap=min_gap, max_gap=max_gap
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return f"  SKIP: {exc}"
    for m in unpaired_opens:
        print(f"  WARNING: unpaired open at {m.start_seconds:.1f}s")
    for m in unpaired_closes:
        print(f"  WARNING: unpaired close at {m.start_seconds:.1f}s")
    if not segments:
        return "  No ad segments detected — nothing to cut."
    try:
        _validate_segments(segments)
    except ValueError as exc:
        return f"  SKIP: {exc}"
    return segments


def _pair_and_cut(
    stem: str,
    source: Path,
    *,
    review_root: Path,
    output_dir: Path,
    ep_state: EpisodeState,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    cleanup: bool,
) -> str:
    """Load labels from ep_state, pair segments, cut. Returns 'cut', 'skipped', or 'failed'."""
    open_manifest = review_root / "open" / stem / "matches_manifest.csv"
    close_manifest = review_root / "close" / stem / "matches_manifest.csv"

    result = _load_ad_segments(open_manifest, close_manifest, ep_state, min_gap, max_gap)
    if isinstance(result, str):
        print(result)
        return "skipped"
    segments = result

    sorted_segs = sorted(segments, key=lambda s: s.cut_start)
    print(f"\n  {len(sorted_segs)} ad segment(s):")
    for i, seg in enumerate(sorted_segs, 1):
        print(
            f"    {i}. [{seg.cut_start:.1f}s → {seg.cut_end:.1f}s]"
            f"  ({seg.cut_end - seg.cut_start:.1f}s)"
        )

    if dry_run:
        return "skipped"

    if not yes:
        resp = input(f"\n  Cut {len(sorted_segs)} ad(s) from {stem}? [y/N] ").strip().lower()
        if resp != "y":
            print("  Skipped.")
            return "skipped"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.mp3"
    spans = _build_keep_spans(sorted_segs)
    filter_complex, _ = _build_filter_complex(spans)
    exit_code = _run_ffmpeg(source, filter_complex, output_path)

    if exit_code != 0:
        print(f"  FAILED: ffmpeg exited {exit_code}", file=sys.stderr)
        return "failed"

    print(f"  Written: {output_path}")

    if cleanup:
        _cleanup_clips(review_root / "open" / stem)
        _cleanup_clips(review_root / "close" / stem)

    return "cut"


def _cmd_cut(args: argparse.Namespace) -> None:
    review_root: Path = args.review_root
    remote_dir: Path = args.remote_dir
    output_dir: Path = args.output_dir
    state_path = review_root / "state.toml"

    state = PipelineState.load(state_path)

    labeled = {stem: ep for stem, ep in state.episodes.items() if ep.is_labeled() and not ep.cut}
    if not labeled:
        print("No labeled episodes found in state.toml. Run remote-review to label clips first.")
        return

    print(f"Found {len(labeled)} labeled episode(s).")
    n_cut = n_skipped = n_failed = 0

    for stem, ep_state in sorted(labeled.items()):
        source = remote_dir / f"{stem}.mp3"
        if not source.exists():
            print(f"SKIP {stem}: source not in {remote_dir}")
            n_skipped += 1
            continue

        print(f"\n{stem}")
        result = _pair_and_cut(
            stem,
            source,
            review_root=review_root,
            output_dir=output_dir,
            ep_state=ep_state,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            yes=args.yes,
            dry_run=args.dry_run,
            cleanup=args.cleanup,
        )
        if result == "cut":
            ep_state.cut = True
            state.save(state_path)
            n_cut += 1
        elif result == "failed":
            n_failed += 1
        else:
            n_skipped += 1

    print(f"\nDone: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# loop subcommand — generate → review → cut per episode in one pass
# ---------------------------------------------------------------------------


@dataclass
class LoopContext:
    args: argparse.Namespace
    state: PipelineState
    state_path: Path
    review_root: Path
    remote_dir: Path
    output_dir: Path
    open_sample: Path
    close_sample: Path
    session_scores: SessionScores


def _generate_episode_clips(
    ep: Path,
    *,
    open_sample: Path,
    close_sample: Path,
    review_root: Path,
    open_threshold: float,
    close_threshold: float,
    z_threshold: float,
    step_seconds: float,
    max_clips: int,
    refine: bool,
    overwrite: bool,
) -> None:
    jobs = [
        (ep, open_sample, f"open/{ep.stem}", open_threshold),
        (ep, close_sample, f"close/{ep.stem}", close_threshold),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(
                _review_one,
                source=src,
                sample=smp,
                bundle_name=bn,
                review_root=review_root,
                threshold=thresh,
                z_threshold=z_threshold,
                step_seconds=step_seconds,
                max_clips=max_clips,
                refine=refine,
                overwrite=overwrite,
            ): bn
            for src, smp, bn, thresh in jobs
        }
        for future in as_completed(futures):
            code = future.result()
            if code != 0:
                _emit(f"  WARNING: clip generation failed for {futures[future]}")


def _review_loop_episode(
    ep: Path,
    *,
    ctx: "LoopContext",
    ep_state: "EpisodeState",
    already_labeled: bool,
    open_threshold: float,
    close_threshold: float,
) -> tuple[float, float]:
    """Review one episode and update thresholds if new labels were recorded."""
    needs_review = not ctx.args.no_interactive and (ctx.args.overwrite or not already_labeled)
    if needs_review:
        _review_bundle(
            ctx.review_root / "open" / ep.stem,
            "open",
            ctx.session_scores,
            snippet_path=ctx.open_sample,
            ep_state=ep_state,
        )
        _review_bundle(
            ctx.review_root / "close" / ep.stem,
            "close",
            ctx.session_scores,
            snippet_path=ctx.close_sample,
            ep_state=ep_state,
        )
        floor_open = ctx.session_scores.open_floor()
        floor_close = ctx.session_scores.close_floor()
        if floor_open is not None:
            open_threshold = max(open_threshold, floor_open)
        if floor_close is not None:
            close_threshold = max(close_threshold, floor_close)
        ctx.state.open_threshold = open_threshold
        ctx.state.close_threshold = close_threshold
    elif already_labeled:
        print("  Already labeled — skipping review.")
    return open_threshold, close_threshold


def _cut_loop_episode(
    ep: Path,
    *,
    ctx: "LoopContext",
    ep_state: "EpisodeState",
) -> str:
    """Pair approved labels and run the audio cut step for one episode."""
    source = ctx.remote_dir / f"{ep.stem}.mp3"
    return _pair_and_cut(
        ep.stem,
        source,
        review_root=ctx.review_root,
        output_dir=ctx.output_dir,
        ep_state=ep_state,
        min_gap=ctx.args.min_gap,
        max_gap=ctx.args.max_gap,
        yes=ctx.args.yes,
        dry_run=ctx.args.dry_run,
        cleanup=ctx.args.cleanup,
    )


def _process_loop_episode(
    ep: Path,
    ep_num: int,
    n_total: int,
    *,
    ctx: "LoopContext",
    open_threshold: float,
    close_threshold: float,
) -> tuple[str, float, float]:
    """Generate, review, and cut one episode. Returns (result, open_threshold, close_threshold)."""
    print(f"\n{'=' * 60}")
    print(f"[{ep_num}/{n_total}] {ep.stem}")
    print("=" * 60)

    ep_state = ctx.state.episode(ep.stem)
    already_labeled = ep_state.is_labeled()
    needs_gen = ctx.args.overwrite or not _clips_exist(ctx.review_root, ep.stem)

    if needs_gen:
        _emit("  Generating clips...")
        _generate_episode_clips(
            ep,
            open_sample=ctx.open_sample,
            close_sample=ctx.close_sample,
            review_root=ctx.review_root,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
            z_threshold=ctx.args.z_threshold,
            step_seconds=ctx.args.step_seconds,
            max_clips=ctx.args.max_clips,
            refine=ctx.args.refine,
            overwrite=ctx.args.overwrite,
        )

    open_threshold, close_threshold = _review_loop_episode(
        ep,
        ctx=ctx,
        ep_state=ep_state,
        already_labeled=already_labeled,
        open_threshold=open_threshold,
        close_threshold=close_threshold,
    )

    result = _cut_loop_episode(ep, ctx=ctx, ep_state=ep_state)
    if result == "cut":
        ep_state.cut = True
    ctx.state.save(ctx.state_path)
    return result, open_threshold, close_threshold


def _cmd_loop(args: argparse.Namespace) -> None:
    """One episode at a time: generate clips → review → cut → next."""
    remote_dir: Path = args.remote_dir
    review_root: Path = args.review_root
    output_dir: Path = args.output_dir
    open_sample = args.snippets_dir / args.open_sample
    close_sample = args.snippets_dir / args.close_sample
    state_path = review_root / "state.toml"

    _validate_remote_inputs(remote_dir, open_sample, close_sample)

    all_full = _full_episodes(remote_dir)
    if not all_full:
        sys.exit(f"No full-length MP3s (>= 10 MB) found in {remote_dir}")

    state = PipelineState.load(state_path)
    open_threshold = state.open_threshold
    close_threshold = state.close_threshold

    episodes = [ep for ep in all_full if args.overwrite or not state.episode(ep.stem).cut]
    n_already_done = len(all_full) - len(episodes)
    print(f"Episodes to process: {len(episodes)}  ({n_already_done} already cut, skipped)")
    if not episodes:
        print("All episodes already cut. Use --overwrite to re-run.")
        return

    session_scores = SessionScores()
    ctx = LoopContext(
        args=args,
        state=state,
        state_path=state_path,
        review_root=review_root,
        remote_dir=remote_dir,
        output_dir=output_dir,
        open_sample=open_sample,
        close_sample=close_sample,
        session_scores=session_scores,
    )
    n_cut = n_skipped = n_failed = 0

    for ep_num, ep in enumerate(episodes, 1):
        result, open_threshold, close_threshold = _process_loop_episode(
            ep,
            ep_num,
            len(episodes),
            ctx=ctx,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
        )
        if result == "cut":
            n_cut += 1
        elif result == "failed":
            n_failed += 1
        else:
            n_skipped += 1

    print(f"\nDone: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _build_review_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "review", help="Generate open/close review bundles and label them interactively"
    )
    p.add_argument(
        "--remote-dir",
        type=Path,
        default=Path("downloads/remote"),
        help="Source MP3 directory (rclone mount)",
    )
    p.add_argument("--snippets-dir", type=Path, default=Path("downloads/snippets"))
    p.add_argument("--open-sample", default="open.mp3")
    p.add_argument("--close-sample", default="close.mp3")
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Initial match score floor (adapts upward after each batch)",
    )
    p.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score cutoff: keep scores >= mean + N*std (default: 3.0)",
    )
    p.add_argument("--step-seconds", type=float, default=0.1)
    p.add_argument("--max-clips", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=10, help="Episodes per batch (default: 10)")
    p.add_argument(
        "--workers", type=int, default=2, help="Parallel clip-generation workers (default: 2)"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate clips and re-review already-labeled episodes",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Generate clips only — skip interactive review session",
    )
    p.add_argument(
        "--refine", action="store_true", help="Fine-grained local refinement of coarse matches"
    )


def _build_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("cut", help="Detect ad segments from labeled bundles and write cleaned MP3s")
    p.add_argument("--remote-dir", type=Path, default=Path("downloads/remote"))
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads/remove"),
        help="Destination for cleaned MP3s (default: downloads/remove)",
    )
    p.add_argument(
        "--min-gap",
        type=float,
        default=-15.0,
        help="Min seconds between open end and close start (default: -15)",
    )
    p.add_argument("--max-gap", type=float, default=600.0)
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete clip files after a successful cut (keeps manifests)",
    )
    p.add_argument(
        "--yes", action="store_true", help="Non-interactive: cut without confirmation prompt"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cut plan for all labeled episodes without running ffmpeg",
    )


def _build_loop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "loop",
        help="One episode at a time: generate clips → review → cut → repeat until done",
    )
    p.add_argument("--remote-dir", type=Path, default=Path("downloads/remote"))
    p.add_argument("--snippets-dir", type=Path, default=Path("downloads/snippets"))
    p.add_argument("--open-sample", default="open.mp3")
    p.add_argument("--close-sample", default="close.mp3")
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    p.add_argument("--output-dir", type=Path, default=Path("downloads/remove"))
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Initial match score floor (adapts upward as you approve clips)",
    )
    p.add_argument("--z-threshold", type=float, default=3.0)
    p.add_argument("--step-seconds", type=float, default=0.1)
    p.add_argument("--max-clips", type=int, default=10)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--min-gap", type=float, default=-15.0)
    p.add_argument("--max-gap", type=float, default=600.0)
    p.add_argument(
        "--yes", action="store_true", help="Cut without asking for confirmation after each review"
    )
    p.add_argument("--dry-run", action="store_true", help="Show cut plan but do not run ffmpeg")
    p.add_argument(
        "--cleanup", action="store_true", help="Delete clip files after a successful cut"
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive review (use existing labels or all manifest rows)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate, re-review, and re-cut already-processed episodes",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote episode review and ad-cut pipeline.")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    _build_review_parser(sub)
    _build_cut_parser(sub)
    _build_loop_parser(sub)
    args = parser.parse_args()

    if args.subcommand == "review":
        _cmd_review(args)
    elif args.subcommand == "cut":
        _cmd_cut(args)
    else:
        _cmd_loop(args)


if __name__ == "__main__":
    main()
