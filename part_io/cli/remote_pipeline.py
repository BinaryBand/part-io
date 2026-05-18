"""Remote episode pipeline: detect ad-break positions, review them, cut them out.

Subcommands:
  review  — detect open/close matches per episode, review each interactively
  cut     — use labeled state.toml to pair and cut ad segments
  loop    — detect → review → cut one episode at a time until done

State is stored entirely in {review-root}/state.toml.
No clip files or manifest CSVs are written.
Delete state.toml to start fresh.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from part_io.adapters.audio.ad_segments import pair_ad_segments
from part_io.adapters.audio.matcher import AudioMatch
from part_io.adapters.process.runner import run_resolved
from part_io.cli.audio_ad_remove import (
    _build_filter_complex,
    _build_keep_spans,
    _run_ffmpeg,
    _validate_segments,
)
from part_io.utils.exec import launch_resolved

# TOML loader: prefer stdlib `tomllib`, fall back to `tomli`.
if TYPE_CHECKING:  # pragma: no cover - type-only import
    import tomllib


# runtime loader uses a temporary name so static analyzers don't see two
# different module types bound to the same name.
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


# ---------------------------------------------------------------------------
# State model — persisted to {review_root}/state.toml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchCandidate:
    """A single detected occurrence of a snippet inside a source file."""

    index: int
    score: float
    start: float  # seconds into source
    end: float  # seconds into source


@dataclass
class EpisodeState:
    source: str = ""  # relative path to source MP3
    open_matches: list[MatchCandidate] = field(default_factory=list)
    close_matches: list[MatchCandidate] = field(default_factory=list)
    open_approved: list[int] = field(default_factory=list)
    open_rejected: list[int] = field(default_factory=list)
    close_approved: list[int] = field(default_factory=list)
    close_rejected: list[int] = field(default_factory=list)
    cut: bool = False

    def has_matches(self) -> bool:
        return bool(self.open_matches or self.close_matches)

    def is_labeled(self) -> bool:
        return bool(
            self.open_approved or self.open_rejected or self.close_approved or self.close_rejected
        )


def _fmt_candidate(m: MatchCandidate) -> str:
    return f"{{index = {m.index}, score = {m.score:.4f}, start = {m.start:.3f}, end = {m.end:.3f}}}"


def _migrate_episode_keys(path: Path) -> None:
    """Re-quote any unquoted [episodes.*] table keys written by older versions.

    Bare TOML keys only allow A-Za-z0-9_-.  Episode stems that include spaces,
    question marks, dots, or other characters cause a parse error in strict
    TOML parsers.  This rewrites the file in-place so every episode key is a
    quoted string, then saves so future loads don't need migration.
    """
    text = path.read_text(encoding="utf-8")
    # Quick check: nothing to do if every episode key is already quoted.
    if not re.search(r'^\[episodes\.[^"]', text, re.MULTILINE):
        return

    fixed_lines = []
    for line in text.splitlines(keepends=True):
        m = re.match(r'^\[episodes\.([^"].+?)\]\s*$', line)
        if m:
            key = m.group(1)
            escaped = key.replace("\\", "\\\\").replace('"', '\\"')
            line = f'[episodes."{escaped}"]\n'
        fixed_lines.append(line)

    path.write_text("".join(fixed_lines), encoding="utf-8")


@dataclass
class PipelineState:
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
        _migrate_episode_keys(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        thresholds = data.get("thresholds", {})
        state = cls(
            open_threshold=float(thresholds.get("open", 0.8)),
            close_threshold=float(thresholds.get("close", 0.8)),
        )
        for stem, ep in data.get("episodes", {}).items():
            state.episodes[stem] = EpisodeState(
                source=str(ep.get("source", "")),
                open_matches=[
                    MatchCandidate(
                        index=int(m["index"]),
                        score=float(m["score"]),
                        start=float(m["start"]),
                        end=float(m["end"]),
                    )
                    for m in ep.get("open_matches", [])
                ],
                close_matches=[
                    MatchCandidate(
                        index=int(m["index"]),
                        score=float(m["score"]),
                        start=float(m["start"]),
                        end=float(m["end"]),
                    )
                    for m in ep.get("close_matches", [])
                ],
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
            "# Edit freely — delete this file to start fresh.\n",
            "\n",
            "[thresholds]\n",
            f"open  = {self.open_threshold:.6g}\n",
            f"close = {self.close_threshold:.6g}\n",
        ]
        for stem, ep in sorted(self.episodes.items()):
            # Always quote the key — bare TOML keys disallow spaces, dots, etc.
            lines.append(f'\n[episodes."{stem}"]\n')
            if ep.source:
                # json.dumps gives a properly-escaped TOML basic string.
                lines.append(f"source = {json.dumps(ep.source)}\n")
            open_m = ", ".join(_fmt_candidate(m) for m in ep.open_matches)
            close_m = ", ".join(_fmt_candidate(m) for m in ep.close_matches)
            lines.append(f"open_matches   = [{open_m}]\n")
            lines.append(f"close_matches  = [{close_m}]\n")
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


# ---------------------------------------------------------------------------
# Detection — runs audio_detect as subprocess, returns MatchCandidate list
# ---------------------------------------------------------------------------


def _detect_matches(
    source: Path,
    sample: Path,
    *,
    threshold: float,
    z_threshold: float | None,
    step_seconds: float,
    max_matches: int,
) -> list[MatchCandidate]:
    """Run audio_detect subprocess and parse JSON from stdout."""
    command = [
        sys.executable,
        "-m",
        "part_io.cli.audio_detect",
        str(source),
        str(sample),
        "--threshold",
        str(threshold),
        "--step-seconds",
        str(step_seconds),
        "--max-matches",
        str(max_matches),
    ]
    if z_threshold is not None:
        command.extend(["--z-threshold", str(z_threshold)])

    result = run_resolved(command, capture_output=True)
    if result.returncode != 0:
        if result.stderr:
            sys.stderr.buffer.write(result.stderr)
            sys.stderr.flush()
        return []
    try:
        data = json.loads(result.stdout)
        return [
            MatchCandidate(
                index=int(m["index"]),
                score=float(m["score"]),
                start=float(m["start"]),
                end=float(m["end"]),
            )
            for m in data
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _emit(f"  WARNING: failed to parse detect output: {exc}")
        return []


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


def _start_audio(path: Path) -> Any:
    return launch_resolved(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            str(path),
        ]
    )


def _start_audio_segment(source: Path, start: float, end: float) -> Any:
    """Stream a time slice from source directly through ffplay without writing to disk."""
    duration = max(0.0, end - start)
    return launch_resolved(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            str(source),
        ]
    )


def _stop_audio(proc: Any | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
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


def _review_bundle(
    matches: list[MatchCandidate],
    kind: str,
    source_path: Path,
    session_scores: SessionScores,
    *,
    snippet_path: Path,
    ep_state: EpisodeState,
) -> None:
    """Present each match for interactive review; update ep_state in-place."""
    if not matches:
        print(f"  [{kind}] No matches found.", file=sys.stderr)
        return

    approved: list[int] = []
    rejected: list[int] = []
    n_skipped = 0
    history: list[tuple[str, int, float]] = []

    i = 0
    while i < len(matches):
        m = matches[i]
        print(
            f"\n  [{kind}] Match {i + 1}/{len(matches)}  score={m.score:.4f}  start={m.start:.1f}s",
            file=sys.stderr,
        )

        undo_hint = "  [u]ndo" if history else ""
        legend = f"  [a]pprove  [r]eject  [p]replay  [c]ompare  [s]kip  [q]uit{undo_hint}  "

        print(legend, end="", flush=True, file=sys.stderr)
        current_proc: Any | None = _start_audio_segment(source_path, m.start, m.end)

        while True:
            key = _getch().lower()
            _stop_audio(current_proc)
            current_proc = None

            if key == "p":
                current_proc = _start_audio_segment(source_path, m.start, m.end)
                print(f"\r{legend}", end="", flush=True, file=sys.stderr)
            elif key == "c":
                current_proc = _start_audio(snippet_path)
                print(f"\r{legend}", end="", flush=True, file=sys.stderr)
            elif key == "a":
                approved.append(m.index)
                (
                    session_scores.approved_open
                    if kind == "open"
                    else session_scores.approved_close
                ).append(m.score)
                history.append(("approved", m.index, m.score))
                print("\n✓ approved", file=sys.stderr)
                i += 1
                break
            elif key == "r":
                rejected.append(m.index)
                (
                    session_scores.rejected_open
                    if kind == "open"
                    else session_scores.rejected_close
                ).append(m.score)
                history.append(("rejected", m.index, m.score))
                print("\n✗ rejected", file=sys.stderr)
                i += 1
                break
            elif key == "s":
                history.append(("skipped", m.index, m.score))
                n_skipped += 1
                print("\n— skipped", file=sys.stderr)
                i += 1
                break
            elif key == "u" and history:
                prev_action, prev_index, prev_score = history.pop()
                if prev_action == "approved":
                    approved.remove(prev_index)
                    scores_list = (
                        session_scores.approved_open
                        if kind == "open"
                        else session_scores.approved_close
                    )
                    scores_list.remove(prev_score)
                elif prev_action == "rejected":
                    rejected.remove(prev_index)
                    scores_list = (
                        session_scores.rejected_open
                        if kind == "open"
                        else session_scores.rejected_close
                    )
                    scores_list.remove(prev_score)
                elif prev_action == "skipped":
                    n_skipped -= 1
                print(f"\n↩ undone ({prev_action})", file=sys.stderr)
                i -= 1
                break
            elif key == "q":
                print("\nQuitting review.", file=sys.stderr)
                raise KeyboardInterrupt

    if kind == "open":
        ep_state.open_approved = approved
        ep_state.open_rejected = rejected
    else:
        ep_state.close_approved = approved
        ep_state.close_rejected = rejected

    print(
        f"\n  [{kind}] {len(approved)} approved  {len(rejected)} rejected  {n_skipped} skipped",
        file=sys.stderr,
    )


def _interactive_review_episode(
    ep: Path,
    session_scores: SessionScores,
    state: PipelineState,
    *,
    open_sample: Path,
    close_sample: Path,
) -> None:
    ep_state = state.episode(ep.stem)
    source_path = Path(ep_state.source) if ep_state.source else ep
    _review_bundle(
        ep_state.open_matches,
        "open",
        source_path,
        session_scores,
        snippet_path=open_sample,
        ep_state=ep_state,
    )
    _review_bundle(
        ep_state.close_matches,
        "close",
        source_path,
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
    apo, rpo = session_scores.approved_open, session_scores.rejected_open
    apc, rpc = session_scores.approved_close, session_scores.rejected_close
    min_open = f"  min score {min(apo):.4f}" if apo else ""
    min_close = f"  min score {min(apc):.4f}" if apc else ""
    print(f"\nBatch {batch_num} summary:")
    print(f"  open:   {len(apo)} approved{min_open}  {len(rpo)} rejected")
    print(f"  close:  {len(apc)} approved{min_close}  {len(rpc)} rejected")
    print(f"  Thresholds → open: {open_threshold:.4f}  close: {close_threshold:.4f}")


# ---------------------------------------------------------------------------
# Cut helpers
# ---------------------------------------------------------------------------


def _candidates_to_audio_matches(
    candidates: list[MatchCandidate], approved: list[int]
) -> list[AudioMatch]:
    """Convert approved MatchCandidates to AudioMatch objects for pair_ad_segments."""
    approved_set = frozenset(approved)
    filtered = [c for c in candidates if not approved_set or c.index in approved_set]
    return [
        AudioMatch(
            start_seconds=c.start,
            end_seconds=c.end,
            duration_seconds=c.end - c.start,
            score=c.score,
        )
        for c in filtered
    ]


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
) -> str:
    """Pair segments from ep_state and cut. Returns 'cut', 'skipped', or 'failed'."""
    opens = _candidates_to_audio_matches(ep_state.open_matches, ep_state.open_approved)
    closes = _candidates_to_audio_matches(ep_state.close_matches, ep_state.close_approved)

    if not opens and not closes:
        print("  No labeled matches — nothing to cut.")
        return "skipped"

    try:
        segments, unpaired_opens, unpaired_closes = pair_ad_segments(
            opens, closes, min_gap=min_gap, max_gap=max_gap
        )
    except (ValueError, KeyError) as exc:
        print(f"  SKIP: {exc}")
        return "skipped"

    for m in unpaired_opens:
        print(f"  WARNING: unpaired open at {m.start_seconds:.1f}s")
    for m in unpaired_closes:
        print(f"  WARNING: unpaired close at {m.start_seconds:.1f}s")

    if not segments:
        print("  No ad segments detected — nothing to cut.")
        return "skipped"

    try:
        _validate_segments(segments)
    except ValueError as exc:
        print(f"  SKIP: {exc}")
        return "skipped"

    sorted_segs = sorted(segments, key=lambda s: s.cut_start)
    print(f"\n  {len(sorted_segs)} ad segment(s):")
    for i, seg in enumerate(sorted_segs, 1):
        print(
            f"    {i}. [{seg.cut_start:.1f}s → {seg.cut_end:.1f}s]  "
            f"({seg.cut_end - seg.cut_start:.1f}s)"
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
    return "cut"


# ---------------------------------------------------------------------------
# review subcommand
# ---------------------------------------------------------------------------


def _detect_batch_job(
    source: Path,
    sample: Path,
    kind: str,
    *,
    threshold: float,
    z_threshold: float | None,
    step_seconds: float,
    max_matches: int,
) -> tuple[str, str, list[MatchCandidate]]:
    candidates = _detect_matches(
        source,
        sample,
        threshold=threshold,
        z_threshold=z_threshold,
        step_seconds=step_seconds,
        max_matches=max_matches,
    )
    return source.stem, kind, candidates


def _cmd_review(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    review_root: Path = args.review_root
    open_sample = args.snippets_dir / args.open_sample
    close_sample = args.snippets_dir / args.close_sample
    state_path = review_root / "state.toml"

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

    state = PipelineState.load(state_path)
    open_threshold = state.open_threshold
    close_threshold = state.close_threshold

    episodes: list[Path] = []
    n_already_done = 0
    for ep in all_full:
        if not args.overwrite and state.episode(ep.stem).is_labeled():
            n_already_done += 1
        else:
            episodes.append(ep)

    print(f"Episodes to process: {len(episodes)}  ({n_already_done} already labeled, skipped)")
    if not episodes:
        print("All episodes already labeled. Use --overwrite to re-run.")
        return

    session_scores = SessionScores()

    for batch_num, batch in enumerate(_chunks(episodes, args.batch_size), 1):
        start_idx = (batch_num - 1) * args.batch_size + 1
        end_idx = min(batch_num * args.batch_size, len(episodes))
        _emit(f"\nBatch {batch_num}: episodes {start_idx}–{end_idx} of {len(episodes)}")

        to_detect = [
            ep for ep in batch if args.overwrite or not state.episode(ep.stem).has_matches()
        ]

        if to_detect:
            jobs = [(ep, open_sample, "open", open_threshold) for ep in to_detect] + [
                (ep, close_sample, "close", close_threshold) for ep in to_detect
            ]
            done = 0
            n_jobs = len(jobs)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        _detect_batch_job,
                        src,
                        smp,
                        kind,
                        threshold=thresh,
                        z_threshold=args.z_threshold,
                        step_seconds=args.step_seconds,
                        max_matches=args.max_matches,
                    ): (src.stem, kind)
                    for src, smp, kind, thresh in jobs
                }
                for future in as_completed(futures):
                    done += 1
                    ep_stem, ep_kind, candidates = future.result()
                    ep_state = state.episode(ep_stem)
                    ep_state.source = str(remote_dir / f"{ep_stem}.mp3")
                    if ep_kind == "open":
                        ep_state.open_matches = candidates
                    else:
                        ep_state.close_matches = candidates
                    _emit(
                        f"  [{done}/{n_jobs}] {ep_kind:5}  {ep_stem}  ({len(candidates)} matches)"
                    )
            state.save(state_path)
        else:
            _emit(f"  All {len(batch)} episode(s) already detected — going straight to review.")

        if not args.no_interactive:
            for ep in batch:
                print(f"\n{'=' * 60}")
                print(f"Episode: {ep.stem}")
                print("=" * 60)
                _interactive_review_episode(
                    ep,
                    session_scores,
                    state,
                    open_sample=open_sample,
                    close_sample=close_sample,
                )

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
        else:
            print("\nMatches detected. Run remote-review without --no-interactive to label them.")


# ---------------------------------------------------------------------------
# cut subcommand
# ---------------------------------------------------------------------------


def _cmd_cut(args: argparse.Namespace) -> None:
    review_root: Path = args.review_root
    remote_dir: Path = args.remote_dir
    output_dir: Path = args.output_dir
    state_path = review_root / "state.toml"

    state = PipelineState.load(state_path)

    labeled = {stem: ep for stem, ep in state.episodes.items() if ep.is_labeled() and not ep.cut}
    if not labeled:
        print("No labeled episodes in state.toml. Run remote-review first.")
        return

    print(f"Found {len(labeled)} labeled episode(s).")
    n_cut = n_skipped = n_failed = 0

    for stem, ep_state in sorted(labeled.items()):
        source = Path(ep_state.source) if ep_state.source else remote_dir / f"{stem}.mp3"
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
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            yes=args.yes,
            dry_run=args.dry_run,
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
# loop subcommand — detect → review → cut per episode in one pass
# ---------------------------------------------------------------------------


def _cmd_loop(args: argparse.Namespace) -> None:
    """One episode at a time: detect matches → review → cut → next."""
    remote_dir: Path = args.remote_dir
    review_root: Path = args.review_root
    output_dir: Path = args.output_dir
    open_sample = args.snippets_dir / args.open_sample
    close_sample = args.snippets_dir / args.close_sample
    state_path = review_root / "state.toml"

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
    n_cut = n_skipped = n_failed = 0

    for ep_num, ep in enumerate(episodes, 1):
        print(f"\n{'=' * 60}")
        print(f"[{ep_num}/{len(episodes)}] {ep.stem}")
        print("=" * 60)

        ep_state = state.episode(ep.stem)
        ep_state.source = str(ep)
        already_labeled = ep_state.is_labeled()

        if args.overwrite or not ep_state.has_matches():
            _emit("  Detecting matches...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_open = pool.submit(
                    _detect_matches,
                    ep,
                    open_sample,
                    threshold=open_threshold,
                    z_threshold=args.z_threshold,
                    step_seconds=args.step_seconds,
                    max_matches=args.max_matches,
                )
                f_close = pool.submit(
                    _detect_matches,
                    ep,
                    close_sample,
                    threshold=close_threshold,
                    z_threshold=args.z_threshold,
                    step_seconds=args.step_seconds,
                    max_matches=args.max_matches,
                )
                ep_state.open_matches = f_open.result()
                ep_state.close_matches = f_close.result()
                _emit(
                    f"  open: {len(ep_state.open_matches)} match(es)  "
                    f"close: {len(ep_state.close_matches)} match(es)"
                )
            state.save(state_path)

        needs_review = not args.no_interactive and (args.overwrite or not already_labeled)
        if needs_review:
            _interactive_review_episode(
                ep,
                session_scores,
                state,
                open_sample=open_sample,
                close_sample=close_sample,
            )
            floor_open = session_scores.open_floor()
            floor_close = session_scores.close_floor()
            if floor_open is not None:
                open_threshold = max(open_threshold, floor_open)
            if floor_close is not None:
                close_threshold = max(close_threshold, floor_close)
            state.open_threshold = open_threshold
            state.close_threshold = close_threshold
        elif already_labeled:
            print("  Already labeled — skipping review.")

        source = Path(ep_state.source)
        result = _pair_and_cut(
            ep.stem,
            source,
            output_dir=output_dir,
            ep_state=ep_state,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            yes=args.yes,
            dry_run=args.dry_run,
        )
        if result == "cut":
            ep_state.cut = True
            n_cut += 1
        elif result == "failed":
            n_failed += 1
        else:
            n_skipped += 1

        state.save(state_path)

    print(f"\nDone: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Match score floor (adapts upward as you approve)",
    )
    p.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score cutoff: keep scores >= mean + N*std (default: 3.0)",
    )
    p.add_argument("--step-seconds", type=float, default=0.1)
    p.add_argument(
        "--max-matches",
        type=int,
        default=10,
        help="Max candidates per snippet type per episode (default: 10)",
    )


def _add_cut_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--min-gap",
        type=float,
        default=-15.0,
        help="Min seconds between open end and close start (default: -15)",
    )
    p.add_argument("--max-gap", type=float, default=600.0)
    p.add_argument("--yes", action="store_true", help="Cut without confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="Show cut plan without running ffmpeg")


def _build_review_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("review", help="Detect matches and review them interactively")
    p.add_argument("--remote-dir", type=Path, default=Path("downloads/remote"))
    p.add_argument("--snippets-dir", type=Path, default=Path("downloads/snippets"))
    p.add_argument("--open-sample", default="open.mp3")
    p.add_argument("--close-sample", default="close.mp3")
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    _add_detect_args(p)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no-interactive", action="store_true", help="Detect only — skip interactive review"
    )


def _build_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("cut", help="Cut ad segments using labels from state.toml")
    p.add_argument("--remote-dir", type=Path, default=Path("downloads/remote"))
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    p.add_argument("--output-dir", type=Path, default=Path("downloads/remove"))
    _add_cut_args(p)


def _build_loop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "loop",
        help="Detect → review → cut one episode at a time until done",
    )
    p.add_argument("--remote-dir", type=Path, default=Path("downloads/remote"))
    p.add_argument("--snippets-dir", type=Path, default=Path("downloads/snippets"))
    p.add_argument("--open-sample", default="open.mp3")
    p.add_argument("--close-sample", default="close.mp3")
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    p.add_argument("--output-dir", type=Path, default=Path("downloads/remove"))
    _add_detect_args(p)
    _add_cut_args(p)
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive review (use existing labels)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-detect, re-review, and re-cut already-processed episodes",
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
