"""Remote episode pipeline: detect ad-break positions, review them, cut them out.

Subcommands:
  review  — detect open/close matches per episode, review each interactively
  cut     — use state.toml to pair and cut ad segments
  loop    — detect → review → cut one episode at a time until done

State is stored entirely in {review_root}/state.toml.
No clip files or manifest CSVs are written.
Delete state.toml to start fresh.
"""

from __future__ import annotations

import argparse
import json
import math
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

# Classification labels
_POS = "positive"
_NEG = "negative"
_UNC = "uncertain"
_UND = "undetected"


# ---------------------------------------------------------------------------
# State model — persisted to {review_root}/state.toml
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A confirmed example of an audio target at a specific position."""

    source: str
    start: float
    end: float
    score: float


@dataclass
class TargetState:
    """Global confirmed examples for one target type (open or close)."""

    positives: list[Segment] = field(default_factory=list)
    negatives: list[Segment] = field(default_factory=list)


@dataclass
class EpisodeState:
    source: str = ""
    open_score: float = 0.0
    open_start: float = 0.0
    open_end: float = 0.0
    open_class: str = _UND
    close_score: float = 0.0
    close_start: float = 0.0
    close_end: float = 0.0
    close_class: str = _UND
    cut: bool = False

    def is_detected(self) -> bool:
        return self.open_class != _UND or self.close_class != _UND

    def is_cuttable(self) -> bool:
        return self.open_class == _POS and self.close_class == _POS


def _fmt_seg(s: Segment) -> str:
    return (
        f"{{source = {json.dumps(s.source)}, "
        f"start = {s.start:.3f}, end = {s.end:.3f}, score = {s.score:.6f}}}"
    )


def _load_target(raw: dict) -> TargetState:
    t = TargetState()
    for seg in raw.get("positives", []):
        t.positives.append(
            Segment(
                source=str(seg["source"]),
                start=float(seg["start"]),
                end=float(seg["end"]),
                score=float(seg["score"]),
            )
        )
    for seg in raw.get("negatives", []):
        t.negatives.append(
            Segment(
                source=str(seg["source"]),
                start=float(seg["start"]),
                end=float(seg["end"]),
                score=float(seg["score"]),
            )
        )
    return t


def _load_episode(raw: dict) -> EpisodeState:
    return EpisodeState(
        source=str(raw.get("source", "")),
        open_score=float(raw.get("open_score", 0.0)),
        open_start=float(raw.get("open_start", 0.0)),
        open_end=float(raw.get("open_end", 0.0)),
        open_class=str(raw.get("open_class", _UND)),
        close_score=float(raw.get("close_score", 0.0)),
        close_start=float(raw.get("close_start", 0.0)),
        close_end=float(raw.get("close_end", 0.0)),
        close_class=str(raw.get("close_class", _UND)),
        cut=bool(raw.get("cut", False)),
    )


def _migrate_episode_keys(path: Path) -> None:
    """Re-quote any unquoted [episodes.*] table keys written by older versions."""
    text = path.read_text(encoding="utf-8")
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


def _migrate_old_episode(
    ep_raw: dict,
    source: str,
    open_target: TargetState,
    close_target: TargetState,
) -> EpisodeState:
    """Convert one old-format episode (match lists + approved/rejected) to new format."""
    ep = EpisodeState(source=source, cut=bool(ep_raw.get("cut", False)))
    for kind, matches_key, approved_key, rejected_key, target in [
        ("open", "open_matches", "open_approved", "open_rejected", open_target),
        ("close", "close_matches", "close_approved", "close_rejected", close_target),
    ]:
        raw_matches = ep_raw.get(matches_key, [])
        approved = list(ep_raw.get(approved_key, []))
        rejected = list(ep_raw.get(rejected_key, []))
        approved_set = frozenset(approved)
        best_raw = None
        if approved:
            best_raw = next((m for m in raw_matches if m["index"] in approved_set), None)
        if best_raw is None and raw_matches:
            best_raw = raw_matches[0]
        if best_raw is None:
            setattr(ep, f"{kind}_class", _UND)
            continue
        score = float(best_raw["score"])
        start = float(best_raw["start"])
        end = float(best_raw["end"])
        setattr(ep, f"{kind}_score", score)
        setattr(ep, f"{kind}_start", start)
        setattr(ep, f"{kind}_end", end)
        if approved:
            setattr(ep, f"{kind}_class", _POS)
            target.positives.append(Segment(source=source, start=start, end=end, score=score))
        elif rejected:
            setattr(ep, f"{kind}_class", _NEG)
            target.negatives.append(Segment(source=source, start=start, end=end, score=score))
        else:
            setattr(ep, f"{kind}_class", _UNC)
    return ep


def _migrate_old_state(data: dict) -> "PipelineState":
    _emit("Migrating state.toml to new format (one-time conversion)...")
    state = PipelineState()
    for stem, ep_raw in data.get("episodes", {}).items():
        source = str(ep_raw.get("source", ""))
        state.episodes[stem] = _migrate_old_episode(
            ep_raw, source, state.open_target, state.close_target
        )
    return state


@dataclass
class PipelineState:
    open_target: TargetState = field(default_factory=TargetState)
    close_target: TargetState = field(default_factory=TargetState)
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
            _emit("Warning: TOML unavailable. Starting with empty state.")
            return cls()
        _migrate_episode_keys(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        episodes_raw = data.get("episodes", {})
        if any("open_matches" in ep for ep in episodes_raw.values()):
            return _migrate_old_state(data)
        state = cls(
            open_target=_load_target(data.get("targets", {}).get("open", {})),
            close_target=_load_target(data.get("targets", {}).get("close", {})),
        )
        for stem, ep_raw in episodes_raw.items():
            state.episodes[stem] = _load_episode(ep_raw)
        return state

    def save(self, path: Path) -> None:
        lines: list[str] = [
            "# Remote episode pipeline state.\n",
            "# Edit freely — delete this file to start fresh.\n",
        ]
        for kind, target in [("open", self.open_target), ("close", self.close_target)]:
            pos = ", ".join(_fmt_seg(s) for s in target.positives)
            neg = ", ".join(_fmt_seg(s) for s in target.negatives)
            lines += [f"\n[targets.{kind}]\n", f"positives = [{pos}]\n", f"negatives = [{neg}]\n"]
        for stem, ep in sorted(self.episodes.items()):
            lines.append(f'\n[episodes."{stem}"]\n')
            if ep.source:
                lines.append(f"source      = {json.dumps(ep.source)}\n")
            lines += [
                f"open_score  = {ep.open_score:.6f}\n",
                f"open_start  = {ep.open_start:.3f}\n",
                f"open_end    = {ep.open_end:.3f}\n",
                f'open_class  = "{ep.open_class}"\n',
                f"close_score = {ep.close_score:.6f}\n",
                f"close_start = {ep.close_start:.3f}\n",
                f"close_end   = {ep.close_end:.3f}\n",
                f'close_class = "{ep.close_class}"\n',
                f"cut = {str(ep.cut).lower()}\n",
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _moe(scores: list[float], k: float = 1.5) -> float:
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return k * math.sqrt(variance)


def _compute_thresholds(target: TargetState, default_floor: float) -> tuple[float, float]:
    """Return (theta_plus, theta_minus) for a target."""
    pos = [s.score for s in target.positives]
    neg = [s.score for s in target.negatives]
    theta_plus = (min(pos) - _moe(pos)) if pos else default_floor
    theta_minus = (max(neg) + _moe(neg)) if neg else -math.inf
    return theta_plus, theta_minus


def _classify_score(score: float, theta_plus: float, theta_minus: float) -> str:
    if score >= theta_plus:
        return _POS
    if score <= theta_minus:
        return _NEG
    return _UNC


def _reclassify_all(state: PipelineState, default_floor: float) -> None:
    """Recompute thresholds and re-classify uncertain episodes in-place.

    Auto-classification only runs when there is at least one confirmed positive
    or negative for a target type — without confirmed examples the threshold is
    meaningless and episodes stay uncertain until the user labels one explicitly.
    """
    tp_o, tm_o = _compute_thresholds(state.open_target, default_floor)
    tp_c, tm_c = _compute_thresholds(state.close_target, default_floor)
    has_open_evidence = bool(state.open_target.positives or state.open_target.negatives)
    has_close_evidence = bool(state.close_target.positives or state.close_target.negatives)
    for ep in state.episodes.values():
        if ep.open_class == _UNC and has_open_evidence:
            ep.open_class = _classify_score(ep.open_score, tp_o, tm_o)
        if ep.close_class == _UNC and has_close_evidence:
            ep.close_class = _classify_score(ep.close_score, tp_c, tm_c)


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
# Detection — runs audio_detect as subprocess, returns best match
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Match:
    score: float
    start: float
    end: float


def _detect_best(
    source: Path,
    sample: Path,
    *,
    threshold: float,
    z_threshold: float | None,
    step_seconds: float,
) -> _Match | None:
    """Run audio_detect subprocess; return the top-scoring match or None."""
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
        "1",
    ]
    if z_threshold is not None:
        command.extend(["--z-threshold", str(z_threshold)])
    result = run_resolved(command, capture_output=True)
    if result.returncode != 0:
        if result.stderr:
            sys.stderr.buffer.write(result.stderr)
            sys.stderr.flush()
        return None
    try:
        data = json.loads(result.stdout)
        if not data:
            return None
        m = data[0]
        return _Match(score=float(m["score"]), start=float(m["start"]), end=float(m["end"]))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _emit(f"  WARNING: failed to parse detect output: {exc}")
        return None


def _detect_batch(
    episodes: list[Path],
    state: PipelineState,
    open_sample: Path,
    close_sample: Path,
    *,
    threshold: float,
    z_threshold: float | None,
    step_seconds: float,
    workers: int,
) -> None:
    """Detect open+close for a batch of episodes in parallel, updating state in-place."""
    ep_by_stem = {ep.stem: ep for ep in episodes}
    jobs = [(ep, open_sample, "open") for ep in episodes] + [
        (ep, close_sample, "close") for ep in episodes
    ]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _detect_best,
                ep,
                sample,
                threshold=threshold,
                z_threshold=z_threshold,
                step_seconds=step_seconds,
            ): (ep.stem, kind)
            for ep, sample, kind in jobs
        }
        for future in as_completed(futures):
            done += 1
            stem, kind = futures[future]
            match = future.result()
            ep_state = state.episode(stem)
            ep_state.source = str(ep_by_stem[stem])
            score_str = f"{match.score:.4f}" if match else "none"
            if kind == "open":
                if match:
                    ep_state.open_score = match.score
                    ep_state.open_start = match.start
                    ep_state.open_end = match.end
                    ep_state.open_class = _UNC
                else:
                    ep_state.open_class = _UND
            else:
                if match:
                    ep_state.close_score = match.score
                    ep_state.close_start = match.start
                    ep_state.close_end = match.end
                    ep_state.close_class = _UNC
                else:
                    ep_state.close_class = _UND
            _emit(f"  [{done}/{len(jobs)}] {kind:5}  {stem}  ({score_str})")


# ---------------------------------------------------------------------------
# Interactive review
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
    return launch_resolved(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])


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
class _UndoEntry:
    stem: str
    kind: str
    action: str
    segment: Segment
    target_list: list[Segment]


def _next_uncertain(
    state: PipelineState, exclude: set[tuple[str, str]] | None = None
) -> tuple[str, str] | None:
    """Return (stem, kind) of the highest-scoring uncertain target, or None."""
    candidates = []
    for stem, ep in state.episodes.items():
        not_skipped_open = exclude is None or ("open", stem) not in exclude
        not_skipped_close = exclude is None or ("close", stem) not in exclude
        if ep.open_class == _UNC and ep.open_score > 0 and not_skipped_open:
            candidates.append((ep.open_score, stem, "open"))
        if ep.close_class == _UNC and ep.close_score > 0 and not_skipped_close:
            candidates.append((ep.close_score, stem, "close"))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, stem, kind = candidates[0]
    return stem, kind


def _review_one_target(
    state: PipelineState,
    stem: str,
    kind: str,
    *,
    open_sample: Path,
    close_sample: Path,
    history: list[_UndoEntry],
    default_floor: float,
) -> str:
    """Review one uncertain target interactively. Returns 'classified', 'skipped', or 'undone'."""
    ep_state = state.episodes[stem]
    source = Path(ep_state.source)
    snippet = open_sample if kind == "open" else close_sample
    target = state.open_target if kind == "open" else state.close_target
    score = ep_state.open_score if kind == "open" else ep_state.close_score
    start = ep_state.open_start if kind == "open" else ep_state.close_start
    end = ep_state.open_end if kind == "open" else ep_state.close_end

    undo_hint = "  [u]ndo" if history else ""
    legend = f"  [a]pprove  [r]eject  [p]replay  [c]ompare  [s]kip  [q]uit{undo_hint}  "
    print(f"\n  [{kind}]  score={score:.4f}  start={start:.1f}s", file=sys.stderr)
    print(legend, end="", flush=True, file=sys.stderr)
    current_proc: Any | None = _start_audio_segment(source, start, end)

    while True:
        key = _getch().lower()
        _stop_audio(current_proc)
        current_proc = None

        if key == "p":
            current_proc = _start_audio_segment(source, start, end)
            print(f"\r{legend}", end="", flush=True, file=sys.stderr)
        elif key == "c":
            current_proc = _start_audio(snippet)
            print(f"\r{legend}", end="", flush=True, file=sys.stderr)
        elif key in ("a", "r"):
            seg = Segment(source=str(source), start=start, end=end, score=score)
            if key == "a":
                target.positives.append(seg)
                ep_class, lst = _POS, target.positives
                print("\n✓ approved", file=sys.stderr)
            else:
                target.negatives.append(seg)
                ep_class, lst = _NEG, target.negatives
                print("\n✗ rejected", file=sys.stderr)
            if kind == "open":
                ep_state.open_class = ep_class
            else:
                ep_state.close_class = ep_class
            history.append(
                _UndoEntry(stem=stem, kind=kind, action=key, segment=seg, target_list=lst)
            )
            _reclassify_all(state, default_floor)
            return "classified"
        elif key == "s":
            print("\n— skipped", file=sys.stderr)
            return "skipped"
        elif key == "u" and history:
            entry = history.pop()
            entry.target_list.remove(entry.segment)
            prev_ep = state.episodes[entry.stem]
            if entry.kind == "open":
                prev_ep.open_class = _UNC
            else:
                prev_ep.close_class = _UNC
            _reclassify_all(state, default_floor)
            print(
                f"\n↩ undone ({entry.action} {entry.kind} for {entry.stem[:16]})",
                file=sys.stderr,
            )
            return "undone"
        elif key == "q":
            print("\nQuitting review.", file=sys.stderr)
            raise KeyboardInterrupt


def _count_uncertain(state: PipelineState) -> int:
    return sum(
        1
        for ep in state.episodes.values()
        for cls in (ep.open_class, ep.close_class)
        if cls == _UNC
    )


def _run_review_loop(
    state: PipelineState,
    *,
    open_sample: Path,
    close_sample: Path,
    default_floor: float,
    state_path: Path,
    max_decisions: int | None = None,
) -> None:
    """Review uncertain targets until the queue is empty, the user quits, or max_decisions reached."""
    history: list[_UndoEntry] = []
    skipped: set[tuple[str, str]] = set()
    decisions = 0
    while max_decisions is None or decisions < max_decisions:
        next_t = _next_uncertain(state, exclude=skipped)
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
            history=history,
            default_floor=default_floor,
        )
        if result == "classified":
            decisions += 1
            skipped.discard((kind, stem))
            state.save(state_path)
        elif result == "skipped":
            skipped.add((kind, stem))
        else:  # undone
            state.save(state_path)
        del ep_state  # avoid unused-variable lint
    if max_decisions is not None and decisions >= max_decisions:
        n_unc = _count_uncertain(state)
        if n_unc:
            _emit(f"\nBatch complete ({decisions} decisions). {n_unc} uncertain remaining — run again.")


# ---------------------------------------------------------------------------
# Cut helpers
# ---------------------------------------------------------------------------


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
    """Pair open/close from ep_state and cut. Returns 'cut', 'skipped', or 'failed'."""
    if not ep_state.is_cuttable():
        print(f"  SKIP {stem}: open and close must both be classified as positive.")
        return "skipped"

    open_match = AudioMatch(
        start_seconds=ep_state.open_start,
        end_seconds=ep_state.open_end,
        duration_seconds=ep_state.open_end - ep_state.open_start,
        score=ep_state.open_score,
    )
    close_match = AudioMatch(
        start_seconds=ep_state.close_start,
        end_seconds=ep_state.close_end,
        duration_seconds=ep_state.close_end - ep_state.close_start,
        score=ep_state.close_score,
    )

    try:
        segments, unpaired_opens, unpaired_closes = pair_ad_segments(
            [open_match], [close_match], min_gap=min_gap, max_gap=max_gap
        )
    except (ValueError, KeyError) as exc:
        print(f"  SKIP: {exc}")
        return "skipped"

    for m in unpaired_opens:
        print(f"  WARNING: unpaired open at {m.start_seconds:.1f}s")
    for m in unpaired_closes:
        print(f"  WARNING: unpaired close at {m.start_seconds:.1f}s")

    if not segments:
        print("  No paired ad segments — nothing to cut.")
        return "skipped"

    try:
        _validate_segments(segments)
    except ValueError as exc:
        print(f"  SKIP: {exc}")
        return "skipped"

    seg = segments[0]
    duration = seg.cut_end - seg.cut_start
    print(f"\n  1 ad segment: [{seg.cut_start:.1f}s → {seg.cut_end:.1f}s]  ({duration:.1f}s)")

    if dry_run:
        return "skipped"

    if not yes:
        resp = input(f"\n  Cut ad from {stem}? [y/N] ").strip().lower()
        if resp != "y":
            print("  Skipped.")
            return "skipped"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.mp3"
    spans = _build_keep_spans(segments)
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
            threshold=args.threshold,
            z_threshold=args.z_threshold,
            step_seconds=args.step_seconds,
            workers=args.workers,
        )
        _reclassify_all(state, args.threshold)
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
        default_floor=args.threshold,
        state_path=state_path,
    )


# ---------------------------------------------------------------------------
# cut subcommand
# ---------------------------------------------------------------------------


def _cmd_cut(args: argparse.Namespace) -> None:
    review_root: Path = args.review_root
    remote_dir: Path = args.remote_dir
    output_dir: Path = args.output_dir
    state_path = review_root / "state.toml"

    state = PipelineState.load(state_path)
    cuttable = {stem: ep for stem, ep in state.episodes.items() if ep.is_cuttable() and not ep.cut}
    if not cuttable:
        print("No cuttable episodes in state.toml — need open and close both positive.")
        return

    print(f"Found {len(cuttable)} cuttable episode(s).")
    n_cut = n_skipped = n_failed = 0

    for stem, ep_state in sorted(cuttable.items()):
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
# loop subcommand — detect → review → cut in one pass
# ---------------------------------------------------------------------------


def _cmd_loop(args: argparse.Namespace) -> None:
    """Detect all undetected → review all uncertain → cut all cuttable."""
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
    to_detect = [
        ep for ep in all_full if args.overwrite or not state.episode(ep.stem).is_detected()
    ]
    if to_detect:
        _emit(f"\nDetecting {len(to_detect)} episode(s)...")
        _detect_batch(
            to_detect,
            state,
            open_sample,
            close_sample,
            threshold=args.threshold,
            z_threshold=args.z_threshold,
            step_seconds=args.step_seconds,
            workers=args.workers,
        )
        _reclassify_all(state, args.threshold)
        state.save(state_path)

    if not args.no_interactive:
        n_unc = _count_uncertain(state)
        _emit(f"\n{n_unc} uncertain target(s) to review.")
        if n_unc:
            _run_review_loop(
                state,
                open_sample=open_sample,
                close_sample=close_sample,
                default_floor=args.threshold,
                state_path=state_path,
            )

    cuttable = [
        (stem, ep) for stem, ep in state.episodes.items() if ep.is_cuttable() and not ep.cut
    ]
    if not cuttable:
        print("\nNo cuttable episodes.")
        return

    print(f"\nCutting {len(cuttable)} episode(s)...")
    n_cut = n_skipped = n_failed = 0
    for stem, ep_state in sorted(cuttable):
        source = Path(ep_state.source) if ep_state.source else remote_dir / f"{stem}.mp3"
        if not source.exists():
            print(f"SKIP {stem}: source not found")
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
# Argument parsers
# ---------------------------------------------------------------------------


def _add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--threshold", type=float, default=0.8, help="Initial theta_plus floor")
    p.add_argument("--z-threshold", type=float, default=3.0, help="Z-score cutoff (mean + N*std)")
    p.add_argument("--step-seconds", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2, help="Parallel workers for detection")


def _add_cut_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-gap", type=float, default=-15.0, help="Min gap: open end → close start")
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
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-interactive", action="store_true", help="Detect only, skip review")


def _build_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("cut", help="Cut ad segments using labels from state.toml")
    p.add_argument("--remote-dir", type=Path, default=Path("downloads/remote"))
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    p.add_argument("--output-dir", type=Path, default=Path("downloads/remove"))
    _add_cut_args(p)


def _build_loop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("loop", help="Detect → review → cut until done")
    p.add_argument("--remote-dir", type=Path, default=Path("downloads/remote"))
    p.add_argument("--snippets-dir", type=Path, default=Path("downloads/snippets"))
    p.add_argument("--open-sample", default="open.mp3")
    p.add_argument("--close-sample", default="close.mp3")
    p.add_argument("--review-root", type=Path, default=Path("downloads/review"))
    p.add_argument("--output-dir", type=Path, default=Path("downloads/remove"))
    _add_detect_args(p)
    _add_cut_args(p)
    p.add_argument("--no-interactive", action="store_true", help="Skip interactive review")
    p.add_argument("--overwrite", action="store_true", help="Re-detect and re-cut episodes")


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
