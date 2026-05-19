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
import json
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from part_io.adapters.audio.ad_segments import pair_ad_segments
from part_io.adapters.audio.matcher import AudioMatch
from part_io.adapters.process.runner import run_resolved
from part_io.cli.audio_ad_remove import (
    _build_filter_complex,
    _run_ffmpeg,
    _spans_from_cuts,
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
_STATE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "schemas" / "remote_pipeline_state.schema.json"
)
_AUDIO_IO: dict[int, tuple[Any, Any, Any]] = {}

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


@dataclass(frozen=True)
class _Match:
    score: float
    start: float
    end: float


@dataclass
class EpisodeState:
    source: str = ""
    open_candidates: list[_Match] = field(default_factory=list)
    open_class: str = _UND
    close_candidates: list[_Match] = field(default_factory=list)
    close_class: str = _UND
    intro_candidates: list[_Match] = field(default_factory=list)
    intro_class: str = _UND
    cut: bool = False

    # Read-only convenience properties so all existing callers of open_score etc. still work.
    @property
    def open_score(self) -> float:
        return self.open_candidates[0].score if self.open_candidates else 0.0

    @property
    def open_start(self) -> float:
        return self.open_candidates[0].start if self.open_candidates else 0.0

    @property
    def open_end(self) -> float:
        return self.open_candidates[0].end if self.open_candidates else 0.0

    @property
    def close_score(self) -> float:
        return self.close_candidates[0].score if self.close_candidates else 0.0

    @property
    def close_start(self) -> float:
        return self.close_candidates[0].start if self.close_candidates else 0.0

    @property
    def close_end(self) -> float:
        return self.close_candidates[0].end if self.close_candidates else 0.0

    @property
    def intro_score(self) -> float:
        return self.intro_candidates[0].score if self.intro_candidates else 0.0

    @property
    def intro_start(self) -> float:
        return self.intro_candidates[0].start if self.intro_candidates else 0.0

    @property
    def intro_end(self) -> float:
        return self.intro_candidates[0].end if self.intro_candidates else 0.0

    def is_detected(self) -> bool:
        return self.open_class != _UND or self.close_class != _UND or self.intro_class != _UND

    def is_cuttable(self) -> bool:
        return self.open_class == _POS and self.close_class == _POS


@dataclass
class RunSettings:
    z_threshold: float | None = None
    step_seconds: float = 0.1
    workers: int = 2
    max_matches: int = 3
    min_gap: float = -15.0
    max_gap: float = 300.0
    yes: bool = False
    dry_run: bool = False
    inclusive: bool = False
    fade: float = 0.5
    quiz_size: int = 10
    no_interactive: bool = False
    overwrite: bool = False
    snippets_dir: str = "downloads/snippets"
    open_sample: str = "open.mp3"
    close_sample: str = "close.mp3"
    intro_sample: str = "intro.mp3"
    output_dir: str = "downloads/remove"
    debug: bool = False


def _fmt_seg(s: Segment) -> str:
    return (
        f"{{source = {json.dumps(s.source)}, "
        f"start = {s.start:.3f}, end = {s.end:.3f}, score = {s.score:.6f}}}"
    )


def _fmt_match(m: _Match) -> str:
    return f"{{score = {m.score:.6f}, start = {m.start:.3f}, end = {m.end:.3f}}}"


def _load_match(raw: dict) -> _Match:
    return _Match(score=float(raw["score"]), start=float(raw["start"]), end=float(raw["end"]))


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
    ep = EpisodeState(
        source=str(raw.get("source", "")),
        open_class=str(raw.get("open_class", _UND)),
        close_class=str(raw.get("close_class", _UND)),
        intro_class=str(raw.get("intro_class", _UND)),
        cut=bool(raw.get("cut", False)),
    )
    # New format: candidates list
    if "open_candidates" in raw:
        ep.open_candidates = [_load_match(m) for m in raw["open_candidates"]]
    elif float(raw.get("open_score", 0.0)) > 0:
        # Migrate old scalar format (open_score / open_start / open_end)
        ep.open_candidates = [
            _Match(
                score=float(raw["open_score"]),
                start=float(raw.get("open_start", 0.0)),
                end=float(raw.get("open_end", 0.0)),
            )
        ]
    if "close_candidates" in raw:
        ep.close_candidates = [_load_match(m) for m in raw["close_candidates"]]
    elif float(raw.get("close_score", 0.0)) > 0:
        ep.close_candidates = [
            _Match(
                score=float(raw["close_score"]),
                start=float(raw.get("close_start", 0.0)),
                end=float(raw.get("close_end", 0.0)),
            )
        ]
    if "intro_candidates" in raw:
        ep.intro_candidates = [_load_match(m) for m in raw["intro_candidates"]]
    return ep


def _load_settings(raw: dict) -> RunSettings:
    return RunSettings(
        z_threshold=float(raw["z_threshold"]) if raw.get("z_threshold") is not None else None,
        step_seconds=float(raw.get("step_seconds", 0.1)),
        workers=int(raw.get("workers", 2)),
        max_matches=int(raw.get("max_matches", 3)),
        min_gap=float(raw.get("min_gap", -15.0)),
        max_gap=float(raw.get("max_gap", 300.0)),
        yes=bool(raw.get("yes", False)),
        dry_run=bool(raw.get("dry_run", False)),
        inclusive=bool(raw.get("inclusive", False)),
        fade=float(raw.get("fade", 0.5)),
        quiz_size=int(raw.get("quiz_size", 10)),
        no_interactive=bool(raw.get("no_interactive", False)),
        overwrite=bool(raw.get("overwrite", False)),
        snippets_dir=str(raw.get("snippets_dir", "downloads/snippets")),
        open_sample=str(raw.get("open_sample", "open.mp3")),
        close_sample=str(raw.get("close_sample", "close.mp3")),
        intro_sample=str(raw.get("intro_sample", "intro.mp3")),
        output_dir=str(raw.get("output_dir", "downloads/remove")),
        debug=bool(raw.get("debug", False)),
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
        cands = [_Match(score=score, start=start, end=end)]
        if kind == "open":
            ep.open_candidates = cands
        else:
            ep.close_candidates = cands
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
    settings: RunSettings = field(default_factory=RunSettings)
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
            settings=_load_settings(data.get("settings", {})),
        )
        for stem, ep_raw in episodes_raw.items():
            state.episodes[stem] = _load_episode(ep_raw)
        return state

    def save(self, path: Path) -> None:
        try:
            schema_ref = os.path.relpath(_STATE_SCHEMA_PATH, path.parent)
            schema_ref = schema_ref.replace("\\", "/")
        except ValueError:
            schema_ref = _STATE_SCHEMA_PATH.resolve().as_uri()

        lines: list[str] = [
            f"#:schema {schema_ref}\n",
            "# Remote episode pipeline state.\n",
            "# Edit freely — delete this file to start fresh.\n",
        ]
        settings = self.settings
        setting_lines = ["\n[settings]\n"]
        if settings.z_threshold is not None:
            setting_lines.append(f"z_threshold = {settings.z_threshold:.6g}\n")
        setting_lines += [
            f"step_seconds = {settings.step_seconds:.6g}\n",
            f"workers = {settings.workers}\n",
            f"max_matches = {settings.max_matches}\n",
            f"min_gap = {settings.min_gap:.6g}\n",
            f"max_gap = {settings.max_gap:.6g}\n",
            f"yes = {str(settings.yes).lower()}\n",
            f"dry_run = {str(settings.dry_run).lower()}\n",
            f"inclusive = {str(settings.inclusive).lower()}\n",
            f"fade = {settings.fade:.6g}\n",
            f"quiz_size = {settings.quiz_size}\n",
            f"no_interactive = {str(settings.no_interactive).lower()}\n",
            f"overwrite = {str(settings.overwrite).lower()}\n",
            f"snippets_dir = {json.dumps(settings.snippets_dir)}\n",
            f"open_sample = {json.dumps(settings.open_sample)}\n",
            f"close_sample = {json.dumps(settings.close_sample)}\n",
            f"intro_sample = {json.dumps(settings.intro_sample)}\n",
            f"output_dir = {json.dumps(settings.output_dir)}\n",
            f"debug = {str(settings.debug).lower()}\n",
        ]
        lines += setting_lines
        for kind, target in [("open", self.open_target), ("close", self.close_target)]:
            pos = ", ".join(_fmt_seg(s) for s in target.positives)
            neg = ", ".join(_fmt_seg(s) for s in target.negatives)
            lines += [f"\n[targets.{kind}]\n", f"positives = [{pos}]\n", f"negatives = [{neg}]\n"]
        for stem, ep in sorted(self.episodes.items()):
            lines.append(f'\n[episodes."{stem}"]\n')
            lines.append(f"source           = {json.dumps(ep.source)}\n")
            oc = ", ".join(_fmt_match(m) for m in ep.open_candidates)
            cc = ", ".join(_fmt_match(m) for m in ep.close_candidates)
            ic = ", ".join(_fmt_match(m) for m in ep.intro_candidates)
            lines += [
                f"open_candidates  = [{oc}]\n",
                f'open_class       = "{ep.open_class}"\n',
                f"close_candidates = [{cc}]\n",
                f'close_class      = "{ep.close_class}"\n',
                f"intro_candidates = [{ic}]\n",
                f'intro_class      = "{ep.intro_class}"\n',
                f"cut = {str(ep.cut).lower()}\n",
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


# Two-tailed 98% t-critical values indexed by sample size n (df = n-1).
# For n >= 31 the normal approximation (2.326) is used.
# n < 2 is not in this table; _moe handles those by returning math.inf.
_T_CRIT: dict[int, float] = {
    2: 31.821,
    3: 6.965,
    4: 4.541,
    5: 3.747,
    6: 3.365,
    7: 3.143,
    8: 2.998,
    9: 2.896,
    10: 2.821,
    15: 2.624,
    20: 2.539,
    25: 2.492,
    30: 2.462,
}
_T_CRIT_LARGE = 2.326


def _t_critical(n: int) -> float:
    """98% two-tailed t-critical value for n samples (df = n-1)."""
    if n < 2:
        return math.inf
    if n >= 31:
        return _T_CRIT_LARGE
    for threshold in sorted(_T_CRIT, reverse=True):
        if n >= threshold:
            return _T_CRIT[threshold]
    return math.inf


_SINGLE_SAMPLE_MOE = 0.05


def _moe(scores: list[float], k: float | None = None) -> float:
    """Margin of error using the t-distribution (98% CI on the sample mean).

    Returns math.inf for n < 2 so that a single confirmed example never
    triggers auto-classification — the uncertain zone collapses only as
    evidence accumulates across multiple confirmed samples.

    Compatibility mode: when *k* is provided, use the legacy stddev-based
    formula used by older tests and callers.
    """
    n = len(scores)
    if k is not None:
        if n == 0:
            return 0.0
        if n == 1:
            return _SINGLE_SAMPLE_MOE
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        return k * math.sqrt(variance)

    if n < 2:
        return math.inf
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)  # Bessel's correction
    return _t_critical(n) * math.sqrt(variance) / math.sqrt(n)


def _compute_thresholds(target: TargetState) -> tuple[float, float]:
    """Return (theta_plus, theta_minus) for a target.

    θ⁺ = min(positives) + moe  — auto-positive requires exceeding the minimum confirmed
                                  positive BY the uncertainty buffer (worst-case threshold).
    θ⁻ = max(negatives) - moe  — auto-negative requires falling below the maximum confirmed
                                  negative BY the uncertainty buffer.
    The uncertain zone (θ⁻, θ⁺) widens with high variance and narrows as evidence accumulates.
    With no positives: θ⁺ = +inf. With no negatives: θ⁻ = -inf.
    With fewer than 2 confirmed samples of either kind, moe = inf so nothing auto-classifies.
    """
    pos = [s.score for s in target.positives]
    neg = [s.score for s in target.negatives]
    theta_plus = (min(pos) + _moe(pos)) if pos else math.inf
    theta_minus = (max(neg) - _moe(neg)) if neg else -math.inf
    return theta_plus, theta_minus


def _classify_score(score: float, theta_plus: float, theta_minus: float) -> str:
    if score >= theta_plus:
        return _POS
    if score <= theta_minus:
        return _NEG
    return _UNC


def _reclassify_all(state: PipelineState) -> None:
    """Recompute MOE-derived thresholds and re-classify uncertain episodes in-place."""
    tp_o, tm_o = _compute_thresholds(state.open_target)
    tp_c, tm_c = _compute_thresholds(state.close_target)
    for ep in state.episodes.values():
        if ep.open_class == _UNC:
            ep.open_class = _classify_score(ep.open_score, tp_o, tm_o)
        if ep.close_class == _UNC:
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


def _resolve_opt(cli_value: Any, state_value: Any) -> Any:
    return state_value if cli_value is None else cli_value


def _apply_sticky_review_args(args: argparse.Namespace, state: PipelineState) -> None:
    s = state.settings
    args.snippets_dir = Path(_resolve_opt(args.snippets_dir, s.snippets_dir))
    args.open_sample = str(_resolve_opt(args.open_sample, s.open_sample))
    args.close_sample = str(_resolve_opt(args.close_sample, s.close_sample))
    args.intro_sample = str(_resolve_opt(args.intro_sample, s.intro_sample))
    args.z_threshold = _resolve_opt(args.z_threshold, s.z_threshold)
    args.step_seconds = float(_resolve_opt(args.step_seconds, s.step_seconds))
    args.workers = int(_resolve_opt(args.workers, s.workers))
    args.max_matches = int(_resolve_opt(args.max_matches, s.max_matches))
    args.no_interactive = bool(_resolve_opt(args.no_interactive, s.no_interactive))
    args.overwrite = bool(_resolve_opt(args.overwrite, s.overwrite))

    s.snippets_dir = str(args.snippets_dir)
    s.open_sample = args.open_sample
    s.close_sample = args.close_sample
    s.intro_sample = args.intro_sample
    s.z_threshold = args.z_threshold
    s.step_seconds = args.step_seconds
    s.workers = args.workers
    s.max_matches = args.max_matches
    s.no_interactive = args.no_interactive
    s.overwrite = args.overwrite


def _apply_sticky_cut_args(args: argparse.Namespace, state: PipelineState) -> None:
    s = state.settings
    args.output_dir = Path(_resolve_opt(args.output_dir, s.output_dir))
    args.min_gap = float(_resolve_opt(args.min_gap, s.min_gap))
    args.max_gap = float(_resolve_opt(args.max_gap, s.max_gap))
    args.yes = bool(_resolve_opt(args.yes, s.yes))
    args.dry_run = bool(_resolve_opt(args.dry_run, s.dry_run))
    args.inclusive = bool(_resolve_opt(args.inclusive, s.inclusive))
    args.fade = float(_resolve_opt(args.fade, s.fade))

    s.output_dir = str(args.output_dir)
    s.min_gap = args.min_gap
    s.max_gap = args.max_gap
    s.yes = args.yes
    s.dry_run = args.dry_run
    s.inclusive = args.inclusive
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
# Detection — runs audio_detect as subprocess, returns top-N matches
# ---------------------------------------------------------------------------


def _detect_matches(
    source: Path,
    sample: Path,
    *,
    floor: float = 0.0,
    z_threshold: float | None,
    step_seconds: float,
    max_matches: int,
) -> list[_Match]:
    """Run audio_detect subprocess; return up to max_matches top-scoring matches."""
    command = [
        sys.executable,
        "-m",
        "part_io.cli.audio_detect",
        str(source),
        str(sample),
        "--threshold",
        str(floor),
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
            _Match(score=float(m["score"]), start=float(m["start"]), end=float(m["end"]))
            for m in data
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _emit(f"  WARNING: failed to parse detect output: {exc}")
        return []


def _detect_batch(
    episodes: list[Path],
    state: PipelineState,
    open_sample: Path,
    close_sample: Path,
    intro_sample: Path | None = None,
    *,
    z_threshold: float | None,
    step_seconds: float,
    workers: int,
    max_matches: int,
) -> None:
    """Detect open+close (and optional intro) matches for a batch.

    Updates pipeline state in-place.
    """
    _, tm_o = _compute_thresholds(state.open_target)
    _, tm_c = _compute_thresholds(state.close_target)
    open_floor = tm_o if math.isfinite(tm_o) else 0.0
    close_floor = tm_c if math.isfinite(tm_c) else 0.0
    if open_floor > 0 or close_floor > 0:
        _emit(f"  Floors from negatives: open={open_floor:.4f}  close={close_floor:.4f}")

    ep_by_stem = {ep.stem: ep for ep in episodes}
    jobs = [(ep, open_sample, "open", open_floor) for ep in episodes] + [
        (ep, close_sample, "close", close_floor) for ep in episodes
    ]
    if intro_sample is not None and intro_sample.exists():
        jobs += [(ep, intro_sample, "intro", 0.0) for ep in episodes]

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _detect_matches,
                ep,
                sample,
                floor=floor,
                z_threshold=z_threshold,
                step_seconds=step_seconds,
                max_matches=max_matches,
            ): (ep.stem, kind)
            for ep, sample, kind, floor in jobs
        }
        for future in as_completed(futures):
            done += 1
            stem, kind = futures[future]
            matches = future.result()
            ep_state = state.episode(stem)
            ep_state.source = str(ep_by_stem[stem])
            score_str = f"{matches[0].score:.4f}" if matches else "none"
            if kind == "open":
                ep_state.open_candidates = matches
                ep_state.open_class = _UNC if matches else _UND
            elif kind == "close":
                ep_state.close_candidates = matches
                ep_state.close_class = _UNC if matches else _UND
            else:  # intro
                ep_state.intro_candidates = matches
                ep_state.intro_class = _UNC if matches else _UND
            _emit(f"  [{done}/{len(jobs)}] {kind:5}  {stem}  ({score_str})")


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
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

except ImportError:

    def _getch() -> str:  # type: ignore[misc]
        line = input()
        return line[0].lower() if line else ""


def _start_audio(path: Path) -> Any:
    stdin_f = open(os.devnull, "rb")
    stdout_f = open(os.devnull, "wb")
    stderr_f = open(os.devnull, "wb")
    try:
        proc = launch_resolved(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            stdin=stdin_f,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    except Exception:
        stdin_f.close()
        stdout_f.close()
        stderr_f.close()
        raise
    _AUDIO_IO[id(proc)] = (stdin_f, stdout_f, stderr_f)
    return proc


def _start_audio_segment(source: Path, start: float, end: float) -> Any:
    """Stream a time slice from source directly through ffplay without writing to disk."""
    duration = max(0.0, end - start)
    stdin_f = open(os.devnull, "rb")
    stdout_f = open(os.devnull, "wb")
    stderr_f = open(os.devnull, "wb")
    try:
        proc = launch_resolved(
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
            ],
            stdin=stdin_f,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    except Exception:
        stdin_f.close()
        stdout_f.close()
        stderr_f.close()
        raise
    _AUDIO_IO[id(proc)] = (stdin_f, stdout_f, stderr_f)
    return proc


def _stop_audio(proc: Any | None) -> None:
    if proc is None:
        return

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()

    handles = _AUDIO_IO.pop(id(proc), None)
    if handles is not None:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                continue


@dataclass
class _UndoEntry:
    stem: str
    kind: str
    action: str
    segment: Segment
    target_list: list[Segment]
    prev_class: str = _UNC


@dataclass(frozen=True)
class _QuizItem:
    stem: str
    kind: str  # "open" or "close"
    candidate_idx: int  # index into open_candidates / close_candidates
    score: float  # candidate score (for sorting)


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


def _review_candidate(
    state: PipelineState,
    item: _QuizItem,
    *,
    open_sample: Path,
    close_sample: Path,
    history: list[_UndoEntry],
) -> str:
    """Review one candidate interactively.

    Returns 'approved', 'rejected', 'skipped', or 'undone'.
    """
    ep_state = state.episodes[item.stem]
    source = Path(ep_state.source)
    snippet = open_sample if item.kind == "open" else close_sample
    target = state.open_target if item.kind == "open" else state.close_target
    all_candidates = ep_state.open_candidates if item.kind == "open" else ep_state.close_candidates
    candidate = all_candidates[item.candidate_idx]
    prev_class = ep_state.open_class if item.kind == "open" else ep_state.close_class
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
            seg = Segment(
                source=str(source), start=candidate.start, end=candidate.end, score=candidate.score
            )
            if key == "a":
                target.positives.append(seg)
                if item.kind == "open":
                    ep_state.open_class = _POS
                else:
                    ep_state.close_class = _POS
                lst = target.positives
                print("\napproved", file=sys.stderr)
                action = "approved"
            else:
                target.negatives.append(seg)
                lst = target.negatives
                print("\nrejected", file=sys.stderr)
                action = "rejected"
            history.append(
                _UndoEntry(
                    stem=item.stem,
                    kind=item.kind,
                    action=key,
                    segment=seg,
                    target_list=lst,
                    prev_class=prev_class,
                )
            )
            _reclassify_all(state)
            return action
        elif key == "s":
            print("\nskipped", file=sys.stderr)
            return "skipped"
        elif key == "u" and history:
            entry = history.pop()
            entry.target_list.remove(entry.segment)
            prev_ep = state.episodes[entry.stem]
            if entry.kind == "open":
                prev_ep.open_class = entry.prev_class
            else:
                prev_ep.close_class = entry.prev_class
            _reclassify_all(state)
            print(
                f"\nundone ({entry.action} {entry.kind} for {entry.stem[:16]})",
                file=sys.stderr,
            )
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
    history: list[_UndoEntry],
) -> str:
    """Review the top candidate for (stem, kind). Returns 'classified', 'skipped', or 'undone'."""
    ep_state = state.episodes[stem]
    all_candidates = ep_state.open_candidates if kind == "open" else ep_state.close_candidates
    if not all_candidates:
        return "skipped"
    item = _QuizItem(stem=stem, kind=kind, candidate_idx=0, score=all_candidates[0].score)
    result = _review_candidate(
        state, item, open_sample=open_sample, close_sample=close_sample, history=history
    )
    return "classified" if result in ("approved", "rejected") else result


def _count_uncertain(state: PipelineState) -> int:
    return sum(
        1
        for ep in state.episodes.values()
        for cls in (ep.open_class, ep.close_class)
        if cls == _UNC
    )


def _collect_uncertain_candidates(state: PipelineState) -> list[_QuizItem]:
    """Return all candidates in the uncertain zone (θ⁻, θ⁺).

    Sorted by (candidate_idx, -score): all top candidates across every uncertain
    (stem, kind) pair come first, then all second candidates, etc. This ensures
    equal representation across pairs before diving deeper into any single one.
    """
    tp_o, tm_o = _compute_thresholds(state.open_target)
    tp_c, tm_c = _compute_thresholds(state.close_target)
    items: list[_QuizItem] = []
    for stem, ep in state.episodes.items():
        if ep.open_class == _UNC:
            for i, c in enumerate(ep.open_candidates):
                if tm_o < c.score < tp_o:
                    items.append(_QuizItem(stem=stem, kind="open", candidate_idx=i, score=c.score))
        if ep.close_class == _UNC:
            for i, c in enumerate(ep.close_candidates):
                if tm_c < c.score < tp_c:
                    items.append(_QuizItem(stem=stem, kind="close", candidate_idx=i, score=c.score))
    items.sort(key=lambda x: (x.candidate_idx, -x.score))
    return items


def _run_review_loop(
    state: PipelineState,
    *,
    open_sample: Path,
    close_sample: Path,
    state_path: Path,
    max_decisions: int | None = None,
) -> None:
    """Review uncertain targets until queue empty, user quits, or max_decisions reached."""
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
        )
        if result == "classified":
            decisions += 1
            state.save(state_path)
            ep = state.episodes[stem]
            ep_class = ep.open_class if kind == "open" else ep.close_class
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
    items: list[_QuizItem],
    *,
    open_sample: Path,
    close_sample: Path,
    state_path: Path,
) -> int:
    """Review a pre-collected set of uncertain candidates. Returns number of decisions made."""
    history: list[_UndoEntry] = []
    skipped_keys: set[tuple[str, str, int]] = set()
    decisions = 0
    remaining = list(items)

    while remaining:
        active = next(
            (
                item
                for item in remaining
                if (item.stem, item.kind, item.candidate_idx) not in skipped_keys
                and state.episodes.get(item.stem) is not None
                and (
                    state.episodes[item.stem].open_class
                    if item.kind == "open"
                    else state.episodes[item.stem].close_class
                )
                == _UNC
                and item.candidate_idx
                < len(
                    state.episodes[item.stem].open_candidates
                    if item.kind == "open"
                    else state.episodes[item.stem].close_candidates
                )
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
                history=history,
            )
        except KeyboardInterrupt:
            _emit("\nInterrupted. Progress saved.")
            break

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
    return decisions


# ---------------------------------------------------------------------------
# Pair and cut
# ---------------------------------------------------------------------------


def _find_best_pair(ep_state: EpisodeState, *, min_gap: float, max_gap: float) -> list[Any] | None:
    """Pass all open/close candidates to pair_ad_segments; return all valid pairs or None."""
    if not ep_state.open_candidates or not ep_state.close_candidates:
        return None
    opens = [
        AudioMatch(
            start_seconds=m.start,
            end_seconds=m.end,
            duration_seconds=m.end - m.start,
            score=m.score,
        )
        for m in ep_state.open_candidates
    ]
    closes = [
        AudioMatch(
            start_seconds=m.start,
            end_seconds=m.end,
            duration_seconds=m.end - m.start,
            score=m.score,
        )
        for m in ep_state.close_candidates
    ]
    try:
        segs, _, _ = pair_ad_segments(opens, closes, min_gap=min_gap, max_gap=max_gap)
    except (ValueError, KeyError):
        return None
    return segs if segs else None


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
    inclusive: bool = False,
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
        n_o = len(ep_state.open_candidates)
        n_c = len(ep_state.close_candidates)
        print(f"  No valid open->close pair ({n_o} open x {n_c} close candidates).")
        return "skipped"

    try:
        _validate_segments(segments)
    except ValueError as exc:
        print(f"  SKIP: {exc}")
        return "skipped"

    print(f"\n  {len(segments)} ad segment(s) to cut:")
    for i, seg in enumerate(segments, 1):
        cut_s = seg.cut_start if inclusive else seg.open_end
        cut_e = seg.cut_end if inclusive else seg.close_start
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
    if inclusive:
        cuts = [(seg.cut_start, seg.cut_end) for seg in segments]
    else:
        cuts = [(seg.open_end, seg.close_start) for seg in segments]

    if debug:
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
                return "failed"
            wrote += 1
        print(f"  Debug clips written: {wrote} -> {clip_dir}")

    spans = _spans_from_cuts(cuts)
    # If intro is detected, add a trim span to remove everything before it
    if ep_state.intro_class != _UND and ep_state.intro_candidates:
        intro_end = ep_state.intro_end
        print(f"  Intro detected at {intro_end:.1f}s - trimming preroll before it")
        # Insert intro trim at the beginning of spans
        spans.insert(0, (0.0, intro_end))
    filter_complex, _ = _build_filter_complex(spans, fade_dur=fade_dur)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        exit_code = _run_ffmpeg(source, filter_complex, temp_path)
        if exit_code != 0:
            temp_path.unlink(missing_ok=True)
            print(f"  FAILED: ffmpeg exited {exit_code}", file=sys.stderr)
            return "failed"

        temp_path.replace(output_path)
        print(f"  Written: {output_path}")
        return "cut"
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        print(f"  FAILED: {exc}", file=sys.stderr)
        return "failed"


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
    inclusive: bool = False,
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
            min_gap=min_gap,
            max_gap=max_gap,
            yes=yes,
            dry_run=dry_run,
            inclusive=inclusive,
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
            z_threshold=args.z_threshold,
            step_seconds=args.step_seconds,
            workers=args.workers,
            max_matches=args.max_matches,
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
        inclusive=args.inclusive,
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

    undetected = [
        ep for ep in all_full if args.overwrite or not state.episode(ep.stem).is_detected()
    ]
    n_already = len(all_full) - len(undetected)
    _emit(f"Episodes: {len(all_full)} total, {n_already} detected, {len(undetected)} to detect")

    # Detect one episode at a time until we have enough uncertain candidates for a quiz.
    quiz_items = _collect_uncertain_candidates(state)
    while len(quiz_items) < args.quiz_size and undetected:
        ep_path = undetected.pop(0)
        _emit(f"\nDetecting {ep_path.stem}...")
        _detect_batch(
            [ep_path],
            state,
            open_sample,
            close_sample,
            intro_sample,
            z_threshold=args.z_threshold,
            step_seconds=args.step_seconds,
            workers=args.workers,
            max_matches=args.max_matches,
        )
        _reclassify_all(state)
        state.save(state_path)
        quiz_items = _collect_uncertain_candidates(state)

    quiz_items = quiz_items[: args.quiz_size]
    n_unc = _count_uncertain(state)

    if not args.no_interactive:
        if quiz_items:
            _emit(f"\n{len(quiz_items)} candidate(s) to review ({n_unc} uncertain total).")
            try:
                _run_quiz(
                    state,
                    quiz_items,
                    open_sample=open_sample,
                    close_sample=close_sample,
                    state_path=state_path,
                )
            except KeyboardInterrupt:
                _emit("\nInterrupted. Progress saved.")
        else:
            msg = (
                f"{n_unc} uncertain remaining — nothing new to review."
                if n_unc
                else "Nothing to review."
            )
            _emit(f"\n{msg}")

    n_cut, n_skipped, n_failed = _cut_cuttable(
        state,
        remote_dir=remote_dir,
        output_dir=output_dir,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        yes=args.yes,
        dry_run=args.dry_run,
        state_path=state_path,
        inclusive=args.inclusive,
        fade_dur=args.fade,
        debug=args.debug,
    )
    n_remain_undet = sum(1 for ep in all_full if not state.episode(ep.stem).is_detected())
    n_remain_unc = _count_uncertain(state)
    _emit(f"\nProgress: {n_remain_undet} undetected, {n_remain_unc} uncertain remaining.")
    if n_cut or n_skipped or n_failed:
        print(f"Cut: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_remain_undet + n_remain_unc > 0:
        _emit("Run again to continue.")
    if n_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--z-threshold",
        type=float,
        default=None,
        metavar="N",
        help="Optional z-score filter: only keep windows > mean + N*std (default: off)",
    )
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


def _build_review_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("review", help="Detect matches and review them interactively")
    _add_remote_dir_arg(p)
    p.add_argument("--snippets-dir", type=Path, default=None)
    p.add_argument("--open-sample", default=None)
    p.add_argument("--close-sample", default=None)
    p.add_argument("--intro-sample", default=None)
    _add_detect_args(p)
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument(
        "--no-interactive", action="store_true", default=None, help="Detect only, skip review"
    )


def _build_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("cut", help="Cut ad segments using labels from __state__.toml")
    _add_remote_dir_arg(p)
    p.add_argument("--output-dir", type=Path, default=None)
    _add_cut_args(p)


def _build_loop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "loop",
        help="Detect one episode at a time, quiz uncertain candidates, then cut. Run repeatedly.",
    )
    _add_remote_dir_arg(p)
    p.add_argument("--snippets-dir", type=Path, default=None)
    p.add_argument("--open-sample", default=None)
    p.add_argument("--close-sample", default=None)
    p.add_argument("--intro-sample", default=None)
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
