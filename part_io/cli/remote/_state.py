from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from part_io.utils.hash import partial_file_hash

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

_LOG = logging.getLogger(__name__)


# Classification labels
_POS = "positive"
_NEG = "negative"
_UNC = "uncertain"
_UND = "undetected"
_AUDIO_KINDS = ("open", "close", "intro", "outro")


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
    source_hash: str | None = None  # SHA-256 of first 64 KB of source; None means unverified
    candidates: dict[str, list[_Match]] = field(
        default_factory=lambda: {kind: [] for kind in _AUDIO_KINDS}
    )
    classes: dict[str, str] = field(default_factory=lambda: {kind: _UND for kind in _AUDIO_KINDS})
    cut: bool = False

    def __init__(
        self,
        source: str = "",
        source_hash: str | None = None,
        candidates: dict[str, list[_Match]] | None = None,
        classes: dict[str, str] | None = None,
        cut: bool = False,
        *,
        open_candidates: list[_Match] | None = None,
        close_candidates: list[_Match] | None = None,
        intro_candidates: list[_Match] | None = None,
        outro_candidates: list[_Match] | None = None,
        open_class: str = _UND,
        close_class: str = _UND,
        intro_class: str = _UND,
        outro_class: str = _UND,
    ) -> None:
        self.source_hash = source_hash
        self.cut = cut
        self.candidates = {kind: [] for kind in _AUDIO_KINDS}
        self.classes = {kind: _UND for kind in _AUDIO_KINDS}

        if candidates is not None:
            for kind in _AUDIO_KINDS:
                self.candidates[kind] = list(candidates.get(kind, []))
        if classes is not None:
            for kind in _AUDIO_KINDS:
                self.classes[kind] = str(classes.get(kind, _UND))

        if open_candidates is not None:
            self.open_candidates = list(open_candidates)
        if close_candidates is not None:
            self.close_candidates = list(close_candidates)
        if intro_candidates is not None:
            self.intro_candidates = list(intro_candidates)
        if outro_candidates is not None:
            self.outro_candidates = list(outro_candidates)

        self.open_class = open_class
        self.close_class = close_class
        self.intro_class = intro_class
        self.outro_class = outro_class

    def candidates_for(self, kind: str) -> list[_Match]:
        return self.candidates.setdefault(kind, [])

    def class_for(self, kind: str) -> str:
        return self.classes.get(kind, _UND)

    def set_class(self, kind: str, value: str) -> None:
        self.classes[kind] = value

    def first_candidate_for(self, kind: str) -> _Match | None:
        candidates = self.candidates_for(kind)
        return candidates[0] if candidates else None

    def score_for(self, kind: str) -> float:
        first = self.first_candidate_for(kind)
        return first.score if first is not None else 0.0

    def start_for(self, kind: str) -> float:
        first = self.first_candidate_for(kind)
        return first.start if first is not None else 0.0

    def end_for(self, kind: str) -> float:
        first = self.first_candidate_for(kind)
        return first.end if first is not None else 0.0

    def source_hash_valid(self, path: Path) -> bool:
        """Return True if *path* matches the stored hash, False if stale or unverified."""
        if not self.source_hash:
            return False
        try:
            return partial_file_hash(path) == self.source_hash
        except OSError:
            return False

    # Compatibility properties: keep old attribute-style accesses while
    # storing data in dict-backed fields.
    @property
    def open_candidates(self) -> list[_Match]:
        return self.candidates_for("open")

    @open_candidates.setter
    def open_candidates(self, value: list[_Match]) -> None:
        self.candidates["open"] = value

    @property
    def close_candidates(self) -> list[_Match]:
        return self.candidates_for("close")

    @close_candidates.setter
    def close_candidates(self, value: list[_Match]) -> None:
        self.candidates["close"] = value

    @property
    def intro_candidates(self) -> list[_Match]:
        return self.candidates_for("intro")

    @intro_candidates.setter
    def intro_candidates(self, value: list[_Match]) -> None:
        self.candidates["intro"] = value

    @property
    def outro_candidates(self) -> list[_Match]:
        return self.candidates_for("outro")

    @outro_candidates.setter
    def outro_candidates(self, value: list[_Match]) -> None:
        self.candidates["outro"] = value

    @property
    def open_class(self) -> str:
        return self.class_for("open")

    @open_class.setter
    def open_class(self, value: str) -> None:
        self.set_class("open", value)

    @property
    def close_class(self) -> str:
        return self.class_for("close")

    @close_class.setter
    def close_class(self, value: str) -> None:
        self.set_class("close", value)

    @property
    def intro_class(self) -> str:
        return self.class_for("intro")

    @intro_class.setter
    def intro_class(self, value: str) -> None:
        self.set_class("intro", value)

    @property
    def outro_class(self) -> str:
        return self.class_for("outro")

    @outro_class.setter
    def outro_class(self, value: str) -> None:
        self.set_class("outro", value)

    # Read-only convenience properties so all existing callers of open_score etc. still work.
    @property
    def open_score(self) -> float:
        return self.score_for("open")

    @property
    def open_start(self) -> float:
        return self.start_for("open")

    @property
    def open_end(self) -> float:
        return self.end_for("open")

    @property
    def close_score(self) -> float:
        return self.score_for("close")

    @property
    def close_start(self) -> float:
        return self.start_for("close")

    @property
    def close_end(self) -> float:
        return self.end_for("close")

    @property
    def intro_score(self) -> float:
        return self.score_for("intro")

    @property
    def intro_start(self) -> float:
        return self.start_for("intro")

    @property
    def intro_end(self) -> float:
        return self.end_for("intro")

    def is_detected(self) -> bool:
        return any(self.class_for(kind) != _UND for kind in _AUDIO_KINDS)

    def is_cuttable(self) -> bool:
        return self.class_for("open") == _POS and self.class_for("close") == _POS


@dataclass
class RunSettings:
    step_seconds: float = 0.1
    workers: int = 2
    max_matches: int = 3
    min_gap: float = -15.0
    max_gap: float = 300.0
    ad_inclusive: bool = (
        True  # include jingle audio in ad cut (open_start→close_end vs open_end→close_start)
    )
    intro_exclusive: bool = (
        True  # trim to intro_start, keeping the intro jingle (False = trim to intro_end)
    )
    fade: float = 0.5
    quiz_size: int = 10
    overwrite: bool = False
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
    raw_hash = raw.get("source_hash")
    ep = EpisodeState(
        source_hash=str(raw_hash) if raw_hash is not None else None,
        cut=bool(raw.get("cut", False)),
    )
    for kind in _AUDIO_KINDS:
        ep.set_class(kind, str(raw.get(f"{kind}_class", _UND)))

        candidates_key = f"{kind}_candidates"
        if candidates_key in raw:
            ep.candidates[kind] = [_load_match(m) for m in raw[candidates_key]]

    # Migrate old scalar format (open_score/open_start/open_end, close_*)
    for kind in ("open", "close"):
        score_key = f"{kind}_score"
        if ep.candidates_for(kind):
            continue
        if float(raw.get(score_key, 0.0)) <= 0:
            continue
        ep.candidates[kind] = [
            _Match(
                score=float(raw[score_key]),
                start=float(raw.get(f"{kind}_start", 0.0)),
                end=float(raw.get(f"{kind}_end", 0.0)),
            )
        ]
    return ep


def _load_settings(raw: dict) -> RunSettings:
    return RunSettings(
        step_seconds=float(raw.get("step_seconds", 0.1)),
        workers=int(raw.get("workers", 2)),
        max_matches=int(raw.get("max_matches", 3)),
        min_gap=float(raw.get("min_gap", -15.0)),
        max_gap=float(raw.get("max_gap", 300.0)),
        ad_inclusive=bool(raw.get("ad_inclusive", True)),
        intro_exclusive=bool(raw.get("intro_exclusive", True)),
        fade=float(raw.get("fade", 0.5)),
        quiz_size=int(raw.get("quiz_size", 10)),
        overwrite=bool(raw.get("overwrite", False)),
        output_dir=str(raw.get("output_dir", "downloads/remove")),
        debug=bool(raw.get("debug", False)),
    )


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to path atomically via a same-directory temp file.

    Uses rename on the same filesystem (instant, no data loss on interruption).
    Falls back to copy+delete when the destination is on a different device
    (e.g. an rclone mount), ensuring the destination file is never left in a
    partial state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", dir=path.parent, delete=False, encoding=encoding
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(path)
    except OSError:
        shutil.copy2(tmp_path, path)
        tmp_path.unlink(missing_ok=True)


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
    _atomic_write_text(path, "".join(fixed_lines))


def _migrate_old_episode(
    ep_raw: dict,
    source: str,
    open_target: TargetState,
    close_target: TargetState,
) -> EpisodeState:
    """Convert one old-format episode (match lists + approved/rejected) to new format."""
    ep = EpisodeState(cut=bool(ep_raw.get("cut", False)))
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
            ep.set_class(kind, _UND)
            continue
        score = float(best_raw["score"])
        start = float(best_raw["start"])
        end = float(best_raw["end"])
        ep.candidates[kind] = [_Match(score=score, start=start, end=end)]
        if approved:
            ep.set_class(kind, _POS)
            target.positives.append(Segment(source=source, start=start, end=end, score=score))
        elif rejected:
            ep.set_class(kind, _NEG)
            target.negatives.append(Segment(source=source, start=start, end=end, score=score))
        else:
            ep.set_class(kind, _UNC)
    return ep


def _migrate_old_state(data: dict) -> "PipelineState":
    _LOG.info("Migrating state.toml to new format (one-time conversion)...")
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
            _LOG.warning("TOML unavailable. Starting with empty state.")
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
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "schemas"
            / "remote_pipeline_state.schema.json"
        )
        try:
            schema_ref = os.path.relpath(schema_path, path.parent)
            schema_ref = schema_ref.replace("\\", "/")
        except ValueError:
            schema_ref = schema_path.resolve().as_uri()

        lines: list[str] = [
            f"#:schema {schema_ref}\n",
            "# Remote episode pipeline state.\n",
            "# Edit freely — delete this file to start fresh.\n",
        ]
        settings = self.settings
        setting_lines = ["\n[settings]\n"]
        setting_lines += [
            f"step_seconds = {settings.step_seconds:.6g}\n",
            f"workers = {settings.workers}\n",
            f"max_matches = {settings.max_matches}\n",
            f"min_gap = {settings.min_gap:.6g}\n",
            f"max_gap = {settings.max_gap:.6g}\n",
            f"ad_inclusive = {str(settings.ad_inclusive).lower()}\n",
            f"intro_exclusive = {str(settings.intro_exclusive).lower()}\n",
            f"fade = {settings.fade:.6g}\n",
            f"quiz_size = {settings.quiz_size}\n",
            f"overwrite = {str(settings.overwrite).lower()}\n",
            f"output_dir = {json.dumps(settings.output_dir)}\n",
            f"debug = {str(settings.debug).lower()}\n",
        ]
        lines += setting_lines
        for kind, target in [("open", self.open_target), ("close", self.close_target)]:
            seen: set[tuple[str, float, float]] = set()
            deduped_pos: list[Segment] = []
            for s in target.positives:
                key = (s.source, s.start, s.end)
                if key not in seen:
                    seen.add(key)
                    deduped_pos.append(s)
            seen = set()
            deduped_neg: list[Segment] = []
            for s in target.negatives:
                key = (s.source, s.start, s.end)
                if key not in seen:
                    seen.add(key)
                    deduped_neg.append(s)
            pos = ", ".join(_fmt_seg(s) for s in deduped_pos)
            neg = ", ".join(_fmt_seg(s) for s in deduped_neg)
            lines += [f"\n[targets.{kind}]\n", f"positives = [{pos}]\n", f"negatives = [{neg}]\n"]
        for stem, ep in sorted(self.episodes.items()):
            lines.append(f'\n[episodes."{stem}"]\n')
            if ep.source_hash is not None:
                lines.append(f"source_hash      = {json.dumps(ep.source_hash)}\n")
            for kind in _AUDIO_KINDS:
                candidates = ep.candidates_for(kind)
                cls = ep.class_for(kind)
                if candidates:
                    key_pad = f"{kind}_candidates".ljust(16)
                    formatted = ", ".join(_fmt_match(m) for m in candidates)
                    lines.append(f"{key_pad} = [{formatted}]\n")
                if cls != _UND:
                    class_pad = f"{kind}_class".ljust(16)
                    lines.append(f'{class_pad} = "{cls}"\n')
            if ep.cut:
                lines.append("cut = true\n")
        _atomic_write_text(path, "".join(lines))


def _target_to_dict_lists(target: TargetState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [
        {"source": s.source, "start": float(s.start), "end": float(s.end), "score": float(s.score)}
        for s in target.positives
    ]
    negatives = [
        {"source": s.source, "start": float(s.start), "end": float(s.end), "score": float(s.score)}
        for s in target.negatives
    ]
    return positives, negatives


def _replace_target_from_dict_lists(
    target: TargetState,
    *,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
) -> None:
    target.positives = [
        Segment(
            source=str(s.get("source", "")),
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            score=float(s.get("score", 0.0)),
        )
        for s in positives
    ]
    target.negatives = [
        Segment(
            source=str(s.get("source", "")),
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            score=float(s.get("score", 0.0)),
        )
        for s in negatives
    ]


__all__ = [
    "Segment",
    "TargetState",
    "_Match",
    "EpisodeState",
    "RunSettings",
    "PipelineState",
    "_fmt_seg",
    "_fmt_match",
    "_target_to_dict_lists",
    "_replace_target_from_dict_lists",
]
