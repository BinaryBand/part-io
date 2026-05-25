from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from part_io.adapters.audio.matcher import _ANALYSIS_RATE, _HOP_SIZE
from part_io.adapters.audio.snippet_profile import decode_matrix, encode_matrix
from part_io.utils.hash import partial_file_hash

# Classification labels
_POS = "positive"
_NEG = "negative"
_UNC = "uncertain"
_UND = "undetected"
_AUDIO_KINDS = ("open", "close", "intro", "outro")


@dataclass
class SnippetEntry:
    """A named snippet and its current detection profile (seed or consensus)."""

    name: str
    profile: np.ndarray  # float32 (n_frames, band_count*2)


@dataclass(frozen=True)
class Segment:
    """A confirmed example of an audio target at a specific position."""

    stem: str
    start: float
    end: float
    score: float


def _b85_to_profile(s: str) -> np.ndarray:
    """Decode a legacy [profiles] base85 checkpoint (migration only)."""
    buf = io.BytesIO(base64.b85decode(s))
    return np.load(buf)


@dataclass
class TargetState:
    """Global confirmed examples for one target type (open or close)."""

    positives: list[Segment] = field(default_factory=list)
    negatives: list[Segment] = field(default_factory=list)


@dataclass
class _Match:
    score: float
    start: float
    end: float
    label: str | None = None


@dataclass
class EpisodeState:
    source_hash: str | None = None  # SHA-256 of first 64 KB of source; None means unverified
    candidates: dict[str, list[_Match]] = field(
        default_factory=lambda: {kind: [] for kind in _AUDIO_KINDS}
    )
    cut: bool = False

    def candidates_for(self, kind: str) -> list[_Match]:
        return self.candidates.setdefault(kind, [])

    @staticmethod
    def _normalized_label(value: str | None) -> str | None:
        if value in (_POS, _NEG):
            return value
        return None

    def class_for(self, kind: str) -> str:
        candidates = self.candidates_for(kind)
        if not candidates:
            return _UND
        labels = [self._normalized_label(match.label) for match in candidates]
        if any(label == _POS for label in labels):
            return _POS
        if all(label == _NEG for label in labels):
            return _NEG
        return _UNC

    def first_positive_candidate_for(self, kind: str) -> _Match | None:
        for candidate in self.candidates_for(kind):
            if self._normalized_label(candidate.label) == _POS:
                return candidate
        return None

    def source_hash_valid(self, path: Path) -> bool:
        """Return True if *path* matches the stored hash, False if stale or unverified."""
        if not self.source_hash:
            return False
        try:
            return partial_file_hash(path) == self.source_hash
        except OSError:
            return False

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

    def is_detected(self) -> bool:
        return any(self.class_for(kind) != _UND for kind in _AUDIO_KINDS)

    def is_cuttable(self) -> bool:
        return self.class_for("open") == _POS and self.class_for("close") == _POS


@dataclass
class DetectSettings:
    step_seconds: float = 0.1
    workers: int = 2
    max_matches: int = 3
    overwrite: bool = False


@dataclass
class CutConfig:
    min_gap: float = -15.0
    max_gap: float = 300.0
    ad_inclusive: bool = True  # open_start→close_end; False = open_end→close_start
    intro_exclusive: bool = True  # trim to intro_start; False = trim to intro_end
    fade: float = 0.5
    output_dir: str = "downloads/remove"
    debug: bool = False


@dataclass
class RunSettings:
    detect: DetectSettings = field(default_factory=DetectSettings)
    cut: CutConfig = field(default_factory=CutConfig)
    quiz_size: int = 10


def _fmt_seg(s: Segment) -> str:
    return (
        f"{{stem = {json.dumps(s.stem)}, "
        f"start = {s.start:.3f}, end = {s.end:.3f}, score = {s.score:.6f}}}"
    )


def _fmt_match(m: _Match) -> str:
    base = f"score = {m.score:.6f}, start = {m.start:.3f}, end = {m.end:.3f}"
    if m.label in (_POS, _NEG):
        return f"{{{base}, label = {json.dumps(m.label)}}}"
    return f"{{{base}}}"


def _load_match(raw: dict) -> _Match:
    label_raw = raw.get("label")
    label = str(label_raw) if label_raw in (_POS, _NEG) else None
    return _Match(
        score=float(raw["score"]),
        start=float(raw["start"]),
        end=float(raw["end"]),
        label=label,
    )


def _seg_stem(raw: dict) -> str:
    if "stem" in raw:
        return str(raw["stem"])
    # migrate: old format stored the full path under "source"
    return Path(str(raw.get("source", ""))).stem


def _load_target(raw: dict) -> TargetState:
    t = TargetState()
    for seg in raw.get("positives", []):
        t.positives.append(
            Segment(
                stem=_seg_stem(seg),
                start=float(seg["start"]),
                end=float(seg["end"]),
                score=float(seg["score"]),
            )
        )
    for seg in raw.get("negatives", []):
        t.negatives.append(
            Segment(
                stem=_seg_stem(seg),
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
        candidates_key = f"{kind}_candidates"
        if candidates_key in raw:
            ep.candidates[kind] = [_load_match(m) for m in raw[candidates_key]]
    return ep


def _load_settings(raw: dict) -> RunSettings:
    d = raw.get("detect", raw)  # fall back to flat raw for migration
    c = raw.get("cut", raw)
    return RunSettings(
        detect=DetectSettings(
            step_seconds=float(d.get("step_seconds", 0.1)),
            workers=int(d.get("workers", 2)),
            max_matches=int(d.get("max_matches", 3)),
            overwrite=bool(d.get("overwrite", False)),
        ),
        cut=CutConfig(
            min_gap=float(c.get("min_gap", -15.0)),
            max_gap=float(c.get("max_gap", 300.0)),
            ad_inclusive=bool(c.get("ad_inclusive", True)),
            intro_exclusive=bool(c.get("intro_exclusive", True)),
            fade=float(c.get("fade", 0.5)),
            output_dir=str(c.get("output_dir", "downloads/remove")),
            debug=bool(c.get("debug", False)),
        ),
        quiz_size=int(raw.get("quiz_size", 10)),
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


@dataclass
class PipelineState:
    open_target: TargetState = field(default_factory=TargetState)
    close_target: TargetState = field(default_factory=TargetState)
    settings: RunSettings = field(default_factory=RunSettings)
    episodes: dict[str, EpisodeState] = field(default_factory=dict)
    snippets: list[SnippetEntry] = field(default_factory=list)

    def episode(self, stem: str) -> EpisodeState:
        if stem not in self.episodes:
            self.episodes[stem] = EpisodeState()
        return self.episodes[stem]

    def profile_for(self, name: str) -> np.ndarray | None:
        for s in self.snippets:
            if s.name == name:
                return s.profile
        return None

    def set_profile(self, name: str, profile: np.ndarray) -> None:
        for s in self.snippets:
            if s.name == name:
                s.profile = profile
                return
        self.snippets.append(SnippetEntry(name=name, profile=profile))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        if not path.exists():
            return cls()
        _migrate_episode_keys(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        state = cls(
            open_target=_load_target(data.get("targets", {}).get("open", {})),
            close_target=_load_target(data.get("targets", {}).get("close", {})),
            settings=_load_settings(data.get("settings", {})),
        )
        for raw_snip in data.get("snippets", []):
            try:
                name = str(raw_snip["name"])
                prof = raw_snip["profile"]
                arr = decode_matrix(
                    str(prof["data"]), int(prof["n_frames"]), int(prof["band_count"])
                )
                state.snippets.append(SnippetEntry(name=name, profile=arr))
            except Exception:  # noqa: BLE001, S110
                pass  # corrupt snippet — will be recomputed on next run
        # Migration: old [profiles] dict → SnippetEntry (removed in this version)
        if not state.snippets:
            for kind, b85 in data.get("profiles", {}).items():
                try:
                    state.snippets.append(
                        SnippetEntry(name=kind, profile=_b85_to_profile(str(b85)))
                    )
                except Exception:  # noqa: BLE001, S110
                    pass
        for stem, ep_raw in data.get("episodes", {}).items():
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
        s = self.settings
        d = s.detect
        c = s.cut
        lines += [
            "\n[settings]\n",
            f"quiz_size = {s.quiz_size}\n",
            "\n[settings.detect]\n",
            f"step_seconds = {d.step_seconds:.6g}\n",
            f"workers      = {d.workers}\n",
            f"max_matches  = {d.max_matches}\n",
            f"overwrite    = {str(d.overwrite).lower()}\n",
            "\n[settings.cut]\n",
            f"min_gap         = {c.min_gap:.6g}\n",
            f"max_gap         = {c.max_gap:.6g}\n",
            f"ad_inclusive    = {str(c.ad_inclusive).lower()}\n",
            f"intro_exclusive = {str(c.intro_exclusive).lower()}\n",
            f"fade            = {c.fade:.6g}\n",
            f"output_dir      = {json.dumps(c.output_dir)}\n",
            f"debug           = {str(c.debug).lower()}\n",
        ]
        for kind, target in [("open", self.open_target), ("close", self.close_target)]:
            seen: set[tuple[str, float, float]] = set()
            deduped_pos: list[Segment] = []
            for s in target.positives:
                key = (s.stem, s.start, s.end)
                if key not in seen:
                    seen.add(key)
                    deduped_pos.append(s)
            seen = set()
            deduped_neg: list[Segment] = []
            for s in target.negatives:
                key = (s.stem, s.start, s.end)
                if key not in seen:
                    seen.add(key)
                    deduped_neg.append(s)
            pos = ", ".join(_fmt_seg(s) for s in deduped_pos)
            neg = ", ".join(_fmt_seg(s) for s in deduped_neg)
            lines += [f"\n[targets.{kind}]\n", f"positives = [{pos}]\n", f"negatives = [{neg}]\n"]
        for s in self.snippets:
            n_frames, width = s.profile.shape
            band_count = width // 2
            lines.append("\n[[snippets]]\n")
            lines.append(f"name = {json.dumps(s.name)}\n")
            lines.append("\n[snippets.profile]\n")
            lines.append(f"n_frames      = {n_frames}\n")
            lines.append(f"analysis_rate = {_ANALYSIS_RATE}\n")
            lines.append(f"hop_size      = {_HOP_SIZE}\n")
            lines.append(f"band_count    = {band_count}\n")
            lines.append(f"data          = {json.dumps(encode_matrix(s.profile))}\n")
        for stem, ep in sorted(self.episodes.items()):
            lines.append(f'\n[episodes."{stem}"]\n')
            if ep.source_hash is not None:
                lines.append(f"source_hash      = {json.dumps(ep.source_hash)}\n")
            for kind in _AUDIO_KINDS:
                candidates = ep.candidates_for(kind)
                if candidates:
                    key_pad = f"{kind}_candidates".ljust(16)
                    formatted = ", ".join(_fmt_match(m) for m in candidates)
                    lines.append(f"{key_pad} = [{formatted}]\n")
            if ep.cut:
                lines.append("cut = true\n")
        _atomic_write_text(path, "".join(lines))


def _target_to_dict_lists(target: TargetState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [
        {"stem": s.stem, "start": float(s.start), "end": float(s.end), "score": float(s.score)}
        for s in target.positives
    ]
    negatives = [
        {"stem": s.stem, "start": float(s.start), "end": float(s.end), "score": float(s.score)}
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
            stem=str(s.get("stem", "")),
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            score=float(s.get("score", 0.0)),
        )
        for s in positives
    ]
    target.negatives = [
        Segment(
            stem=str(s.get("stem", "")),
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            score=float(s.get("score", 0.0)),
        )
        for s in negatives
    ]


__all__ = [
    "Segment",
    "SnippetEntry",
    "TargetState",
    "_Match",
    "EpisodeState",
    "DetectSettings",
    "CutConfig",
    "RunSettings",
    "PipelineState",
    "_fmt_seg",
    "_fmt_match",
    "_target_to_dict_lists",
    "_replace_target_from_dict_lists",
]
