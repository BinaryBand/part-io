"""Compute and persist spectral fingerprints for snippet audio files.

``.profile.toml`` is written alongside each snippet.  It serialises a
:class:`SnippetProfileModel` — provenance metadata, a verbatim keyframe
(frame 0), and ``[[diffs]]`` entries for frames 1 … n_frames-1.

Reconstruct frame *k* from the stored data::

    energy_k = keyframe_energy + sum(diffs[0..k-1].energy)
    delta_k  = keyframe_delta  + sum(diffs[0..k-1].delta)

Relationships between fields::

    duration_s = n_frames * hop_size / analysis_rate
    # full matrix shape: (n_frames, band_count * 2)
    # energy columns: matrix[:, :band_count]
    # delta  columns: matrix[:, band_count:]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tomli_w
from pydantic import BaseModel, ConfigDict

from part_io.adapters.audio.matcher import (
    _ANALYSIS_RATE,
    _BAND_COUNT,
    _HOP_SIZE,
    band_center_frequencies,
    compute_audio_file_profile,
)
from part_io.utils.hash import partial_file_hash

PROFILE_VERSION = 1

_FREQ_LABELS: list[str] = [
    f"band_{i:02d}_{round(hz):d}hz" for i, hz in enumerate(band_center_frequencies())
]

_FLOAT_PRECISION = 5  # decimal places stored in .profile.toml


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FrameDiff(BaseModel):
    """Per-band spectral difference from the previous frame.

    ``energy`` holds the change in normalised log-energy for each of the
    ``band_count`` frequency bands.  ``delta`` holds the change in the
    first-order spectral-delta features for the same bands.

    A value of ``0.0`` means the band was identical to the previous frame;
    positive means it grew, negative means it shrank.
    """

    model_config = ConfigDict(extra="forbid")

    energy: list[float]
    delta: list[float]


class SnippetProfileModel(BaseModel):
    """Delta-encoded spectral profile for one snippet audio file.

    The detection matrix (shape ``n_frames × band_count * 2``) is stored as a
    keyframe followed by ``n_frames - 1`` :class:`FrameDiff` entries.
    Reconstruct frame *k*::

        energy_k = keyframe_energy + sum(d.energy for d in diffs[:k])
        delta_k  = keyframe_delta  + sum(d.delta  for d in diffs[:k])
    """

    model_config = ConfigDict(extra="forbid")

    # provenance
    source_hash: str
    n_frames: int
    analysis_rate: int
    hop_size: int
    band_count: int
    generated_at: str
    profile_version: int
    band_labels: list[str]

    # frame 0 — verbatim
    keyframe_energy: list[float]
    keyframe_delta: list[float]

    # frames 1 … n_frames-1 — difference from the preceding frame
    diffs: list[FrameDiff]


# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProfileData:
    source_hash: str
    n_frames: int
    n_dims: int
    energy: np.ndarray  # shape (n_frames, band_count)
    delta: np.ndarray  # shape (n_frames, band_count)
    generated_at: str


def _compute(snippet_path: Path) -> _ProfileData:
    profile = compute_audio_file_profile(snippet_path)
    if profile.size == 0:
        raise ValueError(
            f"Could not compute profile for {snippet_path}: audio too short or decode failed"
        )
    n_frames, n_dims = profile.shape
    return _ProfileData(
        source_hash=partial_file_hash(snippet_path),
        n_frames=n_frames,
        n_dims=n_dims,
        energy=profile[:, :_BAND_COUNT],
        delta=profile[:, _BAND_COUNT:],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _round_row(row: np.ndarray) -> list[float]:
    return [round(float(v), _FLOAT_PRECISION) for v in row]


def _build_profile_model(d: _ProfileData) -> SnippetProfileModel:
    diffs: list[FrameDiff] = [
        FrameDiff(
            energy=_round_row(d.energy[i] - d.energy[i - 1]),
            delta=_round_row(d.delta[i] - d.delta[i - 1]),
        )
        for i in range(1, d.n_frames)
    ]
    return SnippetProfileModel(
        source_hash=d.source_hash,
        n_frames=d.n_frames,
        analysis_rate=_ANALYSIS_RATE,
        hop_size=_HOP_SIZE,
        band_count=_BAND_COUNT,
        generated_at=d.generated_at,
        profile_version=PROFILE_VERSION,
        band_labels=_FREQ_LABELS,
        keyframe_energy=_round_row(d.energy[0]),
        keyframe_delta=_round_row(d.delta[0]),
        diffs=diffs,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_PROFILE_HEADER = (
    "# Re-generate with: poetry run part-io-tasks snippet-profile <path>\n"
    "#\n"
    "# Reconstruct frame k:\n"
    "#   energy_k = keyframe_energy + sum(diffs[0..k-1].energy)\n"
    "#   delta_k  = keyframe_delta  + sum(diffs[0..k-1].delta)\n"
    "\n"
)


def write_snippet_profile(snippet_path: Path) -> Path:
    """Compute and write a ``.profile.toml`` alongside *snippet_path*.

    Serialises a :class:`SnippetProfileModel` via ``tomli_w``.
    Returns the path of the written file.
    """
    d = _compute(snippet_path)
    model = _build_profile_model(d)
    out = snippet_path.with_suffix(".profile.toml")
    out.write_text(
        f"# Spectral fingerprint for {snippet_path.name} — generated by part-io\n"
        + _PROFILE_HEADER
        + tomli_w.dumps(model.model_dump()),
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------


def is_profile_current(snippet_path: Path) -> bool:
    """Return True if a fresh ``.profile.toml`` already exists for *snippet_path*."""
    profile_path = snippet_path.with_suffix(".profile.toml")
    if not profile_path.exists():
        return False
    try:
        text = profile_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("source_hash"):
                stored = line.split("=", 1)[1].strip().strip('"')
                return stored == partial_file_hash(snippet_path)
        return False
    except OSError:
        return False


__all__ = [
    "SnippetProfileModel",
    "FrameDiff",
    "write_snippet_profile",
    "is_profile_current",
    "PROFILE_VERSION",
]
