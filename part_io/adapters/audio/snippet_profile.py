"""Compute and persist spectral fingerprints for snippet audio files.

``.profile.toml`` is written alongside each snippet.  It serialises a
:class:`SnippetProfileModel` whose ``data`` field stores the full
``(n_frames, band_count * 2)`` float32 detection matrix as a single
base85 string: byte-shuffled then zlib-compressed.

Reconstruct the matrix::

    matrix = decode_matrix(model.data, model.n_frames, model.band_count)
    energy = matrix[:, :model.band_count]   # shape (n_frames, band_count)
    delta  = matrix[:, model.band_count:]   # shape (n_frames, band_count)

Derivable from stored fields::

    duration_s  = n_frames * hop_size / analysis_rate
    band_labels = band_center_frequencies(analysis_rate, band_count)
"""

from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import tomli_w
from pydantic import BaseModel, ConfigDict

from part_io.adapters.audio.matcher import (
    _ANALYSIS_RATE,
    _BAND_COUNT,
    _HOP_SIZE,
    compute_audio_file_profile,
)
from part_io.utils.hash import partial_file_hash

PROFILE_VERSION = 1


# ---------------------------------------------------------------------------
# Encode / decode  (bi-directional transformation layer)
# ---------------------------------------------------------------------------


def encode_matrix(matrix: np.ndarray) -> str:
    """Byte-shuffle, compress, and base85-encode a float32 matrix."""
    raw = matrix.astype(np.float32).tobytes()
    shuffled = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).T.tobytes()
    return base64.b85encode(zlib.compress(shuffled)).decode("ascii")


def decode_matrix(data: str, n_frames: int, band_count: int) -> np.ndarray:
    """Decode a base85 string produced by :func:`encode_matrix` into a float32 matrix."""
    shuffled = zlib.decompress(base64.b85decode(data))
    n_floats = n_frames * band_count * 2
    unshuffled = np.frombuffer(shuffled, dtype=np.uint8).reshape(4, n_floats).T.tobytes()
    return np.frombuffer(unshuffled, dtype=np.float32).copy().reshape(n_frames, band_count * 2)


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------


class SnippetProfileModel(BaseModel):
    """Spectral profile for one snippet audio file.

    ``data`` stores the full detection matrix ``(n_frames, band_count * 2)``
    as a byte-shuffled, zlib-compressed, base85-encoded string.
    Decode with :func:`decode_matrix`.
    """

    model_config = ConfigDict(extra="forbid")

    source_hash: str
    n_frames: int
    analysis_rate: int
    hop_size: int
    band_count: int
    data: str


# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------


class _ProfileData(BaseModel):
    """Internal spectral profile: seed frame plus per-frame temporal deltas.

    Construct via ``_ProfileData(source_hash=..., matrix=<ndarray>)``.
    The ``matrix`` key is consumed by the validator and never stored directly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_hash: str
    seed: np.ndarray  # shape (band_count * 2,) — frame 0 verbatim
    deltas: np.ndarray  # shape (n_frames - 1, band_count * 2) — frame diffs

    @classmethod
    def from_matrix(cls, *, source_hash: str, matrix: np.ndarray) -> "_ProfileData":
        """Build profile data from a full detection matrix."""
        matrix_f32 = np.asarray(matrix, dtype=np.float32)
        return cls(
            source_hash=source_hash,
            seed=matrix_f32[0].copy(),
            deltas=np.diff(matrix_f32, axis=0),
        )

    @property
    def n_frames(self) -> int:
        return len(self.deltas) + 1

    def to_matrix(self) -> np.ndarray:
        """Reconstruct the full (n_frames, band_count * 2) float32 matrix."""
        frames = np.empty((self.n_frames, self.seed.shape[0]), dtype=np.float32)
        frames[0] = self.seed
        frames[1:] = self.seed + np.cumsum(self.deltas, axis=0)
        return frames


def _compute(snippet_path: Path) -> _ProfileData:
    profile = compute_audio_file_profile(snippet_path)
    if profile.size == 0:
        raise ValueError(
            f"Could not compute profile for {snippet_path}: audio too short or decode failed"
        )
    return _ProfileData.from_matrix(source_hash=partial_file_hash(snippet_path), matrix=profile)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_profile_model(d: _ProfileData) -> SnippetProfileModel:
    return SnippetProfileModel(
        source_hash=d.source_hash,
        n_frames=d.n_frames,
        analysis_rate=_ANALYSIS_RATE,
        hop_size=_HOP_SIZE,
        band_count=_BAND_COUNT,
        data=encode_matrix(d.to_matrix()),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_PROFILE_HEADER = (
    "# Re-generate with: poetry run part-io-tasks audio-snippet-profile <path>\n"
    "#\n"
    "# Decode the matrix (Python):\n"
    "#   from part_io.adapters.audio.snippet_profile import decode_matrix\n"
    "#   matrix = decode_matrix(model.data, model.n_frames, model.band_count)\n"
    "#   energy, delta = matrix[:, :band_count], matrix[:, band_count:]\n"
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


def snapshot_snippet_profile(snippet_path: Path) -> SnippetProfileModel:
    """Compute a spectral profile model for *snippet_path* without writing files."""
    d = _compute(snippet_path)
    return _build_profile_model(d)


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
    "encode_matrix",
    "decode_matrix",
    "snapshot_snippet_profile",
    "write_snippet_profile",
    "is_profile_current",
    "PROFILE_VERSION",
]
