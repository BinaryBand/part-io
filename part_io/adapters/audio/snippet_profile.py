"""Compute and persist spectral fingerprints for snippet audio files.

``.profile.toml`` is written alongside each snippet.  It serialises a
:class:`SnippetProfileModel` whose ``[data]`` section stores the full
``(n_frames, band_count * 2)`` float32 detection matrix as zlib-compressed
base64, split into three named parts:

``header``
    The 2-byte zlib CMF+FLG preamble (compression method and flags).
``body``
    The raw deflate stream as a TOML array of 76-character base64 chunks
    (57 raw bytes each — standard PEM line width).
``signature``
    The 4-byte Adler-32 checksum of the uncompressed data.

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
from dataclasses import dataclass
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
# Compressed-data model
# ---------------------------------------------------------------------------


class CompressedData(BaseModel):
    """zlib-compressed float32 matrix, split into its three structural parts.

    Reassemble with ``header + body + signature`` before passing to
    ``zlib.decompress``.  The Adler-32 ``signature`` is verified automatically
    during decompression.
    """

    model_config = ConfigDict(extra="forbid")

    header: str  # base64 of 2-byte zlib CMF+FLG preamble
    body: str  # base85-encoded raw deflate stream
    signature: str  # base64 of 4-byte Adler-32 checksum


# ---------------------------------------------------------------------------
# Encode / decode  (bi-directional transformation layer)
# ---------------------------------------------------------------------------


def encode_matrix(matrix: np.ndarray) -> CompressedData:
    """Compress a float32 matrix and return its split :class:`CompressedData`."""
    compressed = zlib.compress(matrix.astype(np.float32).tobytes())
    header_bytes = compressed[:2]
    signature_bytes = compressed[-4:]
    body_bytes = compressed[2:-4]
    return CompressedData(
        header=base64.b64encode(header_bytes).decode("ascii"),
        body=base64.b85encode(body_bytes).decode("ascii"),
        signature=base64.b64encode(signature_bytes).decode("ascii"),
    )


def decode_matrix(data: CompressedData, n_frames: int, band_count: int) -> np.ndarray:
    """Reassemble and decompress a :class:`CompressedData` into a float32 matrix."""
    compressed = (
        base64.b64decode(data.header)
        + base64.b85decode(data.body)
        + base64.b64decode(data.signature)
    )
    flat = np.frombuffer(zlib.decompress(compressed), dtype=np.float32).copy()
    return flat.reshape(n_frames, band_count * 2)


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------


class SnippetProfileModel(BaseModel):
    """Spectral profile for one snippet audio file.

    ``data`` stores the full detection matrix ``(n_frames, band_count * 2)``
    as a zlib-compressed, split :class:`CompressedData`.
    Decode with :func:`decode_matrix`.
    """

    model_config = ConfigDict(extra="forbid")

    source_hash: str
    n_frames: int
    analysis_rate: int
    hop_size: int
    band_count: int
    data: CompressedData


# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProfileData:
    source_hash: str
    n_frames: int
    energy: np.ndarray  # shape (n_frames, band_count)
    delta: np.ndarray  # shape (n_frames, band_count)


def _compute(snippet_path: Path) -> _ProfileData:
    profile = compute_audio_file_profile(snippet_path)
    if profile.size == 0:
        raise ValueError(
            f"Could not compute profile for {snippet_path}: audio too short or decode failed"
        )
    n_frames, _ = profile.shape
    return _ProfileData(
        source_hash=partial_file_hash(snippet_path),
        n_frames=n_frames,
        energy=profile[:, :_BAND_COUNT],
        delta=profile[:, _BAND_COUNT:],
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_profile_model(d: _ProfileData) -> SnippetProfileModel:
    matrix = np.concatenate([d.energy, d.delta], axis=1)
    return SnippetProfileModel(
        source_hash=d.source_hash,
        n_frames=d.n_frames,
        analysis_rate=_ANALYSIS_RATE,
        hop_size=_HOP_SIZE,
        band_count=_BAND_COUNT,
        data=encode_matrix(matrix),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_PROFILE_HEADER = (
    "# Re-generate with: poetry run part-io-tasks snippet-profile <path>\n"
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
    "CompressedData",
    "encode_matrix",
    "decode_matrix",
    "write_snippet_profile",
    "is_profile_current",
    "PROFILE_VERSION",
]
