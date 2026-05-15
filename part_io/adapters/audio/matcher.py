"""Audio sample matching helpers built around ffmpeg exports.

The detector converts both inputs to mono PCM data and compares normalized
feature sequences over fixed windows. When numpy is available, features are
32-band spectral-energy vectors concatenated with first-order delta features
(64 dimensions total) over a 16 kHz analysis stream; otherwise, it falls back
to a scalar energy profile while preserving the same API.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from pathlib import Path

from part_io.adapters.process.runner import run_resolved

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - environment dependent import
    np = None
    _HAS_NUMPY = False


_ANALYSIS_RATE = 16000
_FRAME_SIZE = 2048
_HOP_SIZE = 1024
_BAND_COUNT = 32


@dataclass(frozen=True)
class AudioMatch:
    """One detected sample occurrence in a longer audio file."""

    start_seconds: float
    end_seconds: float
    duration_seconds: float
    score: float


def _decode_pcm_mono_16k(source: Path) -> list[int]:
    """Decode *source* to signed 16-bit PCM samples at a low analysis rate."""
    result = run_resolved(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(_ANALYSIS_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"ffmpeg failed to decode audio: {source}")

    raw = getattr(result, "stdout", b"")
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError(f"No decoded audio produced for: {source}")

    samples = array("h")
    samples.frombytes(raw)
    return list(samples)


def _build_scalar_profile(samples: list[int], block_size: int, hop_size: int) -> list[list[float]]:
    """Build a fallback scalar-energy feature profile."""
    if block_size <= 0 or hop_size <= 0:
        raise ValueError("block_size and hop_size must be positive")
    if len(samples) < block_size:
        return []

    profile: list[list[float]] = []
    for index in range(0, len(samples) - block_size + 1, hop_size):
        block = samples[index : index + block_size]
        profile.append([sum(abs(value) for value in block) / block_size])
    return profile


def _stack_temporal_deltas(base_features: np.ndarray) -> np.ndarray:
    deltas = np.diff(base_features, axis=0, prepend=base_features[:1])
    return np.concatenate([base_features, deltas], axis=1)


def _build_spectral_profile(samples: list[int], sample_rate: int) -> list[list[float]]:
    """Build a multi-band spectral-energy feature profile.

    This path requires numpy and computes log-spaced bands over each frame.
    """
    if not _HAS_NUMPY:
        return _build_scalar_profile(samples, _FRAME_SIZE, _HOP_SIZE)
    if len(samples) < _FRAME_SIZE:
        return []

    sample_array = np.asarray(samples, dtype=np.float32)
    window = np.hanning(_FRAME_SIZE).astype(np.float32)
    nyquist = sample_rate / 2
    band_edges_hz = np.geomspace(20.0, nyquist, _BAND_COUNT + 1)
    freq_hz = np.fft.rfftfreq(_FRAME_SIZE, d=1.0 / sample_rate)

    bands: list[np.ndarray] = []
    for left, right in zip(band_edges_hz[:-1], band_edges_hz[1:], strict=False):
        mask = np.where((freq_hz >= left) & (freq_hz < right))[0]
        if mask.size == 0:
            nearest = int(np.argmin(np.abs(freq_hz - (left + right) / 2)))
            mask = np.array([nearest])
        bands.append(mask)

    raw_bands: list[np.ndarray] = []
    for index in range(0, len(sample_array) - _FRAME_SIZE + 1, _HOP_SIZE):
        frame = sample_array[index : index + _FRAME_SIZE] * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        vector = np.array([np.log1p(np.mean(spectrum[band])) for band in bands], dtype=np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        raw_bands.append(vector / norm)

    if not raw_bands:
        return []

    band_matrix = np.stack(raw_bands)
    features = _stack_temporal_deltas(band_matrix)

    profile: list[list[float]] = [row.tolist() for row in features]
    return profile


def _z_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    stddev = math.sqrt(variance) or 1.0
    return [(value - mean) / stddev for value in values]


def _normalized_correlation(reference: list[float], window: list[float]) -> float:
    if not reference or not window or len(reference) != len(window):
        return -1.0
    ref_norm = _z_normalize(reference)
    win_norm = _z_normalize(window)
    if not ref_norm or not win_norm:
        return -1.0
    return sum(left * right for left, right in zip(ref_norm, win_norm)) / len(ref_norm)


def _flatten_features(vectors: list[list[float]]) -> list[float]:
    return [value for vector in vectors for value in vector]


def _normalized_similarity(reference: list[list[float]], window: list[list[float]]) -> float:
    """Compute normalized cosine-like similarity on flattened feature sequences."""
    if not reference or not window or len(reference) != len(window):
        return -1.0
    flat_ref = _flatten_features(reference)
    flat_win = _flatten_features(window)
    return _normalized_correlation(flat_ref, flat_win)


def _overlap_ratio(left: AudioMatch, right: AudioMatch) -> float:
    overlap_start = max(left.start_seconds, right.start_seconds)
    overlap_end = min(left.end_seconds, right.end_seconds)
    overlap = max(0.0, overlap_end - overlap_start)
    shortest = min(left.duration_seconds, right.duration_seconds)
    if shortest <= 0:
        return 0.0
    return overlap / shortest


def _suppress_overlapping(matches: list[AudioMatch], min_overlap: float = 0.5) -> list[AudioMatch]:
    """Keep highest-score match among heavily overlapping candidates."""
    if not matches:
        return []
    kept: list[AudioMatch] = []
    for candidate in sorted(matches, key=lambda item: item.score, reverse=True):
        if any(_overlap_ratio(candidate, accepted) > min_overlap for accepted in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: item.start_seconds)


def find_audio_sample_matches(
    *,
    source_path: Path,
    sample_path: Path,
    score_threshold: float = 0.8,
    step_seconds: float = 0.05,
    dedupe_overlap: float = 0.5,
) -> list[AudioMatch]:
    """Find likely occurrences of *sample_path* inside *source_path*.

    The matcher works on short energy fingerprints, which is enough for a first
    deterministic pass and keeps the implementation dependency-light.
    """
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if not 0 <= dedupe_overlap <= 1:
        raise ValueError("dedupe_overlap must be in [0, 1]")

    source_samples = _decode_pcm_mono_16k(source_path)
    sample_samples = _decode_pcm_mono_16k(sample_path)

    sample_rate = _ANALYSIS_RATE
    reference = _build_spectral_profile(sample_samples, sample_rate)
    if not reference:
        return []

    source_profile = _build_spectral_profile(source_samples, sample_rate)
    if not source_profile:
        return []

    frame_hop_seconds = _HOP_SIZE / sample_rate
    hop = max(1, int(step_seconds / frame_hop_seconds))
    matches: list[AudioMatch] = []
    sample_duration = len(sample_samples) / sample_rate

    for start_index in range(0, len(source_profile) - len(reference) + 1, hop):
        window = source_profile[start_index : start_index + len(reference)]
        score = _normalized_similarity(reference, window)
        if score < score_threshold:
            continue

        start_seconds = start_index * frame_hop_seconds
        matches.append(
            AudioMatch(
                start_seconds=round(start_seconds, 3),
                end_seconds=round(start_seconds + sample_duration, 3),
                duration_seconds=round(sample_duration, 3),
                score=round(score, 4),
            )
        )

    return _suppress_overlapping(matches, min_overlap=dedupe_overlap)


__all__ = ["AudioMatch", "find_audio_sample_matches", "_suppress_overlapping"]
