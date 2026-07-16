"""Audio sample matching helpers built around ffmpeg exports.

The detector converts both inputs to mono PCM data and compares normalized
feature sequences over fixed windows. Features are 32-band spectral-energy
vectors concatenated with first-order delta features (64 dimensions total)
over a 16 kHz analysis stream.

Band energies are mean-centered on the *source* recording's average spectrum
before normalization. Uncentered log-spectra share a large common component
(the spectral tilt of all natural audio), which compresses every window score
toward 1.0 and leaves true matches statistically indistinguishable from
speech. After centering, scores behave like correlations: ~0.0 against
unrelated audio and approaching 1.0 for a true occurrence, so fixed
thresholds and prominence z-scores are both meaningful. A consequence is
that scores are source-dependent: the same reference scored against two
different recordings is centered on each recording's own mean spectrum.
"""

from __future__ import annotations

import itertools
from array import array
from dataclasses import dataclass
from functools import cache
from pathlib import Path  # noqa: TC003

import numpy as np

from partio.adapters.process.runner import run_resolved

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


@dataclass(frozen=True)
class BestMatch:
    """The single most likely sample occurrence, with its peak prominence.

    ``prominence`` is the z-score of the similarity peak against the rest of the
    curve (``(peak - median) / std``). It is meaningful when absolute scores are
    compressed -- as they are for speech-heavy audio -- because it measures how
    far the peak stands out from that source's own baseline rather than relying
    on a fixed threshold.
    """

    start_seconds: float
    end_seconds: float
    duration_seconds: float
    score: float
    prominence: float


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


@cache
def _build_filterbank_matrix(
    sample_rate: int,
    frame_size: int = _FRAME_SIZE,
    band_count: int = _BAND_COUNT,
) -> np.ndarray:
    """Build a cached rectangular filterbank for the analysis FFT bins."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    nyquist = sample_rate / 2
    band_edges_hz = np.geomspace(20.0, nyquist, band_count + 1)
    freq_hz = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    filterbank = np.zeros((len(freq_hz), band_count), dtype=np.float32)
    for band_index, (left, right) in enumerate(itertools.pairwise(band_edges_hz)):
        band_bins = np.where((freq_hz >= left) & (freq_hz < right))[0]
        if band_bins.size == 0:
            nearest = int(np.argmin(np.abs(freq_hz - (left + right) / 2)))
            filterbank[nearest, band_index] = 1.0
            continue

        filterbank[band_bins, band_index] = 1.0 / band_bins.size

    return filterbank


def _stack_temporal_deltas(base_features: np.ndarray) -> np.ndarray:
    deltas = np.diff(base_features, axis=0, prepend=base_features[:1])
    return np.concatenate([base_features, deltas], axis=1)


def _band_energy_matrix(samples: list[int], sample_rate: int) -> np.ndarray:
    """Compute raw log band energies per analysis frame (frames x bands)."""
    if len(samples) < _FRAME_SIZE:
        return np.zeros((0, _BAND_COUNT), dtype=np.float32)

    sample_array = np.asarray(samples, dtype=np.float32)
    window = np.hanning(_FRAME_SIZE).astype(np.float32)
    filterbank = _build_filterbank_matrix(sample_rate)

    raw_bands: list[np.ndarray] = []
    for index in range(0, len(sample_array) - _FRAME_SIZE + 1, _HOP_SIZE):
        frame = sample_array[index : index + _FRAME_SIZE] * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        raw_bands.append(np.log1p(spectrum @ filterbank).astype(np.float32))

    return np.stack(raw_bands)


def _finalize_profile(band_matrix: np.ndarray, mean_bands: np.ndarray) -> np.ndarray:
    """Center band energies on *mean_bands*, L2-normalize frames, append deltas.

    Centering removes the spectral tilt shared by all natural audio, which
    otherwise dominates the window dot product and compresses every score
    toward 1.0 regardless of whether the reference is actually present.
    """
    centered = band_matrix - mean_bands
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return _stack_temporal_deltas(centered / norms)


def _window_scores(reference: np.ndarray, source_profile: np.ndarray, hop: int) -> np.ndarray:
    """Score every hop-th sliding window of *source_profile* against *reference*."""
    windowed_profiles = np.lib.stride_tricks.sliding_window_view(
        source_profile,
        window_shape=reference.shape[0],
        axis=0,
    )
    windowed_profiles = np.swapaxes(windowed_profiles, 1, 2)[::hop]
    return np.mean(np.sum(windowed_profiles * reference[None, :, :], axis=2), axis=1)


def _build_match_candidates(  # noqa: PLR0913
    *,
    reference: np.ndarray,
    source_profile: np.ndarray,
    sample_duration: float,
    frame_hop_seconds: float,
    hop: int,
    score_threshold: float,
) -> list[AudioMatch]:
    scores = _window_scores(reference, source_profile, hop)

    matches: list[AudioMatch] = []
    for start_index, score in enumerate(scores):
        if score < score_threshold:
            continue

        start_seconds = start_index * hop * frame_hop_seconds
        matches.append(
            AudioMatch(
                start_seconds=round(start_seconds, 3),
                end_seconds=round(start_seconds + sample_duration, 3),
                duration_seconds=round(sample_duration, 3),
                score=round(float(score), 4),
            )
        )
    return matches


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


def _validate_match_inputs(source_path: Path, sample_path: Path, step_seconds: float) -> None:
    """Validate the inputs shared by the public matching entry points."""
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")


def find_audio_sample_matches(
    *,
    source_path: Path,
    sample_path: Path,
    score_threshold: float = 0.8,
    step_seconds: float = 0.1,
    dedupe_overlap: float = 0.5,
) -> list[AudioMatch]:
    """Find likely occurrences of *sample_path* inside *source_path*.

    The matcher works on short energy fingerprints, which is enough for a first
    deterministic pass and keeps the implementation dependency-light. Both
    profiles are centered on the source's mean spectrum, so *score_threshold*
    acts on correlation-like scores (~0.0 baseline, ~1.0 for a true match).
    """
    _validate_match_inputs(source_path, sample_path, step_seconds)
    if not 0 <= dedupe_overlap <= 1:
        raise ValueError("dedupe_overlap must be in [0, 1]")

    source_samples = _decode_pcm_mono_16k(source_path)
    sample_samples = _decode_pcm_mono_16k(sample_path)

    sample_rate = _ANALYSIS_RATE
    source_bands = _band_energy_matrix(source_samples, sample_rate)
    sample_bands = _band_energy_matrix(sample_samples, sample_rate)
    if sample_bands.size == 0 or source_bands.size == 0:
        return []
    if source_bands.shape[0] < sample_bands.shape[0]:
        return []

    mean_bands = source_bands.mean(axis=0)
    reference = _finalize_profile(sample_bands, mean_bands)
    source_profile = _finalize_profile(source_bands, mean_bands)

    frame_hop_seconds = _HOP_SIZE / sample_rate
    hop = max(1, int(step_seconds / frame_hop_seconds))
    sample_duration = len(sample_samples) / sample_rate
    matches = _build_match_candidates(
        reference=reference,
        source_profile=source_profile,
        sample_duration=sample_duration,
        frame_hop_seconds=frame_hop_seconds,
        hop=hop,
        score_threshold=score_threshold,
    )

    return _suppress_overlapping(matches, min_overlap=dedupe_overlap)


def find_best_sample_match(
    *,
    source_path: Path,
    sample_path: Path,
    step_seconds: float = 0.1,
    search_seconds: float | None = None,
) -> BestMatch | None:
    """Locate the single best occurrence of *sample_path* inside *source_path*.

    Reports the global peak of the similarity curve plus its prominence rather
    than every window above a fixed threshold, which stays robust when scores
    are compressed (common for speech-heavy audio). Pass *search_seconds* to
    restrict the scan to the first N seconds (e.g. an intro region).
    """
    _validate_match_inputs(source_path, sample_path, step_seconds)
    if search_seconds is not None and search_seconds <= 0:
        raise ValueError("search_seconds must be positive")

    sample_samples = _decode_pcm_mono_16k(sample_path)
    sample_rate = _ANALYSIS_RATE
    source_bands = _band_energy_matrix(_decode_pcm_mono_16k(source_path), sample_rate)
    sample_bands = _band_energy_matrix(sample_samples, sample_rate)
    if sample_bands.size == 0 or source_bands.size == 0:
        return None
    if source_bands.shape[0] < sample_bands.shape[0]:
        return None

    mean_bands = source_bands.mean(axis=0)
    reference = _finalize_profile(sample_bands, mean_bands)
    source_profile = _finalize_profile(source_bands, mean_bands)

    frame_hop_seconds = _HOP_SIZE / sample_rate
    hop = max(1, int(step_seconds / frame_hop_seconds))
    scores = _window_scores(reference, source_profile, hop)
    if scores.size == 0:
        return None

    # Prominence baseline is taken over the whole curve so it stays stable, while
    # the search window only constrains *where* the peak is selected.
    spread = float(np.std(scores))
    baseline = float(np.median(scores))
    step = hop * frame_hop_seconds

    limit = int(search_seconds / step) if search_seconds is not None else 0
    search_scores = scores[:limit] if limit >= 1 else scores

    peak_index = int(np.argmax(search_scores))
    peak = float(search_scores[peak_index])
    prominence = 0.0 if spread == 0 else (peak - baseline) / spread

    start_seconds = peak_index * step
    sample_duration = len(sample_samples) / sample_rate
    return BestMatch(
        start_seconds=round(start_seconds, 3),
        end_seconds=round(start_seconds + sample_duration, 3),
        duration_seconds=round(sample_duration, 3),
        score=round(peak, 4),
        prominence=round(prominence, 4),
    )


__all__ = ["AudioMatch", "_suppress_overlapping", "find_audio_sample_matches"]
