"""Audio sample matching helpers built around ffmpeg exports.

The detector converts both inputs to mono PCM data and compares normalized
feature sequences over fixed windows. Features are 32-band spectral-energy
vectors concatenated with first-order delta features (64 dimensions total)
over a 16 kHz analysis stream.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from part_io.adapters.process.runner import run_resolved
from part_io.utils.timing import Timer

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


def _run_pcm_decode(command: list[str]) -> list[int]:
    """Execute an ffmpeg PCM decode *command* and return signed 16-bit samples."""
    with Timer("matcher._run_pcm_decode"):
        result = run_resolved(command, capture_output=True)
    if result.returncode != 0:
        return []
    raw = getattr(result, "stdout", b"")
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return []
    samples = array("h")
    samples.frombytes(raw)
    return list(samples)


@cache
def _decode_pcm_mono_16k(source: Path) -> list[int]:
    """Decode *source* to signed 16-bit PCM samples at a low analysis rate."""
    with Timer("matcher._decode_pcm_mono_16k"):
        samples = _run_pcm_decode(
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
            ]
        )
    if not samples:
        raise ValueError(f"ffmpeg failed to decode audio: {source}")
    return samples


def _decode_pcm_mono_16k_window(
    source: Path,
    start_seconds: float,
    end_seconds: float,
) -> list[int]:
    """Decode a time-bounded window of *source* to signed 16-bit PCM at 16 kHz."""
    duration = max(0.0, end_seconds - start_seconds)
    if duration <= 0:
        return []
    with Timer("matcher._decode_pcm_mono_16k_window"):
        return _run_pcm_decode(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_seconds:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                str(_ANALYSIS_RATE),
                "-f",
                "s16le",
                "pipe:1",
            ]
        )


@cache
def _build_filterbank_matrix(
    sample_rate: int,
    frame_size: int = _FRAME_SIZE,
    band_count: int = _BAND_COUNT,
) -> np.ndarray:
    """Build a cached rectangular filterbank for the analysis FFT bins."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    with Timer("matcher._build_filterbank_matrix"):
        nyquist = sample_rate / 2
    band_edges_hz = np.geomspace(20.0, nyquist, band_count + 1)
    freq_hz = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    filterbank = np.zeros((len(freq_hz), band_count), dtype=np.float32)
    for band_index, (left, right) in enumerate(
        zip(band_edges_hz[:-1], band_edges_hz[1:], strict=False)
    ):
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


def _build_spectral_profile(samples: list[int], sample_rate: int) -> list[list[float]]:
    """Build a multi-band spectral-energy feature profile.

    This path computes cached band energies plus first-order deltas.
    """
    with Timer("matcher._build_spectral_profile"):
        if len(samples) < _FRAME_SIZE:
            return []

        sample_array = np.asarray(samples, dtype=np.float32)
        window = np.hanning(_FRAME_SIZE).astype(np.float32)
        filterbank = _build_filterbank_matrix(sample_rate)

        raw_bands: list[np.ndarray] = []
        for index in range(0, len(sample_array) - _FRAME_SIZE + 1, _HOP_SIZE):
            frame = sample_array[index : index + _FRAME_SIZE] * window
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            vector = np.log1p(spectrum @ filterbank).astype(np.float32)
            norm = float(np.linalg.norm(vector)) or 1.0
            raw_bands.append(vector / norm)

        if not raw_bands:
            return []

        band_matrix = np.stack(raw_bands)
        features = _stack_temporal_deltas(band_matrix)

        profile: list[list[float]] = [row.tolist() for row in features]
        return profile


def _cross_correlation_search(reference: np.ndarray, source_profile: np.ndarray, hop: int):
    with Timer("matcher._cross_correlation_search"):
        n = source_profile.shape[0]
        m = reference.shape[0]

        fft_size = int(2 ** np.ceil(np.log2(n + m)))
        src_fft = np.fft.rfft(source_profile, n=fft_size, axis=0)
        ref_fft = np.fft.rfft(reference, n=fft_size, axis=0)
        corr_all = np.fft.irfft((np.conj(ref_fft) * src_fft).sum(axis=1), n=fft_size)
        scores = corr_all[: n - m + 1][::hop] / m
        return scores


def _windowed_search(reference: np.ndarray, source_profile: np.ndarray, hop: int):
    with Timer("matcher._windowed_search"):
        windowed_profiles = np.lib.stride_tricks.sliding_window_view(
            source_profile,
            window_shape=reference.shape[0],
            axis=0,
        )
        windowed_profiles = np.swapaxes(windowed_profiles, 1, 2)[::hop]
        scores = np.mean(np.sum(windowed_profiles * reference[None, :, :], axis=2), axis=1)
        return scores


def _build_match_candidates(
    *,
    reference: np.ndarray,
    source_profile: np.ndarray,
    sample_duration: float,
    frame_hop_seconds: float,
    hop: int,
    score_threshold: float,
    frame_offset_seconds: float = 0.0,
    z_threshold: float | None = None,
) -> list[AudioMatch]:
    with Timer("matcher._build_match_candidates"):
        scores = _cross_correlation_search(reference, source_profile, hop)

    effective_threshold = score_threshold
    if z_threshold is not None and scores.size > 1:
        std = float(np.std(scores))
        if std > 0:
            effective_threshold = max(score_threshold, float(np.mean(scores)) + z_threshold * std)

    matches: list[AudioMatch] = []
    for start_index, score in enumerate(scores):
        if score < effective_threshold:
            continue

        start_seconds = frame_offset_seconds + start_index * hop * frame_hop_seconds
        matches.append(
            AudioMatch(
                start_seconds=round(start_seconds, 3),
                end_seconds=round(start_seconds + sample_duration, 3),
                duration_seconds=round(sample_duration, 3),
                score=round(float(score), 4),
            )
        )
    return matches


def _normalized_similarity(reference: np.ndarray, window: np.ndarray) -> float:
    """Compute the mean frame-wise cosine similarity between two feature windows."""
    if reference.ndim != 2 or window.ndim != 2 or reference.shape != window.shape:
        return -1.0
    return float(np.mean(np.sum(reference * window, axis=1)))


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


@cache
def _get_source_profile(source_path: Path) -> np.ndarray:
    """Build and cache the full spectral profile for *source_path*.

    Caching ensures the coarse pass and every subsequent refine call share the
    same profile array, avoiding redundant ffmpeg decodes and FFT work.
    """
    with Timer("matcher._get_source_profile"):
        samples = _decode_pcm_mono_16k(source_path)
        return np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)


def _validate_match_search_inputs(
    source_path: Path,
    sample_path: Path,
    step_seconds: float,
    dedupe_overlap: float,
) -> None:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if not 0 <= dedupe_overlap <= 1:
        raise ValueError("dedupe_overlap must be in [0, 1]")


def _prepare_match_search(
    *,
    source_path: Path,
    sample_path: Path,
    search_start_seconds: float | None,
    search_end_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    sample_samples = _decode_pcm_mono_16k(sample_path)
    sample_rate = _ANALYSIS_RATE
    reference = np.asarray(_build_spectral_profile(sample_samples, sample_rate), dtype=np.float32)
    if reference.size == 0:
        return reference, np.asarray([], dtype=np.float32), sample_rate, 0.0

    full_profile = _get_source_profile(source_path)
    if full_profile.size == 0 or full_profile.shape[0] < reference.shape[0]:
        return reference, np.asarray([], dtype=np.float32), sample_rate, 0.0

    frame_hop_seconds = _HOP_SIZE / sample_rate
    if search_start_seconds is not None and search_end_seconds is not None:
        start_frame = max(0, int(search_start_seconds / frame_hop_seconds))
        end_frame = min(
            full_profile.shape[0],
            int(search_end_seconds / frame_hop_seconds) + reference.shape[0],
        )
        source_profile = full_profile[start_frame:end_frame]
        frame_offset_seconds = start_frame * frame_hop_seconds
    else:
        source_profile = full_profile
        frame_offset_seconds = 0.0

    sample_duration = len(sample_samples) / sample_rate
    return reference, source_profile, sample_duration, frame_offset_seconds


def find_audio_sample_matches(
    *,
    source_path: Path,
    sample_path: Path,
    score_threshold: float = 0.8,
    step_seconds: float = 0.1,
    dedupe_overlap: float = 0.5,
    search_start_seconds: float | None = None,
    search_end_seconds: float | None = None,
    z_threshold: float | None = None,
) -> list[AudioMatch]:
    """Find likely occurrences of *sample_path* inside *source_path*.

    The matcher works on short energy fingerprints, which is enough for a first
    deterministic pass and keeps the implementation dependency-light.

    When *search_start_seconds* and *search_end_seconds* are both provided the
    full source profile is still used (preserving delta-feature accuracy at
    boundaries) but only the relevant frame slice is scored, which is much
    cheaper for local refinement searches.
    """
    _validate_match_search_inputs(source_path, sample_path, step_seconds, dedupe_overlap)

    reference, source_profile, sample_duration, frame_offset_seconds = _prepare_match_search(
        source_path=source_path,
        sample_path=sample_path,
        search_start_seconds=search_start_seconds,
        search_end_seconds=search_end_seconds,
    )
    if reference.size == 0 or source_profile.size == 0:
        return []

    if source_profile.shape[0] < reference.shape[0]:
        return []

    sample_rate = _ANALYSIS_RATE
    frame_hop_seconds = _HOP_SIZE / sample_rate
    hop = max(1, int(step_seconds / frame_hop_seconds))
    matches = _build_match_candidates(
        reference=reference,
        source_profile=source_profile,
        sample_duration=sample_duration,
        frame_hop_seconds=frame_hop_seconds,
        hop=hop,
        score_threshold=score_threshold,
        frame_offset_seconds=frame_offset_seconds,
        z_threshold=z_threshold,
    )

    return _suppress_overlapping(matches, min_overlap=dedupe_overlap)


def _shift_match(match: AudioMatch, new_start_seconds: float) -> AudioMatch:
    """Return a copy of *match* with start_seconds set to *new_start_seconds*."""
    start = round(new_start_seconds, 3)
    return AudioMatch(
        start_seconds=start,
        end_seconds=round(start + match.duration_seconds, 3),
        duration_seconds=match.duration_seconds,
        score=match.score,
    )


def anchor_to_onset(
    *,
    match: AudioMatch,
    source_path: Path,
    onset_energy_ratio: float = 0.20,
    smoothing_window: int = 400,
) -> AudioMatch:
    """Shift *match* start_seconds to the first significant energy onset.

    Extracts raw PCM for the matched window, builds a smoothed absolute-value
    energy envelope, and finds the first sample whose energy exceeds
    ``onset_energy_ratio * peak_energy``.  Returns the original match
    unchanged when the window is empty or no clear onset is detected.
    """
    window_samples = _decode_pcm_mono_16k_window(
        source_path, match.start_seconds, match.end_seconds
    )
    if not window_samples:
        return match

    arr = np.abs(np.asarray(window_samples, dtype=np.float32))
    kernel_size = max(1, min(smoothing_window, len(arr)))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    envelope = np.convolve(arr, kernel, mode="same")

    peak_energy = float(envelope.max())
    if peak_energy <= 0:
        return match

    threshold = onset_energy_ratio * peak_energy
    onset_indices = np.where(envelope >= threshold)[0]
    if not onset_indices.size:
        return match

    offset_seconds = int(onset_indices[0]) / _ANALYSIS_RATE
    return _shift_match(match, match.start_seconds + offset_seconds)


def cross_correlate_align(
    *,
    match: AudioMatch,
    source_path: Path,
    sample_path: Path,
    padding_seconds: float = 2.0,
) -> AudioMatch:
    """Refine alignment via waveform cross-correlation (Phase 3 --precise).

    Extracts raw PCM for the candidate region (with padding) and the
    reference sample, computes the normalised cross-correlation, and shifts
    ``start_seconds`` by the lag that maximises correlation.  Returns the
    original match unchanged when PCM extraction fails or signals are silent.
    """
    window_start = max(0.0, match.start_seconds - padding_seconds)
    window_end = match.end_seconds + padding_seconds
    source_window = _decode_pcm_mono_16k_window(source_path, window_start, window_end)
    sample_pcm = _decode_pcm_mono_16k(sample_path)

    if not source_window or not sample_pcm:
        return match

    source_arr = np.asarray(source_window, dtype=np.float32)
    sample_arr = np.asarray(sample_pcm, dtype=np.float32)

    src_norm = float(np.linalg.norm(source_arr))
    smp_norm = float(np.linalg.norm(sample_arr))
    if src_norm == 0 or smp_norm == 0:
        return match

    source_arr = source_arr / src_norm
    sample_arr = sample_arr / smp_norm

    # O(N log N) cross-correlation via FFT
    n = len(source_arr) + len(sample_arr) - 1
    fft_size = 1 << (n - 1).bit_length()  # next power of two
    src_fft = np.fft.rfft(source_arr, n=fft_size)
    smp_fft = np.fft.rfft(sample_arr, n=fft_size)
    corr = np.fft.irfft(src_fft * np.conj(smp_fft), n=fft_size)[:n]
    peak_index = int(np.argmax(corr))
    # lag > 0  →  sample starts lag samples into the source window
    lag_samples = peak_index - (len(sample_arr) - 1)
    lag_seconds = lag_samples / _ANALYSIS_RATE
    return _shift_match(match, max(0.0, window_start + lag_seconds))


__all__ = [
    "AudioMatch",
    "find_audio_sample_matches",
    "anchor_to_onset",
    "cross_correlate_align",
    "_get_source_profile",
    "_suppress_overlapping",
]
