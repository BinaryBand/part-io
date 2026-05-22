"""Audio sample matching helpers built around ffmpeg exports.

The detector converts both inputs to mono PCM data and compares normalized
feature sequences over fixed windows. Features are 32-band spectral-energy
vectors concatenated with first-order delta features (64 dimensions total)
over a 16 kHz analysis stream.
"""

from __future__ import annotations

import logging
import math
from array import array
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from part_io.adapters.process.runner import run_resolved, run_resolved_with_stderr_callback
from part_io.utils.cache import load_npz_profile, save_npz_profile
from part_io.utils.timing import Timer

_ANALYSIS_RATE = 16000
_FRAME_SIZE = 2048
_HOP_SIZE = 1024
_BAND_COUNT = 32
_LOG = logging.getLogger(__name__)
# ~5-minute source chunks for streaming FFT progress (INFO path only)
_CHUNK_FRAMES: int = _ANALYSIS_RATE * 60 * 5 // _HOP_SIZE


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


def _decode_pcm_mono_16k_streaming(source: Path) -> list[int]:
    """Decode *source* to PCM, streaming ffmpeg progress to stderr.

    Used only for full-file decodes where the caller wants live feedback
    (e.g. profile warm-up over a slow network mount).  Not cached — call
    site is responsible for caching the resulting profile.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "pipe:2",
        "-nostats",
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

    current: dict[str, str] = {}

    def _stream_progress(line: str) -> None:
        if "=" not in line:
            return
        key, _, val = line.partition("=")
        current[key] = val
        if key == "progress":
            out_time = current.get("out_time", "?")
            speed = current.get("speed", "?")
            _LOG.info("[decode] %s  %s", out_time, speed)

    returncode, stdout_data = run_resolved_with_stderr_callback(
        cmd,
        on_stderr_line=_stream_progress,
    )

    if returncode != 0 or not stdout_data:
        return []
    samples = array("h")
    samples.frombytes(stdout_data)
    return list(samples)


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

        frames = np.lib.stride_tricks.sliding_window_view(sample_array, _FRAME_SIZE)[::_HOP_SIZE]
        if frames.shape[0] == 0:
            return []

        spectra = np.abs(np.fft.rfft(frames * window)) ** 2  # batched FFT across all frames
        vectors = np.log1p(spectra @ filterbank)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        band_matrix = (vectors / norms).astype(np.float32)

        features = _stack_temporal_deltas(band_matrix)
        return features.tolist()


def _fmt_duration(seconds: float) -> str:
    """Format *seconds* as a compact human-readable duration string."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


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


def _scores_to_matches(
    scores: np.ndarray,
    *,
    score_threshold: float,
    hop: int,
    frame_hop_seconds: float,
    frame_offset_seconds: float,
    sample_duration: float,
) -> list[AudioMatch]:
    matches: list[AudioMatch] = []
    for start_index, score in enumerate(scores):
        if score < score_threshold:
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


def _build_match_candidates(
    *,
    reference: np.ndarray,
    source_profile: np.ndarray,
    sample_duration: float,
    frame_hop_seconds: float,
    hop: int,
    score_threshold: float,
    frame_offset_seconds: float = 0.0,
) -> list[AudioMatch]:
    n = source_profile.shape[0]
    m = reference.shape[0]

    if not _LOG.isEnabledFor(logging.INFO) or n <= _CHUNK_FRAMES:
        with Timer("matcher._build_match_candidates"):
            scores = _cross_correlation_search(reference, source_profile, hop)
        return _scores_to_matches(
            scores,
            score_threshold=score_threshold,
            hop=hop,
            frame_hop_seconds=frame_hop_seconds,
            frame_offset_seconds=frame_offset_seconds,
            sample_duration=sample_duration,
        )

    # Chunked path: one FFT per ~5-minute block, progress logged after each.
    # Overlap of m-1 frames between chunks ensures no match is missed at a boundary.
    stride = max(1, _CHUNK_FRAMES - (m - 1))
    n_chunks = math.ceil(max(1, n - m + 1) / stride)
    all_matches: list[AudioMatch] = []
    for k in range(n_chunks):
        chunk_start = k * stride
        chunk_end = min(n, chunk_start + _CHUNK_FRAMES)
        chunk_offset = frame_offset_seconds + chunk_start * frame_hop_seconds
        scores = _cross_correlation_search(reference, source_profile[chunk_start:chunk_end], hop)
        all_matches.extend(
            _scores_to_matches(
                scores,
                score_threshold=score_threshold,
                hop=hop,
                frame_hop_seconds=frame_hop_seconds,
                frame_offset_seconds=chunk_offset,
                sample_duration=sample_duration,
            )
        )
        _LOG.info(
            "  [fft]  chunk %d/%d  @%s  %d hit(s) so far",
            k + 1,
            n_chunks,
            _fmt_duration(frame_offset_seconds + chunk_start * frame_hop_seconds),
            len(all_matches),
        )
    return all_matches


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
def _load_cached_profile(source_path: Path, cache_dir: Path) -> np.ndarray | None:
    if cache_dir is None:
        return None
    return load_npz_profile(source_path, cache_dir)


def _save_cached_profile(source_path: Path, profile: np.ndarray, cache_dir: Path) -> None:
    if cache_dir is None:
        return
    save_npz_profile(source_path, profile, cache_dir)


def _get_source_profile(
    source_path: Path,
    cache_dir: Path | None = None,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    """Return the full spectral profile for *source_path*.

    When *cache_dir* is provided the profile is persisted there and reused on
    subsequent calls, skipping the ffmpeg decode and FFT work entirely.
    Invalidated automatically when the source file's mtime or size changes.
    When *show_progress* is True, ffmpeg decode progress is streamed to stderr.
    """
    with Timer("matcher._get_source_profile"):
        if cache_dir is not None:
            cached = _load_cached_profile(source_path, cache_dir)
            if cached is not None:
                return cached
        if show_progress:
            samples = _decode_pcm_mono_16k_streaming(source_path)
        else:
            samples = _decode_pcm_mono_16k(source_path)
        profile = np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)
        if cache_dir is not None:
            _save_cached_profile(source_path, profile, cache_dir)
        return profile


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
    profile_cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    sample_samples = _decode_pcm_mono_16k(sample_path)
    sample_rate = _ANALYSIS_RATE
    reference = np.asarray(_build_spectral_profile(sample_samples, sample_rate), dtype=np.float32)
    if reference.size == 0:
        return reference, np.asarray([], dtype=np.float32), sample_rate, 0.0

    full_profile = _get_source_profile(source_path, cache_dir=profile_cache_dir)
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
    profile_cache_dir: Path | None = None,
) -> list[AudioMatch]:
    """Find likely occurrences of *sample_path* inside *source_path*.

    The matcher works on short energy fingerprints, which is enough for a first
    deterministic pass and keeps the implementation dependency-light.

    When *search_start_seconds* and *search_end_seconds* are both provided the
    full source profile is still used (preserving delta-feature accuracy at
    boundaries) but only the relevant frame slice is scored, which is much
    cheaper for local refinement searches.

    When *profile_cache_dir* is provided the source profile is persisted there
    and reused across runs, skipping the ffmpeg decode and FFT work for
    already-profiled episodes.

    """
    _validate_match_search_inputs(source_path, sample_path, step_seconds, dedupe_overlap)

    reference, source_profile, sample_duration, frame_offset_seconds = _prepare_match_search(
        source_path=source_path,
        sample_path=sample_path,
        search_start_seconds=search_start_seconds,
        search_end_seconds=search_end_seconds,
        profile_cache_dir=profile_cache_dir,
    )
    if reference.size == 0 or source_profile.size == 0:
        return []

    if source_profile.shape[0] < reference.shape[0]:
        return []

    sample_rate = _ANALYSIS_RATE
    frame_hop_seconds = _HOP_SIZE / sample_rate
    hop = max(1, int(step_seconds / frame_hop_seconds))
    n_steps = max(0, (source_profile.shape[0] - reference.shape[0] + 1 + hop - 1) // hop)
    _LOG.info(
        "  [search] %s  source=%s  sample=%.1fs  steps=%d",
        sample_path.stem,
        _fmt_duration(source_profile.shape[0] * frame_hop_seconds),
        sample_duration,
        n_steps,
    )
    matches = _build_match_candidates(
        reference=reference,
        source_profile=source_profile,
        sample_duration=sample_duration,
        frame_hop_seconds=frame_hop_seconds,
        hop=hop,
        score_threshold=score_threshold,
        frame_offset_seconds=frame_offset_seconds,
    )

    matches = _suppress_overlapping(matches, min_overlap=dedupe_overlap)
    _LOG.info("  [match]  %s  %d candidate(s)", sample_path.stem, len(matches))

    return matches


def warm_source_profile(source_path: Path, cache_dir: Path) -> None:
    """Pre-compute and cache the spectral profile for *source_path*.

    Call this once per source file before submitting parallel detection jobs
    so that concurrent workers hit the on-disk cache instead of each launching
    their own full-file ffmpeg decode.  ffmpeg progress is streamed to stderr.
    """
    cached = _load_cached_profile(source_path, cache_dir)
    if cached is not None:
        _LOG.info("[profile] %s — loaded from cache", source_path.stem)
        return
    _LOG.info("[profile] %s — building cache", source_path.stem)
    _get_source_profile(source_path, cache_dir=cache_dir, show_progress=True)


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
    # xcorr(smp, src)[k] peaks at k = start offset of smp in source_window
    lag_seconds = peak_index / _ANALYSIS_RATE
    return _shift_match(match, max(0.0, window_start + lag_seconds))


def build_consensus_profile(
    segments: list[tuple[Path, float, float]],
    *,
    min_segments: int = 2,
) -> np.ndarray | None:
    """Build a mean spectral profile from a list of confirmed (source, start, end) segments.

    Decodes each windowed segment, builds its spectral profile, trims all
    profiles to the shortest frame count, and returns their element-wise mean.
    Returns ``None`` when fewer than *min_segments* valid profiles are built.

    Using an averaged template accounts for natural variation in level,
    EQ, and codec compression across different confirmed occurrences of the
    same audio content — analogous to how production fingerprinters build
    consensus embeddings from multiple aligned examples.
    """
    with Timer("matcher.build_consensus_profile"):
        profiles: list[np.ndarray] = []
        for source, start, end in segments:
            samples = _decode_pcm_mono_16k_window(source, start, end)
            if not samples:
                continue
            profile = np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)
            if profile.size > 0:
                profiles.append(profile)

        if len(profiles) < min_segments:
            return None

        min_len = min(p.shape[0] for p in profiles)
        stack = np.stack([p[:min_len] for p in profiles])  # [N, T, F]
        return stack.mean(axis=0)  # [T, F]


def find_audio_sample_matches_from_profile(
    *,
    source_path: Path,
    reference: np.ndarray,
    score_threshold: float = 0.8,
    step_seconds: float = 0.1,
    dedupe_overlap: float = 0.5,
    search_start_seconds: float | None = None,
    search_end_seconds: float | None = None,
    profile_cache_dir: Path | None = None,
) -> list[AudioMatch]:
    """Like ``find_audio_sample_matches`` but uses a pre-built *reference* profile.

    Use this when a consensus template has been averaged from multiple confirmed
    positive segments rather than decoded from a single reference file.
    The ``sample_duration`` is derived from the reference frame count so it
    accurately reflects the consensus template length rather than any single file.
    """
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if reference.size == 0 or reference.ndim != 2:
        return []

    frame_hop_seconds = _HOP_SIZE / _ANALYSIS_RATE
    sample_duration = reference.shape[0] * frame_hop_seconds

    full_profile = _get_source_profile(source_path, cache_dir=profile_cache_dir)
    if full_profile.size == 0 or full_profile.shape[0] < reference.shape[0]:
        return []

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

    hop = max(1, int(step_seconds / frame_hop_seconds))
    n_steps = max(0, (source_profile.shape[0] - reference.shape[0] + 1 + hop - 1) // hop)
    _LOG.info(
        "  [search] consensus  source=%s  sample=%.1fs  steps=%d",
        _fmt_duration(source_profile.shape[0] * frame_hop_seconds),
        sample_duration,
        n_steps,
    )
    matches = _build_match_candidates(
        reference=reference,
        source_profile=source_profile,
        sample_duration=sample_duration,
        frame_hop_seconds=frame_hop_seconds,
        hop=hop,
        score_threshold=score_threshold,
        frame_offset_seconds=frame_offset_seconds,
    )
    matches = _suppress_overlapping(matches, min_overlap=dedupe_overlap)
    _LOG.info("  [match]  consensus  %d candidate(s)", len(matches))
    return matches


def band_center_frequencies(
    sample_rate: int = _ANALYSIS_RATE,
    band_count: int = _BAND_COUNT,
) -> list[float]:
    """Return the geometric-mean center frequency (Hz) for each filterbank band."""
    edges = np.geomspace(20.0, sample_rate / 2, band_count + 1)
    return [float(np.sqrt(edges[i] * edges[i + 1])) for i in range(band_count)]


def compute_audio_file_profile(path: Path) -> np.ndarray:
    """Decode *path* and return its full spectral profile as a float32 array.

    Shape is ``(n_frames, 64)`` — 32 band energies followed by 32 deltas per frame.
    """
    samples = _decode_pcm_mono_16k(path)
    return np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)


__all__ = [
    "AudioMatch",
    "find_audio_sample_matches",
    "find_audio_sample_matches_from_profile",
    "build_consensus_profile",
    "anchor_to_onset",
    "cross_correlate_align",
    "band_center_frequencies",
    "compute_audio_file_profile",
    "_get_source_profile",
    "_suppress_overlapping",
]
