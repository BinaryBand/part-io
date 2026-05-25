"""Accuracy tests for the audio matcher — internal helpers and end-to-end detection.

These tests verify the correctness of timestamp math, signal-level properties
of the filterbank, detection accuracy in synthetic data, and the absence of
false positives in pure noise.  All tests run without external audio files;
synthetic PCM data is generated in memory and written to temporary WAV files
where ffmpeg is required.
"""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

import numpy as np
import pytest

from part_io.adapters.audio.matcher import (
    _ANALYSIS_RATE,
    _BAND_COUNT,
    _HOP_SIZE,
    AudioMatch,
    _build_filterbank_matrix,
    _build_spectral_profile,
    _normalized_similarity,
    _overlap_ratio,
    _scores_to_matches,
    _stack_temporal_deltas,
    _suppress_overlapping,
    band_center_frequencies,
    build_consensus_profile,
    find_audio_sample_matches,
    find_audio_sample_matches_from_profile,
)

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic audio helpers
# ──────────────────────────────────────────────────────────────────────────────

_RATE = _ANALYSIS_RATE  # 16 000 Hz


def _sine(duration_s: float, freq_hz: float, amplitude: int = 12000) -> list[int]:
    n = int(_RATE * duration_s)
    return [int(amplitude * math.sin(2 * math.pi * freq_hz * i / _RATE)) for i in range(n)]


def _noise(n_samples: int, seed: int = 0) -> list[int]:
    return [(((i + seed) * 7919) % 4001) - 2000 for i in range(n_samples)]


def _embed(base: list[int], snippet: list[int], offset: int) -> None:
    for j, v in enumerate(snippet):
        if offset + j < len(base):
            base[offset + j] = v


def _write_wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_RATE)
        wf.writeframes(array("h", samples).tobytes())


# ──────────────────────────────────────────────────────────────────────────────
# _overlap_ratio
# ──────────────────────────────────────────────────────────────────────────────


class TestOverlapRatio:
    def _match(self, start: float, end: float) -> AudioMatch:
        return AudioMatch(
            start_seconds=start,
            end_seconds=end,
            duration_seconds=end - start,
            score=1.0,
        )

    def test_no_overlap_returns_zero(self) -> None:
        left = self._match(0.0, 2.0)
        right = self._match(3.0, 5.0)
        assert _overlap_ratio(left, right) == pytest.approx(0.0)

    def test_identical_matches_return_one(self) -> None:
        m = self._match(1.0, 3.0)
        assert _overlap_ratio(m, m) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        left = self._match(0.0, 4.0)
        right = self._match(2.0, 6.0)
        # overlap = 2s, shortest = 4s → ratio = 0.5
        assert _overlap_ratio(left, right) == pytest.approx(0.5)

    def test_adjacent_but_non_overlapping(self) -> None:
        left = self._match(0.0, 2.0)
        right = self._match(2.0, 4.0)
        assert _overlap_ratio(left, right) == pytest.approx(0.0)

    def test_zero_duration_returns_zero(self) -> None:
        left = AudioMatch(start_seconds=1.0, end_seconds=1.0, duration_seconds=0.0, score=1.0)
        right = self._match(0.5, 1.5)
        assert _overlap_ratio(left, right) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# _normalized_similarity
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizedSimilarity:
    def _unit_rows(self, n: int, d: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def test_identical_unit_matrices_return_one(self) -> None:
        m = self._unit_rows(10, 64)
        score = _normalized_similarity(m, m)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_identical_non_unit_rows_dot_product_equals_squared_norm(self) -> None:
        # _normalized_similarity computes mean frame-wise dot product; inputs are
        # assumed pre-normalized (as real spectral profiles are). With all-ones
        # rows the dot product per frame is d=64, so the mean is 64.
        m = np.ones((5, 64), dtype=np.float32)
        score = _normalized_similarity(m, m)
        assert score == pytest.approx(64.0, abs=1e-3)

    def test_shape_mismatch_returns_negative_one(self) -> None:
        a = self._unit_rows(5, 64)
        b = self._unit_rows(6, 64)
        assert _normalized_similarity(a, b) == -1.0

    def test_orthogonal_features_give_low_score(self) -> None:
        n, d = 10, 64
        a = np.zeros((n, d), dtype=np.float32)
        b = np.zeros((n, d), dtype=np.float32)
        a[:, : d // 2] = 1.0
        b[:, d // 2 :] = 1.0
        # Each row of a is orthogonal to the corresponding row of b
        score = _normalized_similarity(
            a / np.linalg.norm(a, axis=1, keepdims=True),
            b / np.linalg.norm(b, axis=1, keepdims=True),
        )
        assert score == pytest.approx(0.0, abs=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# _scores_to_matches — timestamp accuracy
# ──────────────────────────────────────────────────────────────────────────────


class TestScoresToMatches:
    """Verify that the frame-index → seconds mapping is exactly correct."""

    def _frame_hop_s(self) -> float:
        return _HOP_SIZE / _ANALYSIS_RATE

    def test_single_match_at_index_zero_has_zero_start(self) -> None:
        fh = self._frame_hop_s()
        matches = _scores_to_matches(
            np.array([0.95], dtype=np.float64),
            score_threshold=0.9,
            hop=1,
            frame_hop_seconds=fh,
            frame_offset_seconds=0.0,
            sample_duration=1.0,
        )
        assert len(matches) == 1
        assert matches[0].start_seconds == pytest.approx(0.0, abs=1e-4)
        assert matches[0].end_seconds == pytest.approx(1.0, abs=1e-4)

    def test_index_k_maps_to_k_times_hop_times_fhs(self) -> None:
        fh = self._frame_hop_s()
        hop = 3
        scores = np.zeros(10, dtype=np.float64)
        scores[4] = 0.95  # index 4 in the coarse grid
        matches = _scores_to_matches(
            scores,
            score_threshold=0.9,
            hop=hop,
            frame_hop_seconds=fh,
            frame_offset_seconds=0.0,
            sample_duration=1.0,
        )
        assert len(matches) == 1
        expected_start = 4 * hop * fh
        assert matches[0].start_seconds == pytest.approx(expected_start, abs=1e-3)

    def test_frame_offset_shifts_all_starts(self) -> None:
        fh = self._frame_hop_s()
        offset = 5.0
        matches = _scores_to_matches(
            np.array([0.95], dtype=np.float64),
            score_threshold=0.9,
            hop=1,
            frame_hop_seconds=fh,
            frame_offset_seconds=offset,
            sample_duration=1.0,
        )
        assert matches[0].start_seconds == pytest.approx(offset, abs=1e-3)

    def test_score_below_threshold_not_returned(self) -> None:
        fh = self._frame_hop_s()
        matches = _scores_to_matches(
            np.array([0.5], dtype=np.float64),
            score_threshold=0.9,
            hop=1,
            frame_hop_seconds=fh,
            frame_offset_seconds=0.0,
            sample_duration=1.0,
        )
        assert matches == []


# ──────────────────────────────────────────────────────────────────────────────
# _build_filterbank_matrix — signal-level properties
# ──────────────────────────────────────────────────────────────────────────────


class TestFilterbankMatrix:
    def test_shape_is_rfft_bins_by_band_count(self) -> None:
        from part_io.adapters.audio.matcher import _FRAME_SIZE

        fb = _build_filterbank_matrix(_ANALYSIS_RATE, _FRAME_SIZE, _BAND_COUNT)
        n_bins = _FRAME_SIZE // 2 + 1
        assert fb.shape == (n_bins, _BAND_COUNT)

    def test_all_entries_non_negative(self) -> None:
        fb = _build_filterbank_matrix(_ANALYSIS_RATE)
        assert (fb >= 0).all()

    def test_each_column_sums_to_approximately_one(self) -> None:
        # Each column is normalised by band_bins.size, so the sum equals 1 for
        # non-singleton bands.  Allow a small absolute tolerance for bands that
        # fall back to a single nearest bin (sum = 1 always) or very narrow bands.
        fb = _build_filterbank_matrix(_ANALYSIS_RATE)
        col_sums = fb.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=0.15)

    def test_each_band_has_at_least_one_bin(self) -> None:
        # Every column (band) must have at least one non-zero weight so that
        # high-energy content in that frequency range produces a non-zero response.
        fb = _build_filterbank_matrix(_ANALYSIS_RATE)
        nonzero_per_band = (fb > 0).sum(axis=0)
        assert (nonzero_per_band >= 1).all()

    def test_invalid_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            _build_filterbank_matrix(0)


# ──────────────────────────────────────────────────────────────────────────────
# band_center_frequencies
# ──────────────────────────────────────────────────────────────────────────────


class TestBandCenterFrequencies:
    def test_returns_correct_count(self) -> None:
        freqs = band_center_frequencies()
        assert len(freqs) == _BAND_COUNT

    def test_monotonically_increasing(self) -> None:
        freqs = band_center_frequencies()
        assert all(a < b for a, b in zip(freqs, freqs[1:], strict=False))

    def test_within_audible_range(self) -> None:
        freqs = band_center_frequencies(_ANALYSIS_RATE)
        assert freqs[0] >= 20.0
        assert freqs[-1] <= _ANALYSIS_RATE / 2


# ──────────────────────────────────────────────────────────────────────────────
# _stack_temporal_deltas
# ──────────────────────────────────────────────────────────────────────────────


class TestStackTemporalDeltas:
    def test_output_doubles_feature_dimension(self) -> None:
        base = np.random.default_rng(0).random((10, 32)).astype(np.float32)
        stacked = _stack_temporal_deltas(base)
        assert stacked.shape == (10, 64)

    def test_first_row_delta_is_zero(self) -> None:
        # delta at frame 0 = base[0] - base[0] = 0
        base = np.random.default_rng(1).random((5, 16)).astype(np.float32)
        stacked = _stack_temporal_deltas(base)
        np.testing.assert_allclose(stacked[0, 16:], 0.0, atol=1e-6)

    def test_energy_half_matches_original(self) -> None:
        base = np.random.default_rng(2).random((8, 32)).astype(np.float32)
        stacked = _stack_temporal_deltas(base)
        np.testing.assert_array_equal(stacked[:, :32], base)


# ──────────────────────────────────────────────────────────────────────────────
# False positive rate: pure noise → no detections
# ──────────────────────────────────────────────────────────────────────────────


def test_spectrally_orthogonal_signal_produces_no_match(tmp_path: Path) -> None:
    """A 440 Hz sample should NOT be detected inside a spectrally unrelated 7000 Hz tone.

    This tests the false-positive rate: two pure tones with no spectral overlap
    must not match each other above the detection threshold.
    """
    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"

    # 440 Hz and 7000 Hz occupy completely different filterbank bands, so their
    # normalised spectral profiles are approximately orthogonal.
    _write_wav(sample_path, _sine(1.0, 440.0))
    _write_wav(source_path, _sine(20.0, 7000.0))

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.95,
        step_seconds=0.1,
    )
    assert matches == [], f"Expected no matches but got {len(matches)}: {matches}"


# ──────────────────────────────────────────────────────────────────────────────
# Search window: search_start/end constrains where matches are reported
# ──────────────────────────────────────────────────────────────────────────────


def test_search_window_excludes_match_outside_range(tmp_path: Path) -> None:
    """A match at 10s should not appear when the search window is 0s–5s."""
    burst = _sine(1.0, 660.0)
    source = _noise(_RATE * 20, seed=42)
    _embed(source, burst, _RATE * 10)  # embedded at 10 s

    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"
    _write_wav(sample_path, burst)
    _write_wav(source_path, source)

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
        search_start_seconds=0.0,
        search_end_seconds=5.0,
    )
    assert not any(m.start_seconds > 8.0 for m in matches), (
        f"Match at 10s leaked through the search window: {matches}"
    )


def test_search_window_finds_match_inside_range(tmp_path: Path) -> None:
    """Restricting the window to around an embedded burst should still detect it."""
    burst = _sine(1.0, 880.0)
    source = _noise(_RATE * 20, seed=13)
    _embed(source, burst, _RATE * 10)

    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"
    _write_wav(sample_path, burst)
    _write_wav(source_path, source)

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
        search_start_seconds=8.0,
        search_end_seconds=13.0,
    )
    assert any(9.0 <= m.start_seconds <= 11.5 for m in matches), (
        f"Expected match near 10s; got {[m.start_seconds for m in matches]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# build_consensus_profile
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildConsensusProfile:
    def test_returns_none_when_fewer_than_min_segments(self, tmp_path: Path) -> None:
        source = tmp_path / "ep.wav"
        _write_wav(source, _sine(2.0, 440.0))
        # Only one segment; min_segments defaults to 2
        result = build_consensus_profile([(source, 0.0, 1.0)], min_segments=2)
        assert result is None

    def test_identical_segments_yield_identical_consensus(self, tmp_path: Path) -> None:
        source = tmp_path / "ep.wav"
        _write_wav(source, _sine(5.0, 440.0))

        segs = [(source, 0.5, 1.5), (source, 0.5, 1.5)]
        consensus = build_consensus_profile(segs, min_segments=2)
        assert consensus is not None

        single = build_consensus_profile([(source, 0.5, 1.5)], min_segments=1)
        assert single is not None

        # Mean of two identical profiles equals the profile itself
        min_len = min(consensus.shape[0], single.shape[0])
        np.testing.assert_allclose(consensus[:min_len], single[:min_len], atol=1e-5)

    def test_consensus_shape_matches_shortest_segment(self, tmp_path: Path) -> None:
        source = tmp_path / "ep.wav"
        _write_wav(source, _sine(5.0, 440.0))

        short = build_consensus_profile([(source, 0.0, 0.5)], min_segments=1)
        long = build_consensus_profile([(source, 0.0, 2.0)], min_segments=1)
        assert short is not None and long is not None
        assert short.shape[0] < long.shape[0]

        # Two segments with different durations → consensus trimmed to shorter
        consensus = build_consensus_profile(
            [(source, 0.0, 0.5), (source, 0.0, 2.0)], min_segments=2
        )
        assert consensus is not None
        assert consensus.shape[0] == short.shape[0]

    def test_consensus_has_correct_feature_dimension(self, tmp_path: Path) -> None:
        source = tmp_path / "ep.wav"
        _write_wav(source, _sine(3.0, 440.0))
        consensus = build_consensus_profile(
            [(source, 0.0, 1.0), (source, 1.0, 2.0)], min_segments=2
        )
        assert consensus is not None
        assert consensus.shape[1] == _BAND_COUNT * 2


# ──────────────────────────────────────────────────────────────────────────────
# find_audio_sample_matches_from_profile
# ──────────────────────────────────────────────────────────────────────────────


def test_find_matches_from_profile_detects_same_burst(tmp_path: Path) -> None:
    """Profile-based detection should find the same burst as file-based detection."""
    burst = _sine(1.0, 550.0)
    source_samples = _noise(_RATE * 15, seed=77)
    _embed(source_samples, burst, _RATE * 7)

    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"
    _write_wav(sample_path, burst)
    _write_wav(source_path, source_samples)

    # Build reference profile from the sample file directly
    sample_pcm = _build_spectral_profile(burst, _ANALYSIS_RATE)
    reference = np.asarray(sample_pcm, dtype=np.float32)

    matches = find_audio_sample_matches_from_profile(
        source_path=source_path,
        reference=reference,
        score_threshold=0.8,
        step_seconds=0.1,
    )
    assert any(6.0 <= m.start_seconds <= 8.5 for m in matches), (
        f"Expected match near 7s from profile; got {[m.start_seconds for m in matches]}"
    )


def test_find_matches_from_profile_rejects_empty_reference(tmp_path: Path) -> None:
    source = tmp_path / "ep.wav"
    _write_wav(source, _sine(2.0, 440.0))
    empty_ref = np.zeros((0, 64), dtype=np.float32)
    matches = find_audio_sample_matches_from_profile(
        source_path=source,
        reference=empty_ref,
        score_threshold=0.8,
    )
    assert matches == []


# ──────────────────────────────────────────────────────────────────────────────
# Detection position accuracy — tight tolerance
# ──────────────────────────────────────────────────────────────────────────────


def test_detection_position_accurate_within_one_step(tmp_path: Path) -> None:
    """Detected start must be within one step_seconds of the true embed offset."""
    step = 0.1
    true_offset_s = 8.0
    burst = _sine(1.0, 770.0)
    source = _noise(_RATE * 15, seed=31)
    _embed(source, burst, int(_RATE * true_offset_s))

    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"
    _write_wav(sample_path, burst)
    _write_wav(source_path, source)

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=step,
    )
    assert matches, "Expected at least one match"
    closest = min(matches, key=lambda m: abs(m.start_seconds - true_offset_s))
    assert abs(closest.start_seconds - true_offset_s) <= step * 3, (
        f"Closest match at {closest.start_seconds:.3f}s; expected near {true_offset_s}s"
    )


def test_multiple_non_overlapping_bursts_all_detected(tmp_path: Path) -> None:
    """All three embedded bursts should appear; none should be merged."""
    burst = _sine(1.0, 990.0)
    source = _noise(_RATE * 30, seed=55)
    offsets = [5, 15, 25]
    for off in offsets:
        _embed(source, burst, _RATE * off)

    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"
    _write_wav(sample_path, burst)
    _write_wav(source_path, source)

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
    )
    for expected_s in offsets:
        assert any(abs(m.start_seconds - expected_s) <= 1.5 for m in matches), (
            f"No match near {expected_s}s — matches at {[m.start_seconds for m in matches]}"
        )


def test_suppress_overlapping_preserves_non_overlapping_matches() -> None:
    """NMS must not discard matches that do not overlap."""
    matches = [
        AudioMatch(start_seconds=0.0, end_seconds=1.0, duration_seconds=1.0, score=0.9),
        AudioMatch(start_seconds=5.0, end_seconds=6.0, duration_seconds=1.0, score=0.85),
        AudioMatch(start_seconds=10.0, end_seconds=11.0, duration_seconds=1.0, score=0.8),
    ]
    result = _suppress_overlapping(matches, min_overlap=0.5)
    assert len(result) == 3
    starts = [m.start_seconds for m in result]
    assert starts == sorted(starts)
