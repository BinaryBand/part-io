"""Tests for audio sample matching."""

from __future__ import annotations

import itertools
import math
import wave
from array import array
from pathlib import Path

import pytest

from partio.adapters.audio.matcher import (
    AudioMatch,
    _band_energy_matrix,
    _finalize_profile,
    _suppress_overlapping,
    find_audio_sample_matches,
    find_best_sample_match,
)

ROOT = Path(__file__).resolve().parents[4]
REAL_SOURCE = ROOT / "downloads" / "media" / "dece9384-9892-4b4d-9c13-5298e44d67ab.mp3"
REAL_SAMPLE = ROOT / "downloads" / "snippets" / "close.mp3"


def _write_mono_wav(path: Path, samples: list[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        pcm = array("h", samples)
        wav_file.writeframes(pcm.tobytes())


def _make_sine_wave(sample_rate: int, duration_seconds: float, frequency_hz: float) -> list[int]:
    total_samples = int(sample_rate * duration_seconds)
    return [
        int(12000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate))
        for index in range(total_samples)
    ]


def _make_noise(sample_count: int, seed: int) -> list[int]:
    return [(((index + seed) * 7919) % 4001) - 2000 for index in range(sample_count)]


@pytest.mark.skipif(not REAL_SOURCE.exists(), reason="media not downloaded")
def test_find_audio_sample_matches_reports_expected_region() -> None:
    """The detector should find the known close/open sample region near 22:46."""
    source = REAL_SOURCE
    sample = REAL_SAMPLE

    matches = find_audio_sample_matches(source_path=source, sample_path=sample)

    assert matches
    assert any(1365 <= match.start_seconds <= 1367 for match in matches)


def test_spectral_profile_uses_thirty_two_band_with_deltas() -> None:
    """The spectral extractor should emit 64-element vectors (32 bands + 32 deltas) per frame."""
    sample_rate = 16000
    samples = [
        int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate)) for index in range(2048 * 3)
    ]

    band_matrix = _band_energy_matrix(samples, sample_rate)
    profile = _finalize_profile(band_matrix, band_matrix.mean(axis=0))

    assert profile.shape == (5, 64)


def test_synthetic_burst_is_detected_in_noise(tmp_path: Path) -> None:
    """A clear sine-wave burst should stand out from surrounding white noise."""
    sample_rate = 16000
    burst = _make_sine_wave(sample_rate, 1.0, 440.0)
    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"

    _write_mono_wav(sample_path, burst, sample_rate)

    prefix = _make_noise(sample_rate * 2, seed=1234)
    suffix = _make_noise(sample_rate * 2, seed=5678)
    source_samples = [*prefix, *burst, *suffix]
    _write_mono_wav(source_path, source_samples, sample_rate)

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
    )

    # Mean-centered scoring should isolate the burst with no false positives:
    # every reported match must sit on the burst, not the surrounding noise.
    assert matches
    assert all(1.9 <= match.start_seconds <= 2.2 for match in matches)


def test_find_best_sample_match_picks_prominent_peak(tmp_path: Path) -> None:
    """The peak-picker should land on the burst and report positive prominence."""
    sample_rate = 16000
    burst = _make_sine_wave(sample_rate, 1.0, 440.0)
    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"

    _write_mono_wav(sample_path, burst, sample_rate)
    prefix = _make_noise(sample_rate * 2, seed=1234)
    suffix = _make_noise(sample_rate * 2, seed=5678)
    _write_mono_wav(source_path, [*prefix, *burst, *suffix], sample_rate)

    best = find_best_sample_match(source_path=source_path, sample_path=sample_path)

    assert best is not None
    assert 1.9 <= best.start_seconds <= 2.2
    assert best.prominence > 0.0


def test_find_best_sample_match_honors_search_window(tmp_path: Path) -> None:
    """A search window shorter than the burst offset should exclude the burst."""
    sample_rate = 16000
    burst = _make_sine_wave(sample_rate, 1.0, 440.0)
    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"

    _write_mono_wav(sample_path, burst, sample_rate)
    prefix = _make_noise(sample_rate * 2, seed=1234)
    _write_mono_wav(source_path, [*prefix, *burst], sample_rate)

    best = find_best_sample_match(
        source_path=source_path, sample_path=sample_path, search_seconds=1.0
    )

    assert best is not None
    assert best.start_seconds < 1.0


@pytest.mark.skipif(not REAL_SOURCE.exists(), reason="media not downloaded")
def test_matches_are_sorted_and_non_overlapping_by_default() -> None:
    """Suppression should return deduplicated matches ordered by start time."""
    source = REAL_SOURCE
    sample = REAL_SAMPLE

    matches = find_audio_sample_matches(source_path=source, sample_path=sample)

    starts = [match.start_seconds for match in matches]
    assert starts == sorted(starts)
    for left, right in itertools.pairwise(matches):
        assert right.start_seconds >= left.end_seconds or left.end_seconds - right.start_seconds < (
            0.5 * min(left.duration_seconds, right.duration_seconds)
        )


def test_suppress_overlapping_keeps_best_scored_match() -> None:
    """NMS should retain the strongest match in an overlapping cluster."""
    matches = [
        AudioMatch(start_seconds=10.0, end_seconds=14.75, duration_seconds=4.75, score=0.82),
        AudioMatch(start_seconds=10.4, end_seconds=15.15, duration_seconds=4.75, score=0.97),
        AudioMatch(start_seconds=21.0, end_seconds=25.75, duration_seconds=4.75, score=0.88),
    ]

    deduped = _suppress_overlapping(matches, min_overlap=0.5)

    assert len(deduped) == 2
    assert deduped[0].start_seconds == 10.4
    assert deduped[1].start_seconds == 21.0
