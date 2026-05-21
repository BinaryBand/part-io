"""Tests for audio sample matching."""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

from part_io.adapters.audio.matcher import (
    AudioMatch,
    _build_spectral_profile,
    _suppress_overlapping,
    find_audio_sample_matches,
)


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


def _embed(base: list[int], snippet: list[int], offset: int) -> None:
    for j, v in enumerate(snippet):
        if offset + j < len(base):
            base[offset + j] = v


def test_spectral_profile_uses_thirty_two_band_with_deltas() -> None:
    """The spectral extractor should emit 64-element vectors (32 bands + 32 deltas) per frame."""
    sample_rate = 16000
    samples = [
        int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(0, 2048 * 3)
    ]

    profile = _build_spectral_profile(samples, sample_rate)

    assert len(profile) == 5
    assert all(len(vector) == 64 for vector in profile)


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

    assert matches
    assert any(1.9 <= match.start_seconds <= 2.2 for match in matches)


def test_find_audio_sample_matches_reports_expected_region(tmp_path: Path) -> None:
    """Detector localises the match to within ±0.5s of the embedded position."""
    sample_rate = 16000
    burst = _make_sine_wave(sample_rate, 1.0, 880.0)
    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"
    _write_mono_wav(sample_path, burst, sample_rate)

    source = _make_noise(sample_rate * 15, seed=3333)
    _embed(source, burst, offset=sample_rate * 10)  # burst at exactly 10 s
    _write_mono_wav(source_path, source, sample_rate)

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
    )

    assert matches
    assert any(9.5 <= m.start_seconds <= 10.5 for m in matches)


def test_matches_are_sorted_and_non_overlapping_by_default(tmp_path: Path) -> None:
    """Suppression should return deduplicated matches ordered by start time."""
    sample_rate = 16000
    burst = _make_sine_wave(sample_rate, 1.0, 660.0)
    sample_path = tmp_path / "sample.wav"
    source_path = tmp_path / "source.wav"
    _write_mono_wav(sample_path, burst, sample_rate)

    source = _make_noise(sample_rate * 30, seed=7777)
    for offset_s in [5, 15, 25]:
        _embed(source, burst, offset=sample_rate * offset_s)
    _write_mono_wav(source_path, source, sample_rate)

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
    )

    starts = [match.start_seconds for match in matches]
    assert starts == sorted(starts)
    for left, right in zip(matches, matches[1:], strict=False):
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
