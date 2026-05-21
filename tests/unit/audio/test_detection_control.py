"""Control tests for the audio detector on multi-occurrence synthetic sources.

These tests verify the two core algorithmic properties used by the pipeline:

  Top-N detection: When a snippet appears multiple times in a source at scores
    that may be close to background noise, all confirmed occurrences must appear
    in the global top-N after NMS (mirrors the pipeline's max_matches behaviour).

  Threshold detection: When a snippet is clearly distinguishable from background,
    all occurrences must score above the configured threshold.

Each test embeds a multi-tone snippet at three known positions in a pseudo-random
noise source and asserts that all three are recovered.  Sources are short (~30 s)
so the suite runs in under a second per test.
"""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

from part_io.adapters.audio.matcher import find_audio_sample_matches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mono_wav(path: Path, samples: list[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(array("h", samples).tobytes())


def _make_noise(sample_count: int, seed: int) -> list[int]:
    return [(((i + seed) * 7919) % 4001) - 2000 for i in range(sample_count)]


def _make_chord(sample_rate: int, duration_seconds: float, freqs: tuple[float, ...]) -> list[int]:
    """Multi-frequency sine mixture — spectrally distinctive relative to noise."""
    total = int(sample_rate * duration_seconds)
    per_tone = 1.0 / len(freqs)
    return [
        int(sum(per_tone * math.sin(2 * math.pi * f * i / sample_rate) for f in freqs) * 12000)
        for i in range(total)
    ]


def _embed(base: list[int], snippet: list[int], offset: int) -> None:
    for j, v in enumerate(snippet):
        if offset + j < len(base):
            base[offset + j] = v


def _has_match_near(matches, target: float, tol: float = 1.0) -> bool:
    return any(abs(m.start_seconds - target) <= tol for m in matches)


def _top_n(matches, n: int):
    return sorted(matches, key=lambda m: m.score, reverse=True)[:n]


def _make_multi_occurrence_source(
    tmp_path: Path,
    *,
    snippet_freqs: tuple[float, ...],
    snippet_duration: float = 1.0,
    source_duration: float = 30.0,
    positions: tuple[float, ...] = (5.0, 15.0, 25.0),
    noise_seed: int = 42,
    sample_rate: int = 16000,
) -> tuple[Path, Path, list[float]]:
    """Write sample.wav and source.wav with snippet embedded at *positions*.

    Returns (sample_path, source_path, embed_starts).
    """
    snippet = _make_chord(sample_rate, snippet_duration, snippet_freqs)
    sample_path = tmp_path / "sample.wav"
    _write_mono_wav(sample_path, snippet, sample_rate)

    source = _make_noise(int(sample_rate * source_duration), seed=noise_seed)
    for pos in positions:
        _embed(source, snippet, offset=int(pos * sample_rate))
    source_path = tmp_path / "source.wav"
    _write_mono_wav(source_path, source, sample_rate)

    return sample_path, source_path, list(positions)


# ---------------------------------------------------------------------------
# Top-N detection — snippet must appear in global top-5 regardless of how
# close background scores are.  Uses two independent noise seeds (analogous
# to testing on two separate episodes).
# ---------------------------------------------------------------------------


def test_open_top5_contains_all_confirmed_occurrences_seed_a(tmp_path: Path) -> None:
    """All three embedded positions must appear in the global top-5 by score."""
    sample_path, source_path, positions = _make_multi_occurrence_source(
        tmp_path,
        snippet_freqs=(440.0, 660.0, 880.0),
        noise_seed=1001,
    )

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.0,
        step_seconds=0.1,
    )
    assert matches, "no matches returned — detector found nothing"
    top = _top_n(matches, 5)
    missing = [p for p in positions if not _has_match_near(top, p)]
    assert not missing, (
        f"position(s) {missing} s not in top-5.\n"
        f"Top-5: {[(round(m.start_seconds, 2), m.score) for m in top]}"
    )


def test_open_top5_contains_all_confirmed_occurrences_seed_b(tmp_path: Path) -> None:
    """All three embedded positions must appear in the global top-5 by score (different noise)."""
    sample_path, source_path, positions = _make_multi_occurrence_source(
        tmp_path,
        snippet_freqs=(550.0, 770.0, 990.0),
        noise_seed=2002,
    )

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.0,
        step_seconds=0.1,
    )
    assert matches, "no matches returned — detector found nothing"
    top = _top_n(matches, 5)
    missing = [p for p in positions if not _has_match_near(top, p)]
    assert not missing, (
        f"position(s) {missing} s not in top-5.\n"
        f"Top-5: {[(round(m.start_seconds, 2), m.score) for m in top]}"
    )


# ---------------------------------------------------------------------------
# Threshold detection — all occurrences must score above score_threshold=0.8.
# ---------------------------------------------------------------------------


def test_close_detected_above_threshold_seed_a(tmp_path: Path) -> None:
    """All three embedded positions must be found with the default threshold."""
    sample_path, source_path, positions = _make_multi_occurrence_source(
        tmp_path,
        snippet_freqs=(330.0, 495.0, 825.0),
        noise_seed=3003,
    )

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
    )
    assert matches, "no matches above threshold — detector returned nothing"
    missing = [p for p in positions if not _has_match_near(matches, p)]
    assert not missing, (
        f"position(s) {missing} s not found above threshold 0.8.\n"
        f"Found starts: {sorted(round(m.start_seconds, 2) for m in matches)}"
    )


def test_close_detected_above_threshold_seed_b(tmp_path: Path) -> None:
    """All three embedded positions must be found with the default threshold (different noise)."""
    sample_path, source_path, positions = _make_multi_occurrence_source(
        tmp_path,
        snippet_freqs=(220.0, 440.0, 1100.0),
        noise_seed=4004,
    )

    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=0.8,
        step_seconds=0.1,
    )
    assert matches, "no matches above threshold — detector returned nothing"
    missing = [p for p in positions if not _has_match_near(matches, p)]
    assert not missing, (
        f"position(s) {missing} s not found above threshold 0.8.\n"
        f"Found starts: {sorted(round(m.start_seconds, 2) for m in matches)}"
    )
