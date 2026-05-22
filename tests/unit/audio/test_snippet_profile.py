"""Round-trip tests for snippet_profile.

Coverage:
  raw PCM → spectral profile matrix
  profile matrix → SnippetProfileModel (delta-encoded)
  SnippetProfileModel → reconstructed matrix ≈ original

A secondary test verifies that write_snippet_profile + is_profile_current
honour the source_hash staleness check.
"""

from __future__ import annotations

import math
import tomllib
from unittest.mock import patch

import numpy as np
import pytest

from part_io.adapters.audio.matcher import (
    _ANALYSIS_RATE,
    _BAND_COUNT,
    _build_spectral_profile,
)
from part_io.adapters.audio.snippet_profile import (
    SnippetProfileModel,
    _build_profile_model,
    _ProfileData,
    is_profile_current,
    write_snippet_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sine_mix(duration_seconds: float, frequencies: list[float]) -> list[int]:
    """Sum several sine waves into signed 16-bit PCM at the analysis rate."""
    n = int(_ANALYSIS_RATE * duration_seconds)
    amplitude = 16000 // len(frequencies)
    samples = [0] * n
    for freq in frequencies:
        for i in range(n):
            samples[i] += int(amplitude * math.sin(2 * math.pi * freq * i / _ANALYSIS_RATE))
    return samples


def _profile_data_from_matrix(matrix: np.ndarray) -> _ProfileData:
    n_frames, n_dims = matrix.shape
    return _ProfileData(
        source_hash="cafebabe",
        n_frames=n_frames,
        n_dims=n_dims,
        energy=matrix[:, :_BAND_COUNT],
        delta=matrix[:, _BAND_COUNT:],
        generated_at="2026-01-01T00:00:00Z",
    )


def _reconstruct_matrix(model: SnippetProfileModel) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct per-frame energy and delta arrays from a SnippetProfileModel."""
    n = model.n_frames
    energy = np.zeros((n, model.band_count), dtype=np.float64)
    delta = np.zeros((n, model.band_count), dtype=np.float64)

    energy[0] = model.keyframe_energy
    delta[0] = model.keyframe_delta

    for i, diff in enumerate(model.diffs):
        energy[i + 1] = energy[i] + np.array(diff.energy)
        delta[i + 1] = delta[i] + np.array(diff.delta)

    return energy, delta


# ---------------------------------------------------------------------------
# Round-trip: raw PCM → profile matrix → model → reconstructed matrix
# ---------------------------------------------------------------------------


class TestProfileRoundTrip:
    def _build_profile(self, duration: float = 1.5) -> np.ndarray:
        samples = _make_sine_mix(duration, [440.0, 880.0, 1760.0, 220.0])
        profile = np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)
        assert profile.size > 0, "spectral profile must be non-empty"
        return profile

    def test_model_frame_count(self):
        profile = self._build_profile()
        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        assert model.n_frames == profile.shape[0]
        assert len(model.diffs) == profile.shape[0] - 1

    def test_model_band_count(self):
        profile = self._build_profile()
        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        assert model.band_count == _BAND_COUNT
        assert len(model.keyframe_energy) == _BAND_COUNT
        assert len(model.keyframe_delta) == _BAND_COUNT
        for diff in model.diffs:
            assert len(diff.energy) == _BAND_COUNT
            assert len(diff.delta) == _BAND_COUNT

    def test_reconstruction_matches_original(self):
        """Cumulative sum of diffs must reproduce every original frame."""
        profile = self._build_profile()
        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        energy_rec, delta_rec = _reconstruct_matrix(model)

        # Tolerance: rounding to 5 decimal places, accumulated over n_frames steps.
        atol = model.n_frames * 1e-5

        np.testing.assert_allclose(energy_rec, d.energy, atol=atol, err_msg="energy mismatch")
        np.testing.assert_allclose(delta_rec, d.delta, atol=atol, err_msg="delta mismatch")

    def test_keyframe_matches_first_frame(self):
        profile = self._build_profile()
        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        np.testing.assert_allclose(
            model.keyframe_energy, d.energy[0], atol=1e-5, err_msg="keyframe energy"
        )
        np.testing.assert_allclose(
            model.keyframe_delta, d.delta[0], atol=1e-5, err_msg="keyframe delta"
        )

    def test_silent_frames_produce_zero_diffs(self):
        """Silent frames have identical spectral shapes → all diffs should be zero."""
        # Build a profile from constant (non-zero) samples so L2-norm is stable
        n_samples = _ANALYSIS_RATE * 2
        samples = [8000] * n_samples
        profile = np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)
        if profile.shape[0] < 2:
            pytest.skip("profile too short for this assertion")

        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        for diff in model.diffs:
            np.testing.assert_allclose(
                diff.energy, [0.0] * _BAND_COUNT, atol=1e-5, err_msg="expected zero energy diffs"
            )
            np.testing.assert_allclose(
                diff.delta, [0.0] * _BAND_COUNT, atol=1e-5, err_msg="expected zero delta diffs"
            )

    def test_toml_round_trip(self, tmp_path):
        """SnippetProfileModel survives a model_dump → tomli_w → tomllib parse cycle."""
        import tomli_w

        profile = self._build_profile()
        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        toml_bytes = tomli_w.dumps(model.model_dump()).encode("utf-8")
        parsed = tomllib.loads(toml_bytes.decode("utf-8"))
        restored = SnippetProfileModel.model_validate(parsed)

        assert restored.n_frames == model.n_frames
        assert restored.source_hash == model.source_hash
        assert len(restored.diffs) == len(model.diffs)

        energy_rec, delta_rec = _reconstruct_matrix(restored)
        atol = model.n_frames * 1e-5
        np.testing.assert_allclose(energy_rec, d.energy, atol=atol)
        np.testing.assert_allclose(delta_rec, d.delta, atol=atol)


# ---------------------------------------------------------------------------
# write_snippet_profile + is_profile_current
# ---------------------------------------------------------------------------


class TestWriteAndStaleness:
    def _fake_profile(self) -> np.ndarray:
        samples = _make_sine_mix(1.0, [440.0, 1320.0])
        return np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)

    def test_written_file_is_valid_toml(self, tmp_path):
        snippet = tmp_path / "test.mp3"
        snippet.write_bytes(b"\xff\xfb" * 1024)  # dummy file for hash

        fake_profile = self._fake_profile()
        with patch(
            "part_io.adapters.audio.snippet_profile.compute_audio_file_profile",
            return_value=fake_profile,
        ):
            out = write_snippet_profile(snippet)

        assert out.exists()
        parsed = tomllib.loads(out.read_text(encoding="utf-8"))
        model = SnippetProfileModel.model_validate(parsed)
        assert model.n_frames == fake_profile.shape[0]

    def test_is_profile_current_true_after_write(self, tmp_path):
        snippet = tmp_path / "test.mp3"
        snippet.write_bytes(b"\xff\xfb" * 1024)

        with patch(
            "part_io.adapters.audio.snippet_profile.compute_audio_file_profile",
            return_value=self._fake_profile(),
        ):
            write_snippet_profile(snippet)

        assert is_profile_current(snippet)

    def test_is_profile_current_false_after_file_change(self, tmp_path):
        snippet = tmp_path / "test.mp3"
        snippet.write_bytes(b"\xff\xfb" * 1024)

        with patch(
            "part_io.adapters.audio.snippet_profile.compute_audio_file_profile",
            return_value=self._fake_profile(),
        ):
            write_snippet_profile(snippet)

        snippet.write_bytes(b"\x00" * 2048)  # replace content → hash changes
        assert not is_profile_current(snippet)

    def test_is_profile_current_false_when_missing(self, tmp_path):
        snippet = tmp_path / "test.mp3"
        snippet.write_bytes(b"\xff\xfb" * 1024)
        assert not is_profile_current(snippet)
