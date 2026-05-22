"""Round-trip tests for snippet_profile.

Coverage:
  raw PCM → spectral profile matrix
  profile matrix → SnippetProfileModel (zlib-compressed base64 blob)
  SnippetProfileModel → decoded matrix == original

A secondary test verifies that write_snippet_profile + is_profile_current
honour the source_hash staleness check.
"""

from __future__ import annotations

import base64
import math
import tomllib
from unittest.mock import patch

import numpy as np

from part_io.adapters.audio.matcher import (
    _ANALYSIS_RATE,
    _BAND_COUNT,
    _build_spectral_profile,
)
from part_io.adapters.audio.snippet_profile import (
    CompressedData,
    SnippetProfileModel,
    _build_profile_model,
    _ProfileData,
    decode_matrix,
    encode_matrix,
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
    n_frames, _ = matrix.shape
    return _ProfileData(
        source_hash="cafebabe",
        n_frames=n_frames,
        energy=matrix[:, :_BAND_COUNT],
        delta=matrix[:, _BAND_COUNT:],
    )


# ---------------------------------------------------------------------------
# Encode / decode round-trip
# ---------------------------------------------------------------------------


class TestEncodeDecodeMatrix:
    def test_encode_returns_compressed_data(self):
        matrix = np.random.default_rng(0).random((10, 64)).astype(np.float32)
        result = encode_matrix(matrix)
        assert isinstance(result, CompressedData)

    def test_encode_decode_identity(self):
        matrix = np.random.default_rng(0).random((10, 64)).astype(np.float32)
        restored = decode_matrix(encode_matrix(matrix), 10, 32)
        np.testing.assert_array_equal(restored, matrix)

    def test_header_and_signature_are_fixed_length(self):
        # zlib header is always 2 bytes (→ 4 b64 chars); signature is always 4 bytes (→ 8 b64 chars)
        cd = encode_matrix(np.zeros((5, 64), dtype=np.float32))
        assert len(base64.b64decode(cd.header)) == 2
        assert len(base64.b64decode(cd.signature)) == 4

    def test_body_is_base85_string(self):
        cd = encode_matrix(np.random.default_rng(2).random((73, 64)).astype(np.float32))
        assert isinstance(cd.body, str)
        assert len(cd.body) % 5 == 0, "base85 output length must be a multiple of 5"

    def test_compression_beats_raw_base64(self):
        """Compressed body + header + signature should be smaller than uncompressed base64."""
        matrix = np.random.default_rng(1).random((73, 64)).astype(np.float32)
        cd = encode_matrix(matrix)
        compressed_chars = len(cd.header) + len(cd.body) + len(cd.signature)
        raw_b64_chars = len(base64.b64encode(matrix.tobytes()))
        assert compressed_chars < raw_b64_chars


# ---------------------------------------------------------------------------
# Round-trip: raw PCM → profile matrix → model → decoded matrix
# ---------------------------------------------------------------------------


class TestProfileRoundTrip:
    def _build_profile(self, duration: float = 1.5) -> np.ndarray:
        samples = _make_sine_mix(duration, [440.0, 880.0, 1760.0, 220.0])
        profile = np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)
        assert profile.size > 0, "spectral profile must be non-empty"
        return profile

    def test_model_fields(self):
        profile = self._build_profile()
        model = _build_profile_model(_profile_data_from_matrix(profile))

        assert model.n_frames == profile.shape[0]
        assert model.band_count == _BAND_COUNT
        assert model.analysis_rate == _ANALYSIS_RATE
        assert model.hop_size == 1024  # _HOP_SIZE

    def test_decode_shape(self):
        profile = self._build_profile()
        model = _build_profile_model(_profile_data_from_matrix(profile))

        matrix = decode_matrix(model.data, model.n_frames, model.band_count)
        assert matrix.shape == (profile.shape[0], _BAND_COUNT * 2)

    def test_decode_matches_original(self):
        """Decoded matrix must be bit-for-bit identical to the input."""
        profile = self._build_profile()
        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        matrix = decode_matrix(model.data, model.n_frames, model.band_count)
        np.testing.assert_array_equal(matrix[:, :_BAND_COUNT], d.energy)
        np.testing.assert_array_equal(matrix[:, _BAND_COUNT:], d.delta)

    def test_toml_round_trip(self):
        """SnippetProfileModel survives model_dump → tomli_w → tomllib → model_validate."""
        import tomli_w

        profile = self._build_profile()
        d = _profile_data_from_matrix(profile)
        model = _build_profile_model(d)

        toml_text = tomli_w.dumps(model.model_dump())
        restored = SnippetProfileModel.model_validate(tomllib.loads(toml_text))

        assert restored.n_frames == model.n_frames
        assert restored.source_hash == model.source_hash

        matrix = decode_matrix(restored.data, restored.n_frames, restored.band_count)
        np.testing.assert_array_equal(matrix[:, :_BAND_COUNT], d.energy)
        np.testing.assert_array_equal(matrix[:, _BAND_COUNT:], d.delta)


# ---------------------------------------------------------------------------
# write_snippet_profile + is_profile_current
# ---------------------------------------------------------------------------


class TestWriteAndStaleness:
    def _fake_profile(self) -> np.ndarray:
        samples = _make_sine_mix(1.0, [440.0, 1320.0])
        return np.asarray(_build_spectral_profile(samples, _ANALYSIS_RATE), dtype=np.float32)

    def test_written_file_is_valid_toml(self, tmp_path):
        snippet = tmp_path / "test.mp3"
        snippet.write_bytes(b"\xff\xfb" * 1024)

        fake_profile = self._fake_profile()
        with patch(
            "part_io.adapters.audio.snippet_profile.compute_audio_file_profile",
            return_value=fake_profile,
        ):
            out = write_snippet_profile(snippet)

        assert out.exists()
        model = SnippetProfileModel.model_validate(tomllib.loads(out.read_text(encoding="utf-8")))
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

        snippet.write_bytes(b"\x00" * 2048)
        assert not is_profile_current(snippet)

    def test_is_profile_current_false_when_missing(self, tmp_path):
        snippet = tmp_path / "test.mp3"
        snippet.write_bytes(b"\xff\xfb" * 1024)
        assert not is_profile_current(snippet)
