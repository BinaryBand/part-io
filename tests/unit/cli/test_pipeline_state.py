"""Unit tests for PipelineState / EpisodeState save-load round-trip and classification logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from part_io.adapters.audio.snippet_profile import decode_matrix, encode_matrix
from part_io.cli.remote._state import (
    EpisodeState,
    PipelineState,
    RunSettings,
    Segment,
    SnippetEntry,
    TargetState,
    _Match,
    _migrate_episode_keys,
    _seg_stem,
)

# ──────────────────────────────────────────────────────────────────────────────
# EpisodeState.class_for — derived classification from candidate labels
# ──────────────────────────────────────────────────────────────────────────────


class TestEpisodeClassFor:
    def test_empty_candidates_is_undetected(self) -> None:
        ep = EpisodeState()
        assert ep.class_for("open") == "undetected"

    def test_unlabeled_candidates_is_uncertain(self) -> None:
        ep = EpisodeState()
        ep.candidates["open"] = [_Match(score=0.8, start=1.0, end=2.0)]
        assert ep.class_for("open") == "uncertain"

    def test_any_positive_label_wins(self) -> None:
        ep = EpisodeState()
        ep.candidates["close"] = [
            _Match(score=0.9, start=1.0, end=2.0, label="positive"),
            _Match(score=0.5, start=5.0, end=6.0, label="negative"),
        ]
        assert ep.class_for("close") == "positive"

    def test_all_negative_labels_is_negative(self) -> None:
        ep = EpisodeState()
        ep.candidates["intro"] = [
            _Match(score=0.7, start=1.0, end=2.0, label="negative"),
            _Match(score=0.6, start=3.0, end=4.0, label="negative"),
        ]
        assert ep.class_for("intro") == "negative"

    def test_invalid_label_treated_as_unlabeled(self) -> None:
        ep = EpisodeState()
        ep.candidates["outro"] = [_Match(score=0.5, start=1.0, end=2.0, label="invalid")]
        # invalid label → treated as None → uncertain (not all-negative, not any-positive)
        assert ep.class_for("outro") == "uncertain"

    def test_unknown_kind_returns_undetected(self) -> None:
        ep = EpisodeState()
        assert ep.class_for("sponsor") == "undetected"


# ──────────────────────────────────────────────────────────────────────────────
# EpisodeState derived predicates
# ──────────────────────────────────────────────────────────────────────────────


class TestEpisodePredicates:
    def test_is_detected_false_when_all_undetected(self) -> None:
        assert not EpisodeState().is_detected()

    def test_is_detected_true_when_any_kind_has_candidates(self) -> None:
        ep = EpisodeState()
        ep.candidates["intro"] = [_Match(score=0.9, start=1.0, end=2.0)]
        assert ep.is_detected()

    def test_is_cuttable_false_when_open_missing(self) -> None:
        ep = EpisodeState()
        ep.candidates["close"] = [_Match(score=0.9, start=5.0, end=6.0, label="positive")]
        assert not ep.is_cuttable()

    def test_is_cuttable_false_when_close_missing(self) -> None:
        ep = EpisodeState()
        ep.candidates["open"] = [_Match(score=0.9, start=1.0, end=2.0, label="positive")]
        assert not ep.is_cuttable()

    def test_is_cuttable_true_when_both_positive(self) -> None:
        ep = EpisodeState()
        ep.candidates["open"] = [_Match(score=0.9, start=1.0, end=2.0, label="positive")]
        ep.candidates["close"] = [_Match(score=0.9, start=5.0, end=6.0, label="positive")]
        assert ep.is_cuttable()

    def test_is_cuttable_false_when_either_not_positive(self) -> None:
        ep = EpisodeState()
        ep.candidates["open"] = [_Match(score=0.9, start=1.0, end=2.0, label="positive")]
        ep.candidates["close"] = [_Match(score=0.4, start=5.0, end=6.0, label="negative")]
        assert not ep.is_cuttable()

    def test_first_positive_candidate_for_returns_first_match(self) -> None:
        ep = EpisodeState()
        ep.candidates["open"] = [
            _Match(score=0.4, start=1.0, end=2.0, label="negative"),
            _Match(score=0.9, start=3.0, end=4.0, label="positive"),
            _Match(score=0.8, start=5.0, end=6.0, label="positive"),
        ]
        first = ep.first_positive_candidate_for("open")
        assert first is not None
        assert first.start == 3.0

    def test_first_positive_candidate_for_none_when_absent(self) -> None:
        ep = EpisodeState()
        ep.candidates["close"] = [_Match(score=0.5, start=1.0, end=2.0, label="negative")]
        assert ep.first_positive_candidate_for("close") is None

    def test_source_hash_valid_matches_actual_file(self, tmp_path: Path) -> None:
        from part_io.utils.hash import partial_file_hash

        source = tmp_path / "ep.mp3"
        source.write_bytes(b"x" * 65536)
        ep = EpisodeState(source_hash=partial_file_hash(source))
        assert ep.source_hash_valid(source)

    def test_source_hash_valid_false_after_content_change(self, tmp_path: Path) -> None:
        from part_io.utils.hash import partial_file_hash

        source = tmp_path / "ep.mp3"
        source.write_bytes(b"x" * 65536)
        ep = EpisodeState(source_hash=partial_file_hash(source))
        source.write_bytes(b"y" * 65536)
        assert not ep.source_hash_valid(source)

    def test_source_hash_valid_false_when_none(self, tmp_path: Path) -> None:
        source = tmp_path / "ep.mp3"
        source.write_bytes(b"x" * 65536)
        ep = EpisodeState(source_hash=None)
        assert not ep.source_hash_valid(source)


# ──────────────────────────────────────────────────────────────────────────────
# Property accessors forward to the candidates dict
# ──────────────────────────────────────────────────────────────────────────────


class TestEpisodePropertyAccessors:
    def _ep_with_open(self) -> EpisodeState:
        ep = EpisodeState()
        ep.candidates["open"] = [_Match(score=0.9, start=1.0, end=2.0)]
        return ep

    def test_open_candidates_read(self) -> None:
        ep = self._ep_with_open()
        assert len(ep.open_candidates) == 1
        assert ep.open_candidates[0].score == 0.9

    def test_open_candidates_write(self) -> None:
        ep = EpisodeState()
        ep.open_candidates = [_Match(score=0.7, start=5.0, end=6.0)]
        assert ep.candidates["open"][0].score == 0.7

    def test_intro_candidates_roundtrip(self) -> None:
        ep = EpisodeState()
        ep.intro_candidates = [_Match(score=0.5, start=2.0, end=3.0)]
        assert ep.intro_candidates[0].start == 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Profile serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestProfileSerialisation:
    def _random_profile(self, rng_seed: int = 0) -> np.ndarray:
        return np.random.default_rng(rng_seed).random((30, 64)).astype(np.float32)

    def test_encode_decode_roundtrip_preserves_array(self) -> None:
        profile = self._random_profile()
        n_frames, band_count_x2 = profile.shape
        band_count = band_count_x2 // 2
        restored = decode_matrix(encode_matrix(profile), n_frames, band_count)
        np.testing.assert_array_almost_equal(restored, profile, decimal=5)

    def test_encoded_output_is_ascii_string(self) -> None:
        profile = self._random_profile()
        encoded = encode_matrix(profile)
        assert isinstance(encoded, str)
        encoded.encode("ascii")  # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# _seg_stem migration
# ──────────────────────────────────────────────────────────────────────────────


class TestSegStemMigration:
    def test_reads_stem_field_directly(self) -> None:
        assert _seg_stem({"stem": "ep001"}) == "ep001"

    def test_falls_back_to_source_path_stem(self) -> None:
        assert _seg_stem({"source": "/downloads/remote/ep001.mp3"}) == "ep001"

    def test_missing_both_returns_empty(self) -> None:
        assert _seg_stem({}) == ""


# ──────────────────────────────────────────────────────────────────────────────
# _migrate_episode_keys
# ──────────────────────────────────────────────────────────────────────────────


class TestMigrateEpisodeKeys:
    def test_unquoted_keys_get_quoted(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.toml"
        state_path.write_text(
            "[settings]\nworkers = 2\n\n[episodes.ep_abc123]\ncut = false\n",
            encoding="utf-8",
        )
        _migrate_episode_keys(state_path)
        text = state_path.read_text(encoding="utf-8")
        assert '[episodes."ep_abc123"]' in text

    def test_already_quoted_keys_untouched(self, tmp_path: Path) -> None:
        original = '[settings]\nworkers = 2\n\n[episodes."ep_abc123"]\ncut = false\n'
        state_path = tmp_path / "state.toml"
        state_path.write_text(original, encoding="utf-8")
        _migrate_episode_keys(state_path)
        assert state_path.read_text(encoding="utf-8") == original

    def test_no_episodes_section_is_noop(self, tmp_path: Path) -> None:
        original = "[settings]\nworkers = 2\n"
        state_path = tmp_path / "state.toml"
        state_path.write_text(original, encoding="utf-8")
        _migrate_episode_keys(state_path)
        assert state_path.read_text(encoding="utf-8") == original


# ──────────────────────────────────────────────────────────────────────────────
# PipelineState save → load round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestPipelineStateRoundTrip:
    def _make_state(self) -> PipelineState:
        state = PipelineState()
        state.settings = RunSettings(
            step_seconds=0.05,
            workers=4,
            max_matches=5,
            min_gap=-10.0,
            max_gap=250.0,
            fade=0.3,
            quiz_size=8,
        )
        state.open_target = TargetState(
            positives=[Segment(stem="ep1", start=10.0, end=20.0, score=0.95)],
            negatives=[Segment(stem="ep2", start=5.0, end=8.0, score=0.30)],
        )
        state.close_target = TargetState(
            positives=[Segment(stem="ep3", start=50.0, end=60.0, score=0.88)],
            negatives=[],
        )
        ep = EpisodeState(source_hash="deadbeef", cut=True)
        ep.candidates["open"] = [_Match(score=0.9, start=1.0, end=2.0, label="positive")]
        ep.candidates["close"] = [
            _Match(score=0.7, start=5.0, end=6.0, label="negative"),
            _Match(score=0.6, start=7.0, end=8.0),
        ]
        ep.candidates["intro"] = [_Match(score=0.85, start=30.0, end=31.0)]
        state.episodes["ep001"] = ep
        return state

    def test_empty_state_roundtrips(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = PipelineState()
        state.save(path)
        restored = PipelineState.load(path)
        assert restored.episodes == {}
        assert restored.open_target.positives == []

    def test_settings_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = self._make_state()
        state.save(path)
        r = PipelineState.load(path)
        assert r.settings.step_seconds == pytest.approx(0.05)
        assert r.settings.workers == 4
        assert r.settings.max_matches == 5
        assert r.settings.min_gap == pytest.approx(-10.0)
        assert r.settings.max_gap == pytest.approx(250.0)
        assert r.settings.fade == pytest.approx(0.3)
        assert r.settings.quiz_size == 8

    def test_open_target_positives_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = self._make_state()
        state.save(path)
        r = PipelineState.load(path)
        assert len(r.open_target.positives) == 1
        seg = r.open_target.positives[0]
        assert seg.stem == "ep1"
        assert seg.start == pytest.approx(10.0)
        assert seg.end == pytest.approx(20.0)
        assert seg.score == pytest.approx(0.95)

    def test_open_target_negatives_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = self._make_state()
        state.save(path)
        r = PipelineState.load(path)
        assert len(r.open_target.negatives) == 1
        assert r.open_target.negatives[0].stem == "ep2"

    def test_close_target_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = self._make_state()
        state.save(path)
        r = PipelineState.load(path)
        assert len(r.close_target.positives) == 1
        assert r.close_target.positives[0].stem == "ep3"
        assert r.close_target.negatives == []

    def test_episode_candidates_and_labels_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = self._make_state()
        state.save(path)
        r = PipelineState.load(path)

        ep = r.episodes["ep001"]
        assert ep.source_hash == "deadbeef"
        assert ep.cut is True

        assert len(ep.candidates["open"]) == 1
        assert ep.candidates["open"][0].label == "positive"
        assert ep.candidates["open"][0].score == pytest.approx(0.9)

        assert len(ep.candidates["close"]) == 2
        assert ep.candidates["close"][0].label == "negative"
        assert ep.candidates["close"][1].label is None

        assert len(ep.candidates["intro"]) == 1
        assert ep.candidates["intro"][0].start == pytest.approx(30.0)

    def test_snippet_profile_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = PipelineState()
        original = np.random.default_rng(42).random((20, 64)).astype(np.float32)
        state.snippets.append(SnippetEntry(name="open", profile=original))
        state.save(path)
        r = PipelineState.load(path)
        restored = r.profile_for("open")
        assert restored is not None
        np.testing.assert_array_almost_equal(restored, original, decimal=5)

    def test_multiple_snippets_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = PipelineState()
        state.snippets.append(SnippetEntry(name="open", profile=np.ones((5, 64), dtype=np.float32)))
        state.snippets.append(
            SnippetEntry(name="close", profile=np.zeros((5, 64), dtype=np.float32))
        )
        state.save(path)
        r = PipelineState.load(path)
        assert {s.name for s in r.snippets} == {"open", "close"}

    def test_deduplication_of_target_positives(self, tmp_path: Path) -> None:
        path = tmp_path / "state.toml"
        state = PipelineState()
        seg = Segment(stem="ep1", start=1.0, end=2.0, score=0.9)
        state.open_target.positives = [seg, seg]  # duplicate
        state.save(path)
        r = PipelineState.load(path)
        assert len(r.open_target.positives) == 1

    def test_load_missing_file_returns_empty_state(self, tmp_path: Path) -> None:
        r = PipelineState.load(tmp_path / "nonexistent.toml")
        assert r.episodes == {}
        assert r.open_target.positives == []

    def test_episode_method_creates_default_entry(self) -> None:
        state = PipelineState()
        ep = state.episode("new_stem")
        assert "new_stem" in state.episodes
        assert ep.class_for("open") == "undetected"
