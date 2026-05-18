"""Unit tests for pure/helper functions in part_io.cli.remote_pipeline."""

from __future__ import annotations

import argparse
import json
import math
from unittest.mock import MagicMock, patch

import pytest

from part_io.cli.remote_pipeline import (
    _NEG,
    _POS,
    _UNC,
    _UND,
    EpisodeState,
    PipelineState,
    Segment,
    TargetState,
    _chunks,
    _classify_score,
    _cmd_cut,
    _compute_thresholds,
    _count_uncertain,
    _detect_best,
    _emit,
    _full_episodes,
    _Match,
    _moe,
    _next_uncertain,
    _pair_and_cut,
    _reclassify_all,
    _review_one_target,
    _UndoEntry,
    main,
)

# ---------------------------------------------------------------------------
# Segment / TargetState / EpisodeState
# ---------------------------------------------------------------------------


class TestSegment:
    def test_fields(self):
        s = Segment(source="ep.mp3", start=1.0, end=2.0, score=0.9)
        assert s.source == "ep.mp3"
        assert s.start == pytest.approx(1.0)
        assert s.end == pytest.approx(2.0)
        assert s.score == pytest.approx(0.9)


class TestTargetState:
    def test_default_empty(self):
        t = TargetState()
        assert t.positives == []
        assert t.negatives == []


class TestEpisodeState:
    def test_defaults(self):
        ep = EpisodeState()
        assert ep.open_class == _UND
        assert ep.close_class == _UND
        assert not ep.cut

    def test_is_detected_false_by_default(self):
        assert not EpisodeState().is_detected()

    def test_is_detected_true_when_open_classified(self):
        assert EpisodeState(open_class=_UNC).is_detected()

    def test_is_detected_true_when_close_classified(self):
        assert EpisodeState(close_class=_NEG).is_detected()

    def test_is_cuttable_requires_both_positive(self):
        assert not EpisodeState().is_cuttable()
        assert not EpisodeState(open_class=_POS).is_cuttable()
        assert not EpisodeState(close_class=_POS).is_cuttable()
        assert EpisodeState(open_class=_POS, close_class=_POS).is_cuttable()


# ---------------------------------------------------------------------------
# PipelineState save / load
# ---------------------------------------------------------------------------


class TestPipelineState:
    def test_episode_creates_on_demand(self):
        state = PipelineState()
        ep = state.episode("foo")
        assert isinstance(ep, EpisodeState)
        assert "foo" in state.episodes

    def test_episode_returns_same_object(self):
        state = PipelineState()
        assert state.episode("bar") is state.episode("bar")

    def test_load_missing_file_returns_default(self, tmp_path):
        state = PipelineState.load(tmp_path / "state.toml")
        assert state.episodes == {}
        assert state.open_target.positives == []

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "state.toml"
        PipelineState().save(path)
        assert path.exists()

    def test_roundtrip_episode_fields(self, tmp_path):
        path = tmp_path / "state.toml"
        state = PipelineState()
        ep = state.episode("ep001")
        ep.source = "downloads/remote/ep001.mp3"
        ep.open_score = 0.95
        ep.open_start = 10.0
        ep.open_end = 20.0
        ep.open_class = _POS
        ep.close_class = _NEG
        ep.cut = True
        state.save(path)
        loaded = PipelineState.load(path)
        lep = loaded.episodes["ep001"]
        assert lep.source == "downloads/remote/ep001.mp3"
        assert lep.open_score == pytest.approx(0.95)
        assert lep.open_class == _POS
        assert lep.close_class == _NEG
        assert lep.cut is True

    def test_roundtrip_targets(self, tmp_path):
        path = tmp_path / "state.toml"
        state = PipelineState()
        state.open_target.positives.append(Segment("ep.mp3", 1.0, 2.0, 0.92))
        state.close_target.negatives.append(Segment("ep.mp3", 5.0, 6.0, 0.78))
        state.save(path)
        loaded = PipelineState.load(path)
        assert len(loaded.open_target.positives) == 1
        assert loaded.open_target.positives[0].score == pytest.approx(0.92)
        assert len(loaded.close_target.negatives) == 1
        assert loaded.close_target.negatives[0].start == pytest.approx(5.0)

    def test_migration_from_old_format(self, tmp_path):
        path = tmp_path / "state.toml"
        old_toml = (
            "[thresholds]\nopen = 0.85\nclose = 0.83\n\n"
            '[episodes."ep001"]\n'
            'source = "downloads/remote/ep001.mp3"\n'
            "open_matches = [{index = 1, score = 0.9704, start = 10.0, end = 20.0}]\n"
            "close_matches = [{index = 1, score = 0.8369, start = 30.0, end = 40.0}]\n"
            "open_approved = [1]\nopen_rejected = []\n"
            "close_approved = [1]\nclose_rejected = []\n"
            "cut = false\n"
        )
        path.write_text(old_toml, encoding="utf-8")
        state = PipelineState.load(path)
        ep = state.episodes["ep001"]
        assert ep.open_class == _POS
        assert ep.close_class == _POS
        assert ep.open_score == pytest.approx(0.9704)
        assert len(state.open_target.positives) == 1
        assert len(state.close_target.positives) == 1

    def test_migration_rejected_becomes_negative(self, tmp_path):
        path = tmp_path / "state.toml"
        old_toml = (
            '[episodes."ep002"]\n'
            'source = "ep002.mp3"\n'
            "open_matches = [{index = 1, score = 0.85, start = 5.0, end = 10.0}]\n"
            "close_matches = []\n"
            "open_approved = []\nopen_rejected = [1]\n"
            "close_approved = []\nclose_rejected = []\n"
            "cut = false\n"
        )
        path.write_text(old_toml, encoding="utf-8")
        state = PipelineState.load(path)
        ep = state.episodes["ep002"]
        assert ep.open_class == _NEG
        assert ep.close_class == _UND
        assert len(state.open_target.negatives) == 1

    def test_special_chars_in_stem(self, tmp_path):
        path = tmp_path / "state.toml"
        state = PipelineState()
        ep = state.episode("http://example.com?p=1")
        ep.open_class = _UNC
        state.save(path)
        loaded = PipelineState.load(path)
        assert "http://example.com?p=1" in loaded.episodes


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestMoe:
    def test_empty_returns_zero(self):
        assert _moe([]) == pytest.approx(0.0)

    def test_single_returns_zero(self):
        assert _moe([0.9]) == pytest.approx(0.0)

    def test_two_equal_returns_zero(self):
        assert _moe([0.9, 0.9]) == pytest.approx(0.0)

    def test_nonzero_spread(self):
        scores = [0.9, 0.8]
        mean = 0.85
        std = math.sqrt(((0.9 - mean) ** 2 + (0.8 - mean) ** 2) / 2)
        assert _moe(scores, k=1.5) == pytest.approx(1.5 * std)


class TestComputeThresholds:
    def test_no_positives_uses_default(self):
        t = TargetState()
        tp, tm = _compute_thresholds(t, default_floor=0.8)
        assert tp == pytest.approx(0.8)
        assert tm == pytest.approx(-math.inf)

    def test_positives_set_theta_plus(self):
        t = TargetState(
            positives=[Segment("a.mp3", 0.0, 1.0, 0.9), Segment("b.mp3", 0.0, 1.0, 0.85)]
        )
        tp, _ = _compute_thresholds(t, default_floor=0.8)
        # theta_plus = min(0.9, 0.85) - moe([0.9, 0.85])
        assert tp < 0.85  # moe brings it below min

    def test_negatives_set_theta_minus(self):
        t = TargetState(negatives=[Segment("a.mp3", 0.0, 1.0, 0.7)])
        _, tm = _compute_thresholds(t, default_floor=0.8)
        assert tm == pytest.approx(0.7)  # single negative, moe=0


class TestClassifyScore:
    def test_above_plus_is_positive(self):
        assert _classify_score(0.95, theta_plus=0.9, theta_minus=0.5) == _POS

    def test_below_minus_is_negative(self):
        assert _classify_score(0.4, theta_plus=0.9, theta_minus=0.5) == _NEG

    def test_in_band_is_uncertain(self):
        assert _classify_score(0.7, theta_plus=0.9, theta_minus=0.5) == _UNC

    def test_positive_wins_on_overlap(self):
        # theta_minus >= theta_plus — score at theta_plus should be positive
        assert _classify_score(0.9, theta_plus=0.9, theta_minus=0.95) == _POS


class TestReclassifyAll:
    def test_reclassifies_uncertain_episodes(self):
        state = PipelineState()
        ep = state.episode("ep1")
        ep.open_score = 0.95
        ep.open_class = _UNC
        state.open_target.positives.append(Segment("ep1.mp3", 0.0, 1.0, 0.9))
        _reclassify_all(state, default_floor=0.8)
        assert ep.open_class == _POS

    def test_does_not_reclassify_already_classified(self):
        state = PipelineState()
        ep = state.episode("ep1")
        ep.open_score = 0.5
        ep.open_class = _POS  # manually set — should not be touched
        _reclassify_all(state, default_floor=0.8)
        assert ep.open_class == _POS

    def test_does_not_reclassify_undetected(self):
        state = PipelineState()
        ep = state.episode("ep1")
        ep.open_class = _UND
        _reclassify_all(state, default_floor=0.8)
        assert ep.open_class == _UND

    def test_no_auto_classify_without_evidence(self):
        """Episodes stay uncertain when no confirmed examples exist yet."""
        state = PipelineState()
        ep = state.episode("ep1")
        ep.open_score = 0.95  # above default floor
        ep.open_class = _UNC
        _reclassify_all(state, default_floor=0.8)
        assert ep.open_class == _UNC  # no evidence → no auto-classification


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


class TestDetectBest:
    def _mock_run(self, payload, returncode=0):
        r = MagicMock()
        r.returncode = returncode
        r.stdout = json.dumps(payload).encode() if isinstance(payload, list) else payload
        r.stderr = b""
        return r

    def test_returns_match_on_success(self, tmp_path):
        r = self._mock_run([{"index": 1, "score": 0.9, "start": 5.0, "end": 10.0}])
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            result = _detect_best(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=3.0,
                step_seconds=0.1,
            )
        assert isinstance(result, _Match)
        assert result.score == pytest.approx(0.9)
        assert result.start == pytest.approx(5.0)

    def test_returns_none_on_empty_result(self, tmp_path):
        r = self._mock_run([])
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            result = _detect_best(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=None,
                step_seconds=0.1,
            )
        assert result is None

    def test_returns_none_on_nonzero_returncode(self, tmp_path):
        r = self._mock_run([], returncode=1)
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            result = _detect_best(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=None,
                step_seconds=0.1,
            )
        assert result is None

    def test_returns_none_on_bad_json(self, tmp_path):
        r = MagicMock()
        r.returncode = 0
        r.stdout = b"not-json"
        r.stderr = b""
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            result = _detect_best(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=None,
                step_seconds=0.1,
            )
        assert result is None

    def test_z_threshold_appended(self, tmp_path):
        captured: list[str] = []
        r = MagicMock()
        r.returncode = 0
        r.stdout = b"[]"
        r.stderr = b""

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return r

        with patch("part_io.cli.remote_pipeline.run_resolved", side_effect=fake_run):
            _detect_best(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=2.5,
                step_seconds=0.1,
            )
        assert "--z-threshold" in captured
        assert "2.5" in captured

    def test_max_matches_is_one(self, tmp_path):
        captured: list[str] = []
        r = MagicMock()
        r.returncode = 0
        r.stdout = b"[]"
        r.stderr = b""

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return r

        with patch("part_io.cli.remote_pipeline.run_resolved", side_effect=fake_run):
            _detect_best(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=None,
                step_seconds=0.1,
            )
        idx = captured.index("--max-matches")
        assert captured[idx + 1] == "1"


# ---------------------------------------------------------------------------
# Review helpers
# ---------------------------------------------------------------------------


class TestCountUncertain:
    def test_empty_state(self):
        assert _count_uncertain(PipelineState()) == 0

    def test_counts_uncertain_open_and_close(self):
        state = PipelineState()
        ep = state.episode("ep1")
        ep.open_class = _UNC
        ep.close_class = _UNC
        assert _count_uncertain(state) == 2

    def test_does_not_count_undetected(self):
        state = PipelineState()
        state.episode("ep1").open_class = _UND
        assert _count_uncertain(state) == 0


class TestNextUncertain:
    def test_returns_none_when_no_uncertain(self):
        state = PipelineState()
        state.episode("ep1").open_class = _POS
        assert _next_uncertain(state) is None

    def test_returns_highest_score_first(self):
        state = PipelineState()
        ep1 = state.episode("ep1")
        ep1.open_class = _UNC
        ep1.open_score = 0.85
        ep2 = state.episode("ep2")
        ep2.open_class = _UNC
        ep2.open_score = 0.92
        result = _next_uncertain(state)
        assert result == ("ep2", "open")

    def test_excludes_skipped(self):
        state = PipelineState()
        ep1 = state.episode("ep1")
        ep1.open_class = _UNC
        ep1.open_score = 0.92
        ep2 = state.episode("ep2")
        ep2.open_class = _UNC
        ep2.open_score = 0.85
        result = _next_uncertain(state, exclude={("open", "ep1")})
        assert result == ("ep2", "open")

    def test_ignores_zero_score(self):
        state = PipelineState()
        ep = state.episode("ep1")
        ep.open_class = _UNC
        ep.open_score = 0.0
        assert _next_uncertain(state) is None


class TestReviewOneTarget:
    def _make_state(self, tmp_path, stem="ep001"):
        state = PipelineState()
        ep = state.episode(stem)
        ep.source = str(tmp_path / f"{stem}.mp3")
        ep.open_score = 0.92
        ep.open_start = 5.0
        ep.open_end = 15.0
        ep.open_class = _UNC
        ep.close_score = 0.87
        ep.close_start = 30.0
        ep.close_end = 40.0
        ep.close_class = _UNC
        return state

    def _mock_proc(self):
        p = MagicMock()
        p.poll.return_value = None
        return p

    def _review(self, state, stem, kind, keys, tmp_path, history=None):
        if history is None:
            history = []
        with patch("part_io.cli.remote_pipeline._getch", side_effect=keys):
            with patch(
                "part_io.cli.remote_pipeline._start_audio_segment",
                return_value=self._mock_proc(),
            ):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    return _review_one_target(
                        state,
                        stem,
                        kind,
                        open_sample=tmp_path / "open.mp3",
                        close_sample=tmp_path / "close.mp3",
                        history=history,
                        default_floor=0.8,
                    )

    def test_approve_classifies_positive(self, tmp_path):
        state = self._make_state(tmp_path)
        result = self._review(state, "ep001", "open", ["a"], tmp_path)
        assert result == "classified"
        assert state.episodes["ep001"].open_class == _POS
        assert len(state.open_target.positives) == 1

    def test_reject_classifies_negative(self, tmp_path):
        state = self._make_state(tmp_path)
        result = self._review(state, "ep001", "close", ["r"], tmp_path)
        assert result == "classified"
        assert state.episodes["ep001"].close_class == _NEG
        assert len(state.close_target.negatives) == 1

    def test_skip_returns_skipped_and_leaves_uncertain(self, tmp_path):
        state = self._make_state(tmp_path)
        result = self._review(state, "ep001", "open", ["s"], tmp_path)
        assert result == "skipped"
        assert state.episodes["ep001"].open_class == _UNC

    def test_quit_raises_keyboard_interrupt(self, tmp_path):
        state = self._make_state(tmp_path)
        with pytest.raises(KeyboardInterrupt):
            self._review(state, "ep001", "open", ["q"], tmp_path)

    def test_approve_triggers_reclassify(self, tmp_path):
        state = self._make_state(tmp_path)
        # ep001 open_score = 0.92; after approval theta_plus = 0.92 (moe=0 with one point)
        # ep002 at 0.95 is above theta_plus and should auto-classify positive
        ep2 = state.episode("ep002")
        ep2.source = str(tmp_path / "ep002.mp3")
        ep2.open_score = 0.95
        ep2.open_class = _UNC
        self._review(state, "ep001", "open", ["a"], tmp_path)
        assert state.episodes["ep002"].open_class == _POS

    def test_undo_reverts_previous_decision(self, tmp_path):
        state = self._make_state(tmp_path)
        history: list[_UndoEntry] = []
        # Approve open
        self._review(state, "ep001", "open", ["a"], tmp_path, history=history)
        assert state.episodes["ep001"].open_class == _POS
        assert len(history) == 1
        # Review close, press u to undo the open approval
        result = self._review(state, "ep001", "close", ["u"], tmp_path, history=history)
        assert result == "undone"
        assert state.episodes["ep001"].open_class == _UNC
        assert len(state.open_target.positives) == 0
        assert len(history) == 0

    def test_close_kind_uses_close_target(self, tmp_path):
        state = self._make_state(tmp_path)
        self._review(state, "ep001", "close", ["a"], tmp_path)
        assert len(state.close_target.positives) == 1
        assert len(state.open_target.positives) == 0


# ---------------------------------------------------------------------------
# Cut helpers
# ---------------------------------------------------------------------------


class TestPairAndCut:
    def _make_ep(self):
        ep = EpisodeState()
        ep.source = "ep001.mp3"
        ep.open_score = 0.9
        ep.open_start = 0.0
        ep.open_end = 10.0
        ep.open_class = _POS
        ep.close_score = 0.9
        ep.close_start = 30.0
        ep.close_end = 40.0
        ep.close_class = _POS
        return ep

    def test_returns_skipped_when_not_cuttable(self, tmp_path):
        ep = EpisodeState(open_class=_UNC, close_class=_POS)
        result = _pair_and_cut(
            "ep001",
            tmp_path / "ep.mp3",
            output_dir=tmp_path / "out",
            ep_state=ep,
            min_gap=-15.0,
            max_gap=600.0,
            yes=True,
            dry_run=False,
        )
        assert result == "skipped"

    def test_dry_run_returns_skipped(self, tmp_path):
        seg = MagicMock()
        seg.cut_start, seg.cut_end = 10.0, 30.0
        with patch("part_io.cli.remote_pipeline.pair_ad_segments", return_value=([seg], [], [])):
            with patch("part_io.cli.remote_pipeline._validate_segments"):
                result = _pair_and_cut(
                    "ep001",
                    tmp_path / "ep.mp3",
                    output_dir=tmp_path / "out",
                    ep_state=self._make_ep(),
                    min_gap=-15.0,
                    max_gap=600.0,
                    yes=True,
                    dry_run=True,
                )
        assert result == "skipped"

    def test_ffmpeg_failure_returns_failed(self, tmp_path):
        seg = MagicMock()
        seg.cut_start, seg.cut_end = 10.0, 30.0
        with patch("part_io.cli.remote_pipeline.pair_ad_segments", return_value=([seg], [], [])):
            with patch("part_io.cli.remote_pipeline._validate_segments"):
                with patch("part_io.cli.remote_pipeline._build_keep_spans", return_value=[]):
                    with patch(
                        "part_io.cli.remote_pipeline._build_filter_complex",
                        return_value=("f", 2),
                    ):
                        with patch("part_io.cli.remote_pipeline._run_ffmpeg", return_value=1):
                            result = _pair_and_cut(
                                "ep001",
                                tmp_path / "ep.mp3",
                                output_dir=tmp_path / "out",
                                ep_state=self._make_ep(),
                                min_gap=-15.0,
                                max_gap=600.0,
                                yes=True,
                                dry_run=False,
                            )
        assert result == "failed"

    def test_successful_cut(self, tmp_path):
        seg = MagicMock()
        seg.cut_start, seg.cut_end = 10.0, 30.0
        with patch("part_io.cli.remote_pipeline.pair_ad_segments", return_value=([seg], [], [])):
            with patch("part_io.cli.remote_pipeline._validate_segments"):
                with patch("part_io.cli.remote_pipeline._build_keep_spans", return_value=[]):
                    with patch(
                        "part_io.cli.remote_pipeline._build_filter_complex",
                        return_value=("f", 2),
                    ):
                        with patch("part_io.cli.remote_pipeline._run_ffmpeg", return_value=0):
                            result = _pair_and_cut(
                                "ep001",
                                tmp_path / "ep.mp3",
                                output_dir=tmp_path / "out",
                                ep_state=self._make_ep(),
                                min_gap=-15.0,
                                max_gap=600.0,
                                yes=True,
                                dry_run=False,
                            )
        assert result == "cut"

    def test_pair_exception_returns_skipped(self, tmp_path):
        with patch("part_io.cli.remote_pipeline.pair_ad_segments", side_effect=ValueError("bad")):
            result = _pair_and_cut(
                "ep001",
                tmp_path / "ep.mp3",
                output_dir=tmp_path / "out",
                ep_state=self._make_ep(),
                min_gap=-15.0,
                max_gap=600.0,
                yes=True,
                dry_run=False,
            )
        assert result == "skipped"

    def test_no_segments_returns_skipped(self, tmp_path):
        with patch("part_io.cli.remote_pipeline.pair_ad_segments", return_value=([], [], [])):
            result = _pair_and_cut(
                "ep001",
                tmp_path / "ep.mp3",
                output_dir=tmp_path / "out",
                ep_state=self._make_ep(),
                min_gap=-15.0,
                max_gap=600.0,
                yes=True,
                dry_run=False,
            )
        assert result == "skipped"


# ---------------------------------------------------------------------------
# cmd_cut
# ---------------------------------------------------------------------------


class TestCmdCut:
    def _save_cuttable_state(self, tmp_path):
        state = PipelineState()
        ep = state.episode("ep001")
        ep.source = str(tmp_path / "remote" / "ep001.mp3")
        ep.open_class = _POS
        ep.close_class = _POS
        state_path = tmp_path / "review" / "state.toml"
        state.save(state_path)
        return state_path

    def _make_args(self, tmp_path):
        return argparse.Namespace(
            review_root=tmp_path / "review",
            remote_dir=tmp_path / "remote",
            output_dir=tmp_path / "out",
            min_gap=-15.0,
            max_gap=600.0,
            yes=True,
            dry_run=False,
        )

    def test_no_cuttable_episodes_returns_early(self, tmp_path, capsys):
        PipelineState().save(tmp_path / "review" / "state.toml")
        _cmd_cut(self._make_args(tmp_path))
        assert "No cuttable" in capsys.readouterr().out

    def test_missing_source_skips(self, tmp_path, capsys):
        self._save_cuttable_state(tmp_path)
        args = self._make_args(tmp_path)
        args.remote_dir.mkdir(parents=True, exist_ok=True)
        _cmd_cut(args)
        assert "SKIP" in capsys.readouterr().out

    def test_cut_result_marks_episode_as_cut(self, tmp_path):
        self._save_cuttable_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="cut"):
            _cmd_cut(self._make_args(tmp_path))
        assert PipelineState.load(tmp_path / "review" / "state.toml").episodes["ep001"].cut is True

    def test_failed_result_exits_1(self, tmp_path):
        self._save_cuttable_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="failed"):
            with pytest.raises(SystemExit) as exc:
                _cmd_cut(self._make_args(tmp_path))
        assert exc.value.code == 1

    def test_skipped_not_marked_as_cut(self, tmp_path):
        self._save_cuttable_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="skipped"):
            _cmd_cut(self._make_args(tmp_path))
        assert PipelineState.load(tmp_path / "review" / "state.toml").episodes["ep001"].cut is False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestChunks:
    def test_even_split(self):
        assert list(_chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        assert list(_chunks([1, 2, 3], 2)) == [[1, 2], [3]]

    def test_empty(self):
        assert list(_chunks([], 3)) == []

    def test_size_larger_than_list(self):
        assert list(_chunks([1, 2], 10)) == [[1, 2]]


class TestFullEpisodes:
    _MIN = 10 * 1024 * 1024

    def test_returns_large_mp3s_sorted(self, tmp_path):
        (tmp_path / "small.mp3").write_bytes(b"x")
        big1 = tmp_path / "ep_b.mp3"
        big1.write_bytes(b"x" * self._MIN)
        big2 = tmp_path / "ep_a.mp3"
        big2.write_bytes(b"x" * self._MIN)
        assert _full_episodes(tmp_path) == sorted([big1, big2])

    def test_empty_directory(self, tmp_path):
        assert _full_episodes(tmp_path) == []

    def test_ignores_non_mp3(self, tmp_path):
        (tmp_path / "ep.wav").write_bytes(b"x" * self._MIN)
        assert _full_episodes(tmp_path) == []

    def test_ignores_small_mp3(self, tmp_path):
        (tmp_path / "ep.mp3").write_bytes(b"x" * 100)
        assert _full_episodes(tmp_path) == []


class TestEmit:
    def test_writes_to_stderr(self, capsys):
        _emit("hello world")
        assert "hello world" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main dispatch
# ---------------------------------------------------------------------------


class TestMain:
    def test_review_dispatches(self):
        with patch("part_io.cli.remote_pipeline._cmd_review") as m:
            with patch("sys.argv", ["remote_pipeline", "review"]):
                main()
        m.assert_called_once()

    def test_cut_dispatches(self):
        with patch("part_io.cli.remote_pipeline._cmd_cut") as m:
            with patch("sys.argv", ["remote_pipeline", "cut"]):
                main()
        m.assert_called_once()

    def test_loop_dispatches(self):
        with patch("part_io.cli.remote_pipeline._cmd_loop") as m:
            with patch("sys.argv", ["remote_pipeline", "loop"]):
                main()
        m.assert_called_once()

    def test_no_subcommand_exits(self):
        with patch("sys.argv", ["remote_pipeline"]):
            with pytest.raises(SystemExit):
                main()
