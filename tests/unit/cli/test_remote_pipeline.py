"""Unit tests for pure/helper functions in part_io.cli.remote_pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from part_io.cli.remote_pipeline import (
    EpisodeState,
    MatchCandidate,
    PipelineState,
    SessionScores,
    _candidates_to_audio_matches,
    _chunks,
    _cmd_cut,
    _detect_batch_job,
    _detect_matches,
    _emit,
    _full_episodes,
    _interactive_review_episode,
    _pair_and_cut,
    _print_batch_summary,
    _review_bundle,
    main,
)


class TestMatchCandidate:
    def test_fields(self):
        m = MatchCandidate(index=0, score=0.9, start=1.0, end=2.0)
        assert m.index == 0
        assert m.score == pytest.approx(0.9)


class TestEpisodeState:
    def test_is_labeled_false_by_default(self):
        assert not EpisodeState().is_labeled()

    def test_is_labeled_open_approved(self):
        assert EpisodeState(open_approved=[0]).is_labeled()

    def test_is_labeled_open_rejected(self):
        assert EpisodeState(open_rejected=[1]).is_labeled()

    def test_is_labeled_close_approved(self):
        assert EpisodeState(close_approved=[2]).is_labeled()

    def test_is_labeled_close_rejected(self):
        assert EpisodeState(close_rejected=[3]).is_labeled()

    def test_has_matches_false_by_default(self):
        assert not EpisodeState().has_matches()

    def test_has_matches_true_with_open(self):
        assert EpisodeState(open_matches=[MatchCandidate(0, 0.9, 1.0, 2.0)]).has_matches()

    def test_has_matches_true_with_close(self):
        assert EpisodeState(close_matches=[MatchCandidate(0, 0.9, 1.0, 2.0)]).has_matches()


class TestPipelineState:
    def test_default_thresholds(self):
        state = PipelineState()
        assert state.open_threshold == pytest.approx(0.8)
        assert state.close_threshold == pytest.approx(0.8)

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
        assert state.open_threshold == pytest.approx(0.8)
        assert state.episodes == {}

    def test_save_and_reload_thresholds(self, tmp_path):
        path = tmp_path / "state.toml"
        PipelineState(open_threshold=0.75, close_threshold=0.9).save(path)
        loaded = PipelineState.load(path)
        assert loaded.open_threshold == pytest.approx(0.75)
        assert loaded.close_threshold == pytest.approx(0.9)

    def test_save_and_reload_episode_labels(self, tmp_path):
        path = tmp_path / "state.toml"
        state = PipelineState()
        ep = state.episode("ep001")
        ep.open_approved = [0, 2]
        ep.close_rejected = [1]
        ep.cut = True
        state.save(path)
        lep = PipelineState.load(path).episodes["ep001"]
        assert lep.open_approved == [0, 2]
        assert lep.close_rejected == [1]
        assert lep.cut is True

    def test_save_and_reload_match_candidates(self, tmp_path):
        path = tmp_path / "state.toml"
        state = PipelineState()
        ep = state.episode("ep001")
        ep.open_matches = [MatchCandidate(index=0, score=0.91, start=5.0, end=15.0)]
        ep.source = "downloads/remote/ep001.mp3"
        state.save(path)
        lep = PipelineState.load(path).episodes["ep001"]
        assert len(lep.open_matches) == 1
        assert lep.open_matches[0].score == pytest.approx(0.91)
        assert lep.source == "downloads/remote/ep001.mp3"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "state.toml"
        PipelineState().save(path)
        assert path.exists()


class TestSessionScores:
    def test_open_floor_none_when_empty(self):
        assert SessionScores().open_floor() is None

    def test_close_floor_none_when_empty(self):
        assert SessionScores().close_floor() is None

    def test_open_floor_returns_min_scaled(self):
        ss = SessionScores(approved_open=[0.9, 0.85, 0.92])
        assert ss.open_floor() == pytest.approx(0.85 * 0.995)

    def test_close_floor_returns_min_scaled(self):
        assert SessionScores(approved_close=[0.8]).close_floor() == pytest.approx(0.8 * 0.995)


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


class TestPrintBatchSummary:
    def test_prints_without_scores(self, capsys):
        _print_batch_summary(1, SessionScores(), 0.8, 0.8)
        out = capsys.readouterr().out
        assert "Batch 1 summary" in out
        assert "0.8000" in out

    def test_prints_min_score_when_approved(self, capsys):
        ss = SessionScores(approved_open=[0.9, 0.85])
        _print_batch_summary(2, ss, 0.85, 0.8)
        assert "0.8500" in capsys.readouterr().out


class TestCandidatesToAudioMatches:
    def test_empty_approved_includes_all(self):
        cands = [MatchCandidate(0, 0.9, 1.0, 2.0), MatchCandidate(1, 0.85, 5.0, 6.0)]
        assert len(_candidates_to_audio_matches(cands, approved=[])) == 2

    def test_approved_list_filters(self):
        cands = [
            MatchCandidate(0, 0.9, 1.0, 2.0),
            MatchCandidate(1, 0.85, 5.0, 6.0),
            MatchCandidate(2, 0.8, 10.0, 11.0),
        ]
        result = _candidates_to_audio_matches(cands, approved=[0, 2])
        assert len(result) == 2
        assert {m.start_seconds for m in result} == {1.0, 10.0}

    def test_empty_candidates(self):
        assert _candidates_to_audio_matches([], approved=[]) == []

    def test_fields_mapped(self):
        result = _candidates_to_audio_matches([MatchCandidate(0, 0.9, 3.0, 7.0)], approved=[])
        assert result[0].start_seconds == pytest.approx(3.0)
        assert result[0].end_seconds == pytest.approx(7.0)
        assert result[0].score == pytest.approx(0.9)


class TestDetectMatches:
    def _mock_run(self, payload, returncode=0, stderr=b""):
        r = MagicMock()
        r.returncode = returncode
        r.stdout = json.dumps(payload).encode() if isinstance(payload, list) else payload
        r.stderr = stderr
        return r

    def test_returns_candidates_on_success(self, tmp_path):
        r = self._mock_run([{"index": 0, "score": 0.9, "start": 5.0, "end": 10.0}])
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            result = _detect_matches(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=3.0,
                step_seconds=0.1,
                max_matches=10,
            )
        assert len(result) == 1
        assert result[0].index == 0
        assert result[0].score == pytest.approx(0.9)

    def test_returns_empty_on_nonzero_returncode(self, tmp_path):
        r = self._mock_run([], returncode=1)
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            result = _detect_matches(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=None,
                step_seconds=0.1,
                max_matches=10,
            )
        assert result == []

    def test_returns_empty_on_bad_json(self, tmp_path):
        r = MagicMock()
        r.returncode = 0
        r.stdout = b"not-json"
        r.stderr = b""
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            result = _detect_matches(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=None,
                step_seconds=0.1,
                max_matches=10,
            )
        assert result == []

    def test_z_threshold_appended(self, tmp_path):
        captured = []
        r = MagicMock()
        r.returncode = 0
        r.stdout = b"[]"
        r.stderr = b""

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return r

        with patch("part_io.cli.remote_pipeline.run_resolved", side_effect=fake_run):
            _detect_matches(
                tmp_path / "ep.mp3",
                tmp_path / "open.mp3",
                threshold=0.8,
                z_threshold=2.5,
                step_seconds=0.1,
                max_matches=5,
            )
        assert "--z-threshold" in captured
        assert "2.5" in captured


class TestDetectBatchJob:
    def test_returns_stem_kind_candidates(self, tmp_path):
        r = MagicMock()
        r.returncode = 0
        r.stdout = b'[{"index":0,"score":0.9,"start":1.0,"end":2.0}]'
        r.stderr = b""
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=r):
            stem, kind, cands = _detect_batch_job(
                tmp_path / "ep001.mp3",
                tmp_path / "open.mp3",
                "open",
                threshold=0.8,
                z_threshold=None,
                step_seconds=0.1,
                max_matches=10,
            )
        assert stem == "ep001"
        assert kind == "open"
        assert len(cands) == 1


class TestReviewBundle:
    def _mock_proc(self):
        p = MagicMock()
        p.poll.return_value = None
        return p

    def test_no_matches_prints_message(self, tmp_path, capsys):
        _review_bundle(
            [],
            "open",
            tmp_path / "ep.mp3",
            SessionScores(),
            snippet_path=tmp_path / "s.mp3",
            ep_state=EpisodeState(),
        )
        assert "No matches found" in capsys.readouterr().out

    def test_approve_updates_ep_state(self, tmp_path):
        matches = [MatchCandidate(index=0, score=0.9, start=5.0, end=15.0)]
        ep_state = EpisodeState()
        ss = SessionScores()
        with patch("part_io.cli.remote_pipeline._getch", side_effect=["a"]):
            with patch(
                "part_io.cli.remote_pipeline._start_audio_segment", return_value=self._mock_proc()
            ):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    _review_bundle(
                        matches,
                        "open",
                        tmp_path / "ep.mp3",
                        ss,
                        snippet_path=tmp_path / "s.mp3",
                        ep_state=ep_state,
                    )
        assert 0 in ep_state.open_approved
        assert ss.approved_open == [pytest.approx(0.9)]

    def test_reject_updates_ep_state(self, tmp_path):
        matches = [MatchCandidate(index=1, score=0.85, start=1.0, end=5.0)]
        ep_state = EpisodeState()
        ss = SessionScores()
        with patch("part_io.cli.remote_pipeline._getch", side_effect=["r"]):
            with patch(
                "part_io.cli.remote_pipeline._start_audio_segment", return_value=self._mock_proc()
            ):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    _review_bundle(
                        matches,
                        "close",
                        tmp_path / "ep.mp3",
                        ss,
                        snippet_path=tmp_path / "s.mp3",
                        ep_state=ep_state,
                    )
        assert 1 in ep_state.close_rejected
        assert ss.rejected_close == [pytest.approx(0.85)]

    def test_quit_raises_keyboard_interrupt(self, tmp_path):
        matches = [MatchCandidate(index=0, score=0.9, start=1.0, end=2.0)]
        with patch("part_io.cli.remote_pipeline._getch", side_effect=["q"]):
            with patch(
                "part_io.cli.remote_pipeline._start_audio_segment", return_value=self._mock_proc()
            ):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    with pytest.raises(KeyboardInterrupt):
                        _review_bundle(
                            matches,
                            "open",
                            tmp_path / "ep.mp3",
                            SessionScores(),
                            snippet_path=tmp_path / "s.mp3",
                            ep_state=EpisodeState(),
                        )

    def test_skip_advances_without_labeling(self, tmp_path):
        matches = [MatchCandidate(index=0, score=0.9, start=1.0, end=2.0)]
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._getch", side_effect=["s"]):
            with patch(
                "part_io.cli.remote_pipeline._start_audio_segment", return_value=self._mock_proc()
            ):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    _review_bundle(
                        matches,
                        "open",
                        tmp_path / "ep.mp3",
                        SessionScores(),
                        snippet_path=tmp_path / "s.mp3",
                        ep_state=ep_state,
                    )
        assert ep_state.open_approved == []
        assert ep_state.open_rejected == []


class TestInteractiveReviewEpisode:
    def test_calls_review_bundle_twice(self, tmp_path):
        state = PipelineState()
        ep = tmp_path / "ep001.mp3"
        with patch("part_io.cli.remote_pipeline._review_bundle") as mock_rb:
            _interactive_review_episode(
                ep,
                SessionScores(),
                state,
                open_sample=tmp_path / "open.mp3",
                close_sample=tmp_path / "close.mp3",
            )
        assert mock_rb.call_count == 2

    def test_uses_ep_state_source_when_set(self, tmp_path):
        state = PipelineState()
        ep_state = state.episode("ep001")
        ep_state.source = str(tmp_path / "remote" / "ep001.mp3")
        ep = tmp_path / "ep001.mp3"
        with patch("part_io.cli.remote_pipeline._review_bundle") as mock_rb:
            _interactive_review_episode(
                ep,
                SessionScores(),
                state,
                open_sample=tmp_path / "open.mp3",
                close_sample=tmp_path / "close.mp3",
            )
        called_source = mock_rb.call_args_list[0][0][2]
        assert called_source == Path(ep_state.source)


class TestPairAndCut:
    def _ep_with_matches(self):
        ep = EpisodeState()
        ep.open_matches = [MatchCandidate(0, 0.9, 0.0, 10.0)]
        ep.close_matches = [MatchCandidate(0, 0.9, 30.0, 40.0)]
        return ep

    def test_returns_skipped_when_no_matches(self, tmp_path):
        result = _pair_and_cut(
            "ep001",
            tmp_path / "ep.mp3",
            output_dir=tmp_path / "out",
            ep_state=EpisodeState(),
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
                    ep_state=self._ep_with_matches(),
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
                        "part_io.cli.remote_pipeline._build_filter_complex", return_value=("f", 2)
                    ):
                        with patch("part_io.cli.remote_pipeline._run_ffmpeg", return_value=1):
                            result = _pair_and_cut(
                                "ep001",
                                tmp_path / "ep.mp3",
                                output_dir=tmp_path / "out",
                                ep_state=self._ep_with_matches(),
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
                        "part_io.cli.remote_pipeline._build_filter_complex", return_value=("f", 2)
                    ):
                        with patch("part_io.cli.remote_pipeline._run_ffmpeg", return_value=0):
                            result = _pair_and_cut(
                                "ep001",
                                tmp_path / "ep.mp3",
                                output_dir=tmp_path / "out",
                                ep_state=self._ep_with_matches(),
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
                ep_state=self._ep_with_matches(),
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
                ep_state=self._ep_with_matches(),
                min_gap=-15.0,
                max_gap=600.0,
                yes=True,
                dry_run=False,
            )
        assert result == "skipped"


class TestCmdCut:
    def _save_labeled_state(self, tmp_path):
        state = PipelineState()
        ep = state.episode("ep001")
        ep.open_approved = [0]
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

    def test_no_labeled_episodes_returns_early(self, tmp_path, capsys):
        PipelineState().save(tmp_path / "review" / "state.toml")
        _cmd_cut(self._make_args(tmp_path))
        assert "No labeled episodes" in capsys.readouterr().out

    def test_missing_source_skips(self, tmp_path, capsys):
        self._save_labeled_state(tmp_path)
        args = self._make_args(tmp_path)
        args.remote_dir.mkdir(parents=True, exist_ok=True)
        _cmd_cut(args)
        assert "SKIP" in capsys.readouterr().out

    def test_cut_result_marks_episode_as_cut(self, tmp_path):
        self._save_labeled_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="cut"):
            _cmd_cut(self._make_args(tmp_path))
        assert PipelineState.load(tmp_path / "review" / "state.toml").episodes["ep001"].cut is True

    def test_failed_result_exits_1(self, tmp_path):
        self._save_labeled_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="failed"):
            with pytest.raises(SystemExit) as exc:
                _cmd_cut(self._make_args(tmp_path))
        assert exc.value.code == 1

    def test_skipped_not_marked_as_cut(self, tmp_path):
        self._save_labeled_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="skipped"):
            _cmd_cut(self._make_args(tmp_path))
        assert PipelineState.load(tmp_path / "review" / "state.toml").episodes["ep001"].cut is False


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
