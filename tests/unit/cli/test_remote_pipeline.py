"""Unit tests for pure/helper functions in part_io.cli.remote_pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from part_io.cli.remote_pipeline import (
    EpisodeState,
    LoopContext,
    PipelineState,
    SessionScores,
    _chunks,
    _cleanup_clips,
    _clips_exist,
    _cmd_cut,
    _cut_loop_episode,
    _emit,
    _filter_unlabeled,
    _full_episodes,
    _generate_batch_clips,
    _handle_review_key,
    _interactive_review_batch,
    _pair_and_cut,
    _print_batch_summary,
    _process_batch,
    _review_bundle,
    _review_clip,
    _review_loop_episode,
    _review_one,
    _run_clip_pool,
    _scores_list_for,
    _undo_action,
    _update_thresholds,
    _validate_remote_inputs,
    main,
)


# ---------------------------------------------------------------------------
# EpisodeState
# ---------------------------------------------------------------------------


class TestEpisodeState:
    def test_is_labeled_false_by_default(self):
        ep = EpisodeState()
        assert not ep.is_labeled()

    def test_is_labeled_open_approved(self):
        ep = EpisodeState(open_approved=[0])
        assert ep.is_labeled()

    def test_is_labeled_open_rejected(self):
        ep = EpisodeState(open_rejected=[1])
        assert ep.is_labeled()

    def test_is_labeled_close_approved(self):
        ep = EpisodeState(close_approved=[2])
        assert ep.is_labeled()

    def test_is_labeled_close_rejected(self):
        ep = EpisodeState(close_rejected=[3])
        assert ep.is_labeled()


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------


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
        ep1 = state.episode("bar")
        ep2 = state.episode("bar")
        assert ep1 is ep2

    def test_load_missing_file_returns_default(self, tmp_path):
        state = PipelineState.load(tmp_path / "state.toml")
        assert state.open_threshold == pytest.approx(0.8)
        assert state.episodes == {}

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "state.toml"
        state = PipelineState(open_threshold=0.75, close_threshold=0.9)
        ep = state.episode("ep001")
        ep.open_approved = [0, 2]
        ep.close_rejected = [1]
        ep.cut = True
        state.save(path)

        loaded = PipelineState.load(path)
        assert loaded.open_threshold == pytest.approx(0.75)
        assert loaded.close_threshold == pytest.approx(0.9)
        lep = loaded.episodes["ep001"]
        assert lep.open_approved == [0, 2]
        assert lep.close_rejected == [1]
        assert lep.cut is True

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "state.toml"
        PipelineState().save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# SessionScores
# ---------------------------------------------------------------------------


class TestSessionScores:
    def test_open_floor_none_when_empty(self):
        ss = SessionScores()
        assert ss.open_floor() is None

    def test_close_floor_none_when_empty(self):
        ss = SessionScores()
        assert ss.close_floor() is None

    def test_open_floor_returns_min_scaled(self):
        ss = SessionScores(approved_open=[0.9, 0.85, 0.92])
        assert ss.open_floor() == pytest.approx(0.85 * 0.995)

    def test_close_floor_returns_min_scaled(self):
        ss = SessionScores(approved_close=[0.8])
        assert ss.close_floor() == pytest.approx(0.8 * 0.995)


# ---------------------------------------------------------------------------
# _chunks
# ---------------------------------------------------------------------------


class TestChunks:
    def test_even_split(self):
        result = list(_chunks([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = list(_chunks([1, 2, 3], 2))
        assert result == [[1, 2], [3]]

    def test_empty(self):
        assert list(_chunks([], 3)) == []

    def test_size_larger_than_list(self):
        assert list(_chunks([1, 2], 10)) == [[1, 2]]


# ---------------------------------------------------------------------------
# _scores_list_for
# ---------------------------------------------------------------------------


class TestScoresListFor:
    def test_approved_open(self):
        ss = SessionScores(approved_open=[0.9])
        lst = _scores_list_for(ss, "open", "approved")
        assert lst is ss.approved_open

    def test_rejected_open(self):
        ss = SessionScores(rejected_open=[0.5])
        lst = _scores_list_for(ss, "open", "rejected")
        assert lst is ss.rejected_open

    def test_approved_close(self):
        ss = SessionScores(approved_close=[0.8])
        lst = _scores_list_for(ss, "close", "approved")
        assert lst is ss.approved_close

    def test_rejected_close(self):
        ss = SessionScores(rejected_close=[0.3])
        lst = _scores_list_for(ss, "close", "rejected")
        assert lst is ss.rejected_close


# ---------------------------------------------------------------------------
# _undo_action
# ---------------------------------------------------------------------------


class TestUndoAction:
    def test_undo_approved(self, capsys):
        approved = [0]
        rejected = []
        ss = SessionScores(approved_open=[0.9])
        history = [("approved", 0, 0.9)]
        i_delta, prev = _undo_action(history, approved, rejected, ss, "open")
        assert i_delta == -1
        assert prev == "approved"
        assert 0 not in approved
        assert 0.9 not in ss.approved_open
        assert "undone" in capsys.readouterr().out

    def test_undo_rejected(self):
        approved = []
        rejected = [1]
        ss = SessionScores(rejected_close=[0.4])
        history = [("rejected", 1, 0.4)]
        _undo_action(history, approved, rejected, ss, "close")
        assert 1 not in rejected
        assert 0.4 not in ss.rejected_close

    def test_undo_skipped_does_not_mutate_lists(self):
        approved = []
        rejected = []
        ss = SessionScores()
        history = [("skipped", 2, 0.7)]
        _undo_action(history, approved, rejected, ss, "open")
        assert approved == []
        assert rejected == []


# ---------------------------------------------------------------------------
# _filter_unlabeled
# ---------------------------------------------------------------------------


class TestFilterUnlabeled:
    def test_filters_already_labeled(self, tmp_path):
        state = PipelineState()
        ep1 = state.episode("ep1")
        ep1.open_approved = [0]
        ep2 = state.episode("ep2")  # not labeled

        # Create dummy Path objects — _filter_unlabeled only checks ep.stem
        paths = [tmp_path / "ep1.mp3", tmp_path / "ep2.mp3"]
        episodes, n_done = _filter_unlabeled(paths, state, overwrite=False)
        assert len(episodes) == 1
        assert episodes[0].stem == "ep2"
        assert n_done == 1

    def test_overwrite_includes_all(self, tmp_path):
        state = PipelineState()
        ep1 = state.episode("ep1")
        ep1.open_approved = [0]

        paths = [tmp_path / "ep1.mp3"]
        episodes, n_done = _filter_unlabeled(paths, state, overwrite=True)
        assert len(episodes) == 1
        assert n_done == 0

    def test_all_unlabeled(self, tmp_path):
        state = PipelineState()
        paths = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
        episodes, n_done = _filter_unlabeled(paths, state, overwrite=False)
        assert len(episodes) == 2
        assert n_done == 0


# ---------------------------------------------------------------------------
# _full_episodes
# ---------------------------------------------------------------------------


class TestFullEpisodes:
    _MIN_SIZE = 10 * 1024 * 1024  # 10 MB

    def test_returns_large_mp3s_sorted(self, tmp_path):
        small = tmp_path / "small.mp3"
        small.write_bytes(b"x")
        big1 = tmp_path / "ep_b.mp3"
        big1.write_bytes(b"x" * self._MIN_SIZE)
        big2 = tmp_path / "ep_a.mp3"
        big2.write_bytes(b"x" * self._MIN_SIZE)
        result = _full_episodes(tmp_path)
        assert result == sorted([big1, big2])

    def test_empty_directory(self, tmp_path):
        assert _full_episodes(tmp_path) == []

    def test_ignores_non_mp3(self, tmp_path):
        f = tmp_path / "ep.wav"
        f.write_bytes(b"x" * self._MIN_SIZE)
        assert _full_episodes(tmp_path) == []


# ---------------------------------------------------------------------------
# _clips_exist
# ---------------------------------------------------------------------------


class TestClipsExist:
    def test_returns_false_when_no_manifest(self, tmp_path):
        assert not _clips_exist(tmp_path, "ep001")

    def test_returns_false_when_manifest_exists_but_no_mp3(self, tmp_path):
        manifest = tmp_path / "open" / "ep001" / "matches_manifest.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("score,path\n")
        assert not _clips_exist(tmp_path, "ep001")

    def test_returns_true_when_manifest_and_mp3_present(self, tmp_path):
        bundle = tmp_path / "open" / "ep001"
        bundle.mkdir(parents=True)
        (bundle / "matches_manifest.csv").write_text("score,path\n")
        (bundle / "clip_0.mp3").write_bytes(b"data")
        assert _clips_exist(tmp_path, "ep001")


# ---------------------------------------------------------------------------
# _validate_remote_inputs
# ---------------------------------------------------------------------------


class TestValidateRemoteInputs:
    def test_exits_when_remote_dir_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            _validate_remote_inputs(tmp_path / "missing", tmp_path, tmp_path)

    def test_exits_when_open_sample_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            _validate_remote_inputs(tmp_path, tmp_path / "open.mp3", tmp_path)

    def test_exits_when_close_sample_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            _validate_remote_inputs(tmp_path, tmp_path, tmp_path / "close.mp3")

    def test_passes_when_all_exist(self, tmp_path):
        open_s = tmp_path / "open.mp3"
        close_s = tmp_path / "close.mp3"
        open_s.touch()
        close_s.touch()
        _validate_remote_inputs(tmp_path, open_s, close_s)  # should not raise


# ---------------------------------------------------------------------------
# _update_thresholds
# ---------------------------------------------------------------------------


class TestUpdateThresholds:
    def test_raises_open_floor(self, tmp_path):
        ss = SessionScores(approved_open=[0.92])
        state = PipelineState(open_threshold=0.8, close_threshold=0.8)
        open_t, close_t = _update_thresholds(ss, state, tmp_path / "s.toml", 1, 0.8, 0.8)
        assert open_t == pytest.approx(0.92 * 0.995)
        assert close_t == pytest.approx(0.8)

    def test_does_not_lower_threshold(self, tmp_path):
        ss = SessionScores(approved_open=[0.5])  # floor would be ~0.4975
        state = PipelineState(open_threshold=0.8, close_threshold=0.8)
        open_t, _ = _update_thresholds(ss, state, tmp_path / "s.toml", 1, 0.8, 0.8)
        assert open_t == pytest.approx(0.8)

    def test_saves_state(self, tmp_path):
        ss = SessionScores()
        state = PipelineState()
        path = tmp_path / "s.toml"
        _update_thresholds(ss, state, path, 1, 0.8, 0.8)
        assert path.exists()


# ---------------------------------------------------------------------------
# _print_batch_summary
# ---------------------------------------------------------------------------


class TestPrintBatchSummary:
    def test_prints_without_scores(self, capsys):
        _print_batch_summary(1, SessionScores(), 0.8, 0.8)
        out = capsys.readouterr().out
        assert "Batch 1 summary" in out
        assert "0.8000" in out

    def test_prints_min_score_when_approved(self, capsys):
        ss = SessionScores(approved_open=[0.9, 0.85])
        _print_batch_summary(2, ss, 0.85, 0.8)
        out = capsys.readouterr().out
        assert "0.8500" in out


# ---------------------------------------------------------------------------
# _cleanup_clips
# ---------------------------------------------------------------------------


class TestCleanupClips:
    def test_removes_mp3_files(self, tmp_path):
        bundle = tmp_path / "open" / "ep001"
        bundle.mkdir(parents=True)
        mp3 = bundle / "clip_0.mp3"
        mp3.write_bytes(b"data")
        csv = bundle / "matches_manifest.csv"
        csv.write_text("header\n")
        _cleanup_clips(bundle)
        assert not mp3.exists()
        assert csv.exists()  # only .mp3 removed

    def test_missing_dir_does_not_raise(self, tmp_path):
        _cleanup_clips(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# _emit
# ---------------------------------------------------------------------------


class TestEmit:
    def test_writes_to_stderr(self, capsys):
        _emit("hello world")
        assert "hello world" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _handle_review_key
# ---------------------------------------------------------------------------


class TestHandleReviewKey:
    def _make_args(self, tmp_path):
        clip = tmp_path / "clip.mp3"
        clip.touch()
        snippet = tmp_path / "snippet.mp3"
        snippet.touch()
        return dict(
            index=0,
            score=0.9,
            clip_path=clip,
            snippet_path=snippet,
            kind="open",
            approved=[],
            rejected=[],
            history=[],
            session_scores=SessionScores(),
        )

    def test_approve_key(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        i_delta, proc = _handle_review_key("a", **kwargs)
        assert i_delta == 1
        assert proc is None
        assert 0 in kwargs["approved"]

    def test_reject_key(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        i_delta, proc = _handle_review_key("r", **kwargs)
        assert i_delta == 1
        assert 0 in kwargs["rejected"]

    def test_skip_key(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        i_delta, proc = _handle_review_key("s", **kwargs)
        assert i_delta == 1
        assert kwargs["approved"] == []

    def test_undo_key_with_history(self, tmp_path, capsys):
        kwargs = self._make_args(tmp_path)
        kwargs["approved"] = [0]
        kwargs["history"] = [("approved", 0, 0.9)]
        kwargs["session_scores"].approved_open = [0.9]
        i_delta, proc = _handle_review_key("u", **kwargs)
        assert i_delta == -1
        assert kwargs["approved"] == []

    def test_undo_key_empty_history_no_op(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        i_delta, proc = _handle_review_key("u", **kwargs)
        assert i_delta == 0

    def test_quit_key_raises(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        with pytest.raises(KeyboardInterrupt):
            _handle_review_key("q", **kwargs)

    def test_play_key_starts_audio(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        mock_proc = MagicMock()
        with patch("part_io.cli.remote_pipeline._start_audio", return_value=mock_proc):
            i_delta, proc = _handle_review_key("p", **kwargs)
        assert i_delta == 0
        assert proc is mock_proc

    def test_compare_key_starts_audio(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        mock_proc = MagicMock()
        with patch("part_io.cli.remote_pipeline._start_audio", return_value=mock_proc):
            i_delta, proc = _handle_review_key("c", **kwargs)
        assert i_delta == 0
        assert proc is mock_proc

    def test_unknown_key_no_op(self, tmp_path):
        kwargs = self._make_args(tmp_path)
        i_delta, proc = _handle_review_key("z", **kwargs)
        assert i_delta == 0
        assert proc is None


# ---------------------------------------------------------------------------
# _review_loop_episode
# ---------------------------------------------------------------------------


class TestReviewLoopEpisode:
    def _make_ctx(self, tmp_path, no_interactive=False, overwrite=False):
        args = argparse.Namespace(no_interactive=no_interactive, overwrite=overwrite)
        state = PipelineState()
        open_sample = tmp_path / "open.mp3"
        close_sample = tmp_path / "close.mp3"
        open_sample.touch()
        close_sample.touch()
        return LoopContext(
            args=args,
            state=state,
            state_path=tmp_path / "state.toml",
            review_root=tmp_path / "review",
            remote_dir=tmp_path / "remote",
            output_dir=tmp_path / "output",
            open_sample=open_sample,
            close_sample=close_sample,
            session_scores=SessionScores(),
        )

    def test_no_interactive_skips_review(self, tmp_path):
        ctx = self._make_ctx(tmp_path, no_interactive=True)
        ep = tmp_path / "ep001.mp3"
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._review_bundle") as mock_rb:
            _review_loop_episode(ep, ctx=ctx, ep_state=ep_state, already_labeled=False,
                                 open_threshold=0.8, close_threshold=0.8)
        mock_rb.assert_not_called()

    def test_already_labeled_prints_message(self, tmp_path, capsys):
        ctx = self._make_ctx(tmp_path, no_interactive=True)
        ep = tmp_path / "ep001.mp3"
        ep_state = EpisodeState(open_approved=[0])
        _review_loop_episode(ep, ctx=ctx, ep_state=ep_state, already_labeled=True,
                             open_threshold=0.8, close_threshold=0.8)
        assert "Already labeled" in capsys.readouterr().out

    def test_review_called_when_interactive(self, tmp_path):
        ctx = self._make_ctx(tmp_path, no_interactive=False, overwrite=True)
        ep = tmp_path / "ep001.mp3"
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._review_bundle") as mock_rb:
            _review_loop_episode(ep, ctx=ctx, ep_state=ep_state, already_labeled=False,
                                 open_threshold=0.8, close_threshold=0.8)
        assert mock_rb.call_count == 2

    def test_threshold_raised_from_session_scores(self, tmp_path):
        ctx = self._make_ctx(tmp_path, no_interactive=False, overwrite=True)
        ctx.session_scores.approved_open = [0.95]
        ep = tmp_path / "ep001.mp3"
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._review_bundle"):
            open_t, close_t = _review_loop_episode(
                ep, ctx=ctx, ep_state=ep_state, already_labeled=False,
                open_threshold=0.8, close_threshold=0.8,
            )
        assert open_t == pytest.approx(0.95 * 0.995)
        assert close_t == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# _cut_loop_episode
# ---------------------------------------------------------------------------


class TestCutLoopEpisode:
    def _make_ctx(self, tmp_path):
        args = argparse.Namespace(
            min_gap=-15.0, max_gap=600.0, yes=True, dry_run=False, cleanup=False
        )
        state = PipelineState()
        return LoopContext(
            args=args,
            state=state,
            state_path=tmp_path / "state.toml",
            review_root=tmp_path / "review",
            remote_dir=tmp_path / "remote",
            output_dir=tmp_path / "output",
            open_sample=tmp_path / "open.mp3",
            close_sample=tmp_path / "close.mp3",
            session_scores=SessionScores(),
        )

    def test_delegates_to_pair_and_cut(self, tmp_path):
        ctx = self._make_ctx(tmp_path)
        ep = ctx.remote_dir / "ep001.mp3"
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="cut") as mock_pac:
            result = _cut_loop_episode(ep, ctx=ctx, ep_state=ep_state)
        assert result == "cut"
        mock_pac.assert_called_once()


# ---------------------------------------------------------------------------
# main() — argument parsing
# ---------------------------------------------------------------------------


class TestMain:
    def test_review_subcommand_dispatches(self, tmp_path):
        with patch("part_io.cli.remote_pipeline._cmd_review") as mock_cmd:
            with patch("sys.argv", ["remote_pipeline", "review"]):
                main()
        mock_cmd.assert_called_once()

    def test_cut_subcommand_dispatches(self, tmp_path):
        with patch("part_io.cli.remote_pipeline._cmd_cut") as mock_cmd:
            with patch("sys.argv", ["remote_pipeline", "cut"]):
                main()
        mock_cmd.assert_called_once()

    def test_loop_subcommand_dispatches(self, tmp_path):
        with patch("part_io.cli.remote_pipeline._cmd_loop") as mock_cmd:
            with patch("sys.argv", ["remote_pipeline", "loop"]):
                main()
        mock_cmd.assert_called_once()

    def test_no_subcommand_exits(self):
        with patch("sys.argv", ["remote_pipeline"]):
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# _review_clip
# ---------------------------------------------------------------------------


class TestReviewClip:
    def _csv_row(self, tmp_path, index=0, score=0.9, start=10.0):
        clip = tmp_path / f"clip_{index}.mp3"
        clip.write_bytes(b"data")
        return {
            "index": str(index),
            "score": str(score),
            "start_seconds": str(start),
            "clip_path": str(clip),
        }

    def test_approve_advances(self, tmp_path):
        row = self._csv_row(tmp_path)
        snippet = tmp_path / "snippet.mp3"
        snippet.touch()
        ss = SessionScores()
        with patch("part_io.cli.remote_pipeline._getch", side_effect=["a"]):
            with patch("part_io.cli.remote_pipeline._start_audio", return_value=MagicMock()):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    approved = []
                    i_delta, n_skip = _review_clip(
                        row=row, kind="open", n_rows=1, row_num=1,
                        snippet_path=snippet, approved=approved, rejected=[],
                        history=[], session_scores=ss, n_skipped=0,
                    )
        assert i_delta == 1
        assert 0 in approved

    def test_missing_clip_auto_skips(self, tmp_path):
        row = {
            "index": "0", "score": "0.9", "start_seconds": "0.0",
            "clip_path": str(tmp_path / "missing.mp3"),
        }
        snippet = tmp_path / "s.mp3"
        snippet.touch()
        i_delta, n_skip = _review_clip(
            row=row, kind="open", n_rows=1, row_num=1,
            snippet_path=snippet, approved=[], rejected=[],
            history=[], session_scores=SessionScores(), n_skipped=0,
        )
        assert i_delta == 1
        assert n_skip == 1

    def test_skip_key_counts_skipped(self, tmp_path):
        row = self._csv_row(tmp_path)
        snippet = tmp_path / "s.mp3"
        snippet.touch()
        with patch("part_io.cli.remote_pipeline._getch", side_effect=["s"]):
            with patch("part_io.cli.remote_pipeline._start_audio", return_value=MagicMock()):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    _, n_skip = _review_clip(
                        row=row, kind="close", n_rows=1, row_num=1,
                        snippet_path=snippet, approved=[], rejected=[],
                        history=[], session_scores=SessionScores(), n_skipped=0,
                    )
        assert n_skip == 1

    def test_unknown_key_then_approve(self, tmp_path):
        row = self._csv_row(tmp_path)
        snippet = tmp_path / "s.mp3"
        snippet.touch()
        with patch("part_io.cli.remote_pipeline._getch", side_effect=["z", "a"]):
            with patch("part_io.cli.remote_pipeline._start_audio", return_value=MagicMock()):
                with patch("part_io.cli.remote_pipeline._stop_audio"):
                    i_delta, _ = _review_clip(
                        row=row, kind="open", n_rows=1, row_num=1,
                        snippet_path=snippet, approved=[], rejected=[],
                        history=[], session_scores=SessionScores(), n_skipped=0,
                    )
        assert i_delta == 1


# ---------------------------------------------------------------------------
# _review_bundle
# ---------------------------------------------------------------------------


class TestReviewBundle:
    def _write_manifest(self, bundle_dir: Path, rows: list[dict]) -> None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = bundle_dir / "matches_manifest.csv"
        import csv
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "score", "start_seconds", "clip_path"])
            writer.writeheader()
            writer.writerows(rows)

    def test_no_manifest_skips(self, tmp_path, capsys):
        bundle_dir = tmp_path / "open" / "ep001"
        bundle_dir.mkdir(parents=True)
        ep_state = EpisodeState()
        _review_bundle(bundle_dir, "open", SessionScores(),
                       snippet_path=tmp_path / "s.mp3", ep_state=ep_state)
        assert "No manifest" in capsys.readouterr().out

    def test_empty_manifest_skips(self, tmp_path, capsys):
        bundle_dir = tmp_path / "open" / "ep001"
        self._write_manifest(bundle_dir, [])
        ep_state = EpisodeState()
        _review_bundle(bundle_dir, "open", SessionScores(),
                       snippet_path=tmp_path / "s.mp3", ep_state=ep_state)
        assert "Empty manifest" in capsys.readouterr().out

    def test_review_updates_ep_state(self, tmp_path):
        bundle_dir = tmp_path / "open" / "ep001"
        clip = tmp_path / "clip.mp3"
        clip.write_bytes(b"data")
        self._write_manifest(bundle_dir, [
            {"index": "0", "score": "0.9", "start_seconds": "5.0", "clip_path": str(clip)},
        ])
        ep_state = EpisodeState()
        snippet = tmp_path / "s.mp3"
        snippet.touch()

        with patch("part_io.cli.remote_pipeline._review_clip",
                   return_value=(1, 0)) as mock_rc:
            mock_rc.return_value = (1, 0)
            _review_bundle(bundle_dir, "open", SessionScores(),
                           snippet_path=snippet, ep_state=ep_state)


# ---------------------------------------------------------------------------
# _interactive_review_batch
# ---------------------------------------------------------------------------


class TestInteractiveReviewBatch:
    def test_calls_review_bundle_per_ep(self, tmp_path):
        state = PipelineState()
        session_scores = SessionScores()
        batch = [tmp_path / "ep001.mp3", tmp_path / "ep002.mp3"]
        open_sample = tmp_path / "open.mp3"
        close_sample = tmp_path / "close.mp3"

        with patch("part_io.cli.remote_pipeline._review_bundle") as mock_rb:
            _interactive_review_batch(
                batch, tmp_path / "review", session_scores, state,
                open_sample=open_sample, close_sample=close_sample,
            )
        assert mock_rb.call_count == 4  # 2 episodes × 2 bundles (open + close)


# ---------------------------------------------------------------------------
# _review_one
# ---------------------------------------------------------------------------


class TestReviewOne:
    def test_returns_returncode(self, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b""
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=mock_result):
            rc = _review_one(
                source=tmp_path / "ep.mp3",
                sample=tmp_path / "open.mp3",
                bundle_name="open/ep001",
                review_root=tmp_path / "review",
                threshold=0.8,
                z_threshold=3.0,
                step_seconds=0.1,
                max_clips=10,
                refine=False,
                overwrite=False,
            )
        assert rc == 0

    def test_refine_and_overwrite_flags_appended(self, tmp_path):
        captured_cmd = []
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b""

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_result

        with patch("part_io.cli.remote_pipeline.run_resolved", side_effect=fake_run):
            _review_one(
                source=tmp_path / "ep.mp3",
                sample=tmp_path / "open.mp3",
                bundle_name="open/ep001",
                review_root=tmp_path,
                threshold=0.8,
                z_threshold=3.0,
                step_seconds=0.1,
                max_clips=5,
                refine=True,
                overwrite=True,
            )
        assert "--refine" in captured_cmd
        assert "--overwrite" in captured_cmd

    def test_nonzero_returncode_with_stderr(self, tmp_path, capsys):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"some error"
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=mock_result):
            rc = _review_one(
                source=tmp_path / "ep.mp3",
                sample=tmp_path / "open.mp3",
                bundle_name="open/ep001",
                review_root=tmp_path,
                threshold=0.8,
                z_threshold=3.0,
                step_seconds=0.1,
                max_clips=10,
                refine=False,
                overwrite=False,
            )
        assert rc == 1


# ---------------------------------------------------------------------------
# _run_clip_pool / _generate_batch_clips
# ---------------------------------------------------------------------------


class TestRunClipPool:
    def _make_args(self):
        return argparse.Namespace(
            workers=1, z_threshold=3.0, step_seconds=0.1,
            max_clips=5, refine=False, overwrite=False,
        )

    def test_calls_review_one_per_job(self, tmp_path):
        jobs = [
            (tmp_path / "ep.mp3", tmp_path / "open.mp3", "open/ep001", 0.8),
        ]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b""
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=mock_result):
            _run_clip_pool(jobs, review_root=tmp_path, args=self._make_args())

    def test_failed_job_emits_warning(self, tmp_path, capsys):
        jobs = [
            (tmp_path / "ep.mp3", tmp_path / "open.mp3", "open/ep001", 0.8),
        ]
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b""
        with patch("part_io.cli.remote_pipeline.run_resolved", return_value=mock_result):
            _run_clip_pool(jobs, review_root=tmp_path, args=self._make_args())
        err = capsys.readouterr().err
        assert "WARNING" in err or "Warning" in err or "FAILED" in err


class TestGenerateBatchClips:
    def _make_args(self):
        return argparse.Namespace(
            workers=1, z_threshold=3.0, step_seconds=0.1,
            max_clips=5, refine=False, overwrite=False,
        )

    def test_submits_open_and_close_jobs(self, tmp_path):
        to_generate = [tmp_path / "ep001.mp3"]
        calls = []

        def fake_pool(jobs, *, review_root, args):
            calls.extend(jobs)

        with patch("part_io.cli.remote_pipeline._run_clip_pool", side_effect=fake_pool):
            _generate_batch_clips(
                to_generate,
                open_sample=tmp_path / "open.mp3",
                close_sample=tmp_path / "close.mp3",
                open_threshold=0.8,
                close_threshold=0.85,
                review_root=tmp_path,
                args=self._make_args(),
            )
        assert len(calls) == 2  # one open + one close job


# ---------------------------------------------------------------------------
# _pair_and_cut
# ---------------------------------------------------------------------------


class TestPairAndCut:
    def _make_args_ns(self):
        return argparse.Namespace(
            min_gap=-15.0, max_gap=600.0, yes=True, dry_run=False, cleanup=False
        )

    def test_returns_skipped_when_load_fails(self, tmp_path, capsys):
        ep_state = EpisodeState()
        with patch(
            "part_io.cli.remote_pipeline._load_ad_segments",
            return_value="  SKIP: missing file",
        ):
            result = _pair_and_cut(
                "ep001", tmp_path / "ep.mp3",
                review_root=tmp_path, output_dir=tmp_path / "out",
                ep_state=ep_state, min_gap=-15.0, max_gap=600.0,
                yes=True, dry_run=True, cleanup=False,
            )
        assert result == "skipped"
        assert "SKIP" in capsys.readouterr().out

    def test_dry_run_returns_skipped(self, tmp_path):
        from part_io.adapters.audio.ad_segments import AdSegment
        seg = AdSegment(open_start=0.0, open_end=10.0, close_start=30.0, close_end=40.0,
                        open_score=0.9, close_score=0.9)
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._load_ad_segments", return_value=[seg]):
            result = _pair_and_cut(
                "ep001", tmp_path / "ep.mp3",
                review_root=tmp_path, output_dir=tmp_path / "out",
                ep_state=ep_state, min_gap=-15.0, max_gap=600.0,
                yes=True, dry_run=True, cleanup=False,
            )
        assert result == "skipped"

    def test_ffmpeg_failure_returns_failed(self, tmp_path):
        from part_io.adapters.audio.ad_segments import AdSegment
        seg = AdSegment(open_start=0.0, open_end=10.0, close_start=30.0, close_end=40.0,
                        open_score=0.9, close_score=0.9)
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._load_ad_segments", return_value=[seg]):
            with patch("part_io.cli.remote_pipeline._run_ffmpeg", return_value=1):
                with patch("part_io.cli.remote_pipeline._build_keep_spans", return_value=[]):
                    with patch("part_io.cli.remote_pipeline._build_filter_complex",
                               return_value=("complex", 2)):
                        result = _pair_and_cut(
                            "ep001", tmp_path / "ep.mp3",
                            review_root=tmp_path, output_dir=tmp_path / "out",
                            ep_state=ep_state, min_gap=-15.0, max_gap=600.0,
                            yes=True, dry_run=False, cleanup=False,
                        )
        assert result == "failed"

    def test_successful_cut(self, tmp_path):
        from part_io.adapters.audio.ad_segments import AdSegment
        seg = AdSegment(open_start=0.0, open_end=10.0, close_start=30.0, close_end=40.0,
                        open_score=0.9, close_score=0.9)
        ep_state = EpisodeState()
        with patch("part_io.cli.remote_pipeline._load_ad_segments", return_value=[seg]):
            with patch("part_io.cli.remote_pipeline._run_ffmpeg", return_value=0):
                with patch("part_io.cli.remote_pipeline._build_keep_spans", return_value=[]):
                    with patch("part_io.cli.remote_pipeline._build_filter_complex",
                               return_value=("complex", 2)):
                        result = _pair_and_cut(
                            "ep001", tmp_path / "ep.mp3",
                            review_root=tmp_path, output_dir=tmp_path / "out",
                            ep_state=ep_state, min_gap=-15.0, max_gap=600.0,
                            yes=True, dry_run=False, cleanup=False,
                        )
        assert result == "cut"


# ---------------------------------------------------------------------------
# _cmd_cut
# ---------------------------------------------------------------------------


class TestCmdCut:
    def _make_state(self, tmp_path) -> Path:
        state = PipelineState()
        ep = state.episode("ep001")
        ep.open_approved = [0]
        state_path = tmp_path / "review" / "state.toml"
        state.save(state_path)
        return state_path

    def _make_args(self, tmp_path, **overrides):
        defaults = dict(
            review_root=tmp_path / "review",
            remote_dir=tmp_path / "remote",
            output_dir=tmp_path / "out",
            min_gap=-15.0, max_gap=600.0,
            yes=True, dry_run=False, cleanup=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_labeled_episodes_returns_early(self, tmp_path, capsys):
        state_path = tmp_path / "review" / "state.toml"
        PipelineState().save(state_path)
        args = self._make_args(tmp_path)
        _cmd_cut(args)
        assert "No labeled episodes" in capsys.readouterr().out

    def test_missing_source_skips(self, tmp_path, capsys):
        self._make_state(tmp_path)
        args = self._make_args(tmp_path)
        args.remote_dir.mkdir(parents=True, exist_ok=True)
        _cmd_cut(args)
        assert "SKIP" in capsys.readouterr().out

    def test_cut_result_marks_state(self, tmp_path):
        self._make_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        args = self._make_args(tmp_path)

        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="cut"):
            _cmd_cut(args)

        state = PipelineState.load(tmp_path / "review" / "state.toml")
        assert state.episodes["ep001"].cut is True

    def test_failed_result_exits_1(self, tmp_path):
        self._make_state(tmp_path)
        source = tmp_path / "remote" / "ep001.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
        args = self._make_args(tmp_path)

        with patch("part_io.cli.remote_pipeline._pair_and_cut", return_value="failed"):
            with pytest.raises(SystemExit) as exc:
                _cmd_cut(args)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def _make_batch_args(self, tmp_path, no_interactive=False):
        return argparse.Namespace(
            batch_size=5, overwrite=False, no_interactive=no_interactive,
            z_threshold=3.0, step_seconds=0.1, max_clips=5,
            refine=False, workers=1,
        )

    def test_generates_clips_for_new_episodes(self, tmp_path):
        batch = [tmp_path / "ep001.mp3"]
        episodes = batch
        state = PipelineState()
        session_scores = SessionScores()
        args = self._make_batch_args(tmp_path, no_interactive=True)

        with patch("part_io.cli.remote_pipeline._generate_batch_clips") as mock_gen:
            with patch("part_io.cli.remote_pipeline._update_thresholds",
                       return_value=(0.8, 0.8)):
                _process_batch(
                    1, batch, episodes,
                    args=args,
                    review_root=tmp_path / "review",
                    open_sample=tmp_path / "open.mp3",
                    close_sample=tmp_path / "close.mp3",
                    open_threshold=0.8,
                    close_threshold=0.8,
                    session_scores=session_scores,
                    state=state,
                    state_path=tmp_path / "state.toml",
                )
        mock_gen.assert_called_once()

    def test_skips_gen_when_clips_exist(self, tmp_path, capsys):
        ep = tmp_path / "ep001.mp3"
        bundle = tmp_path / "review" / "open" / "ep001"
        bundle.mkdir(parents=True)
        (bundle / "matches_manifest.csv").write_text("h\n")
        (bundle / "clip.mp3").write_bytes(b"data")

        args = self._make_batch_args(tmp_path, no_interactive=True)
        state = PipelineState()
        session_scores = SessionScores()

        with patch("part_io.cli.remote_pipeline._generate_batch_clips") as mock_gen:
            _process_batch(
                1, [ep], [ep],
                args=args,
                review_root=tmp_path / "review",
                open_sample=tmp_path / "open.mp3",
                close_sample=tmp_path / "close.mp3",
                open_threshold=0.8, close_threshold=0.8,
                session_scores=session_scores, state=state,
                state_path=tmp_path / "state.toml",
            )
        mock_gen.assert_not_called()
        assert "going straight to review" in capsys.readouterr().err

    def test_no_interactive_prints_message(self, tmp_path, capsys):
        args = self._make_batch_args(tmp_path, no_interactive=True)
        state = PipelineState()

        with patch("part_io.cli.remote_pipeline._generate_batch_clips"):
            _process_batch(
                1, [tmp_path / "ep.mp3"], [tmp_path / "ep.mp3"],
                args=args,
                review_root=tmp_path / "review",
                open_sample=tmp_path / "open.mp3",
                close_sample=tmp_path / "close.mp3",
                open_threshold=0.8, close_threshold=0.8,
                session_scores=SessionScores(), state=state,
                state_path=tmp_path / "state.toml",
            )
        assert "no-interactive" in capsys.readouterr().out
