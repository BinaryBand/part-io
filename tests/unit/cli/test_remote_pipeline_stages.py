from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from part_io.cli import remote_pipeline as rp
from part_io.cli.remote._state import PipelineState, _Match


def test_cmd_precache_exits_cleanly_when_remote_dir_missing(tmp_path: Path) -> None:
    args = argparse.Namespace(
        remote_dir=tmp_path / "missing",
        verbose=False,
        sleep=0.0,
        overwrite=False,
        background=False,
    )

    with pytest.raises(SystemExit, match="Remote dir not found"):
        rp._cmd_precache(args)


def test_launch_background_job_creates_pid_and_log_parent(tmp_path: Path, monkeypatch) -> None:
    class _Proc:
        pid = 4242

    monkeypatch.setattr("part_io.utils.exec.launch_resolved", lambda *args, **kwargs: _Proc())

    remote_dir = tmp_path / "downloads" / "remote"
    rp._launch_background_job(
        remote_dir=remote_dir,
        job_name="prep-quiz",
        cmd=["python", "-m", "part_io.cli.remote_pipeline", "prep-quiz"],
    )

    assert (tmp_path / "downloads" / ".prep_quiz.pid").exists()
    assert (tmp_path / "downloads" / ".prep_quiz.log").exists()


def test_cmd_prep_cut_exits_when_state_has_no_episodes(tmp_path: Path) -> None:
    remote_dir = tmp_path / "downloads" / "remote"
    remote_dir.mkdir(parents=True)

    args = argparse.Namespace(
        remote_dir=remote_dir,
        verbose=False,
        open_seed=None,
        close_seed=None,
        intro_seed=None,
        outro_seed=None,
    )

    with pytest.raises(SystemExit, match="No detection data"):
        rp._cmd_prep_cut(args)


def test_cmd_prep_cut_exits_when_prep_quiz_still_running(tmp_path: Path, monkeypatch) -> None:
    remote_dir = tmp_path / "downloads" / "remote"
    remote_dir.mkdir(parents=True)

    # Plant a pid file that looks live by monkeypatching _background_running.
    monkeypatch.setattr(rp, "_background_running", lambda _path: 9999)

    args = argparse.Namespace(
        remote_dir=remote_dir,
        verbose=False,
        open_seed=None,
        close_seed=None,
        intro_seed=None,
        outro_seed=None,
    )

    with pytest.raises(SystemExit, match="still running"):
        rp._cmd_prep_cut(args)


def test_cmd_prep_cut_uses_default_snippets_near_remote_dir(tmp_path: Path, monkeypatch) -> None:
    remote_dir = tmp_path / "downloads" / "remote"
    remote_dir.mkdir(parents=True)

    snippets_dir = remote_dir.parent / "snippets"
    snippets_dir.mkdir(parents=True)
    (snippets_dir / "open.mp3").write_bytes(b"open")
    (snippets_dir / "close.mp3").write_bytes(b"close")

    state = PipelineState()
    ep = state.episode("ep001")
    ep.open_candidates = [_Match(score=0.9, start=1.0, end=2.0)]
    state.save(remote_dir / "__state__.toml")

    captured: dict[str, dict[str, Path]] = {}

    def fake_run_review_loop(
        state: PipelineState,
        *,
        snippets: dict[str, Path],
        state_path: Path,
        remote_dir: Path,
        max_decisions: int | None = None,
    ) -> None:
        _ = state
        _ = state_path
        _ = remote_dir
        _ = max_decisions
        captured["snippets"] = snippets

    monkeypatch.setattr(rp, "_run_review_loop", fake_run_review_loop)

    args = argparse.Namespace(
        remote_dir=remote_dir,
        verbose=False,
        open_seed=None,
        close_seed=None,
        intro_seed=None,
        outro_seed=None,
    )
    rp._cmd_prep_cut(args)

    assert captured["snippets"]["open"] == snippets_dir / "open.mp3"
    assert captured["snippets"]["close"] == snippets_dir / "close.mp3"
