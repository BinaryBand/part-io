"""Tests for remote pipeline config-init: snippet profiles embedded in __state__.toml."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import pytest

from part_io.adapters.audio.snippet_profile import SnippetProfileModel
from part_io.cli import remote_pipeline as rp
from part_io.cli.remote._state import PipelineState, SnippetEntry


def _fake_profile(n_frames: int = 2, band_count: int = 1) -> np.ndarray:
    return np.zeros((n_frames, band_count * 2), dtype=np.float32)


def _make_fake_snapshot(n_frames: int = 2, band_count: int = 1):
    def fake_snapshot(_path: Path) -> SnippetProfileModel:
        return SnippetProfileModel(
            source_hash="abc123",
            n_frames=n_frames,
            analysis_rate=16000,
            hop_size=1024,
            band_count=band_count,
            data="ignored",
        )

    return fake_snapshot


def test_config_init_writes_snippets_to_state(tmp_path: Path, monkeypatch) -> None:
    open_seed = tmp_path / "open.mp3"
    close_seed = tmp_path / "close.mp3"
    open_seed.write_bytes(b"open")
    close_seed.write_bytes(b"close")

    monkeypatch.setattr(rp, "snapshot_snippet_profile", _make_fake_snapshot())
    monkeypatch.setattr(rp, "decode_matrix", lambda data, n_frames, band_count: _fake_profile())

    args = argparse.Namespace(
        remote_dir=tmp_path,
        open_seed=open_seed,
        close_seed=close_seed,
        intro_seed=None,
        outro_seed=None,
        force=False,
    )

    rp._cmd_config_init(args)

    state_path = tmp_path / "__state__.toml"
    assert state_path.exists()
    parsed = tomllib.loads(state_path.read_text(encoding="utf-8"))
    assert len(parsed["snippets"]) == 2
    names = {s["name"] for s in parsed["snippets"]}
    assert names == {"open", "close"}


def test_config_init_exits_when_snippets_already_present(tmp_path: Path, monkeypatch) -> None:
    state = PipelineState()
    state.snippets.append(SnippetEntry(name="open", profile=_fake_profile()))
    state_path = tmp_path / "__state__.toml"
    state.save(state_path)

    open_seed = tmp_path / "open.mp3"
    open_seed.write_bytes(b"x")
    close_seed = tmp_path / "close.mp3"
    close_seed.write_bytes(b"x")

    args = argparse.Namespace(
        remote_dir=tmp_path,
        open_seed=open_seed,
        close_seed=close_seed,
        intro_seed=None,
        outro_seed=None,
        force=False,
    )

    with pytest.raises(SystemExit):
        rp._cmd_config_init(args)


def test_config_init_force_overwrites_existing(tmp_path: Path, monkeypatch) -> None:
    state = PipelineState()
    state.snippets.append(SnippetEntry(name="open", profile=_fake_profile()))
    state_path = tmp_path / "__state__.toml"
    state.save(state_path)

    open_seed = tmp_path / "open.mp3"
    close_seed = tmp_path / "close.mp3"
    open_seed.write_bytes(b"open")
    close_seed.write_bytes(b"close")

    monkeypatch.setattr(rp, "snapshot_snippet_profile", _make_fake_snapshot())
    monkeypatch.setattr(rp, "decode_matrix", lambda data, n_frames, band_count: _fake_profile())

    args = argparse.Namespace(
        remote_dir=tmp_path,
        open_seed=open_seed,
        close_seed=close_seed,
        intro_seed=None,
        outro_seed=None,
        force=True,
    )

    rp._cmd_config_init(args)  # must not raise

    state2 = PipelineState.load(state_path)
    assert {s.name for s in state2.snippets} == {"open", "close"}


def test_config_init_exits_when_seed_file_missing(tmp_path: Path) -> None:
    args = argparse.Namespace(
        remote_dir=tmp_path,
        open_seed=tmp_path / "nonexistent.mp3",
        close_seed=tmp_path / "also_missing.mp3",
        intro_seed=None,
        outro_seed=None,
        force=False,
    )

    with pytest.raises(SystemExit):
        rp._cmd_config_init(args)
