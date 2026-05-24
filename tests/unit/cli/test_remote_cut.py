from __future__ import annotations

from pathlib import Path

from part_io.cli.remote._cut import CutSettings, _cut_cuttable
from part_io.cli.remote._state import EpisodeState, PipelineState, _Match


def test_cut_cuttable_skips_when_source_missing(tmp_path: Path) -> None:
    state = PipelineState()
    state.episodes["ep001"] = EpisodeState(
        candidates={
            "open": [_Match(score=0.9, start=10.0, end=20.0, label="positive")],
            "close": [_Match(score=0.9, start=40.0, end=50.0, label="positive")],
            "intro": [],
            "outro": [],
        }
    )

    n_cut, n_skipped, n_failed = _cut_cuttable(
        state,
        remote_dir=tmp_path / "remote",
        output_dir=tmp_path / "out",
        settings=CutSettings(min_gap=-15.0, max_gap=300.0, yes=True, dry_run=True),
        state_path=tmp_path / "state.toml",
    )

    assert (n_cut, n_skipped, n_failed) == (0, 1, 0)


def test_cut_cuttable_marks_episode_cut_and_saves(tmp_path: Path, monkeypatch) -> None:
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir(parents=True)
    (remote_dir / "ep002.mp3").write_bytes(b"x")

    state = PipelineState()
    state.episodes["ep002"] = EpisodeState(
        candidates={
            "open": [_Match(score=0.9, start=10.0, end=20.0, label="positive")],
            "close": [_Match(score=0.9, start=40.0, end=50.0, label="positive")],
            "intro": [],
            "outro": [],
        }
    )

    save_calls: dict[str, int] = {"count": 0}

    def fake_save(path: Path) -> None:
        _ = path
        save_calls["count"] += 1

    monkeypatch.setattr(state, "save", fake_save)

    def fake_pair_and_cut(*_args, **_kwargs) -> str:
        return "cut"

    monkeypatch.setattr("part_io.cli.remote._cut._pair_and_cut", fake_pair_and_cut)

    n_cut, n_skipped, n_failed = _cut_cuttable(
        state,
        remote_dir=remote_dir,
        output_dir=tmp_path / "out",
        settings=CutSettings(min_gap=-15.0, max_gap=300.0, yes=True, dry_run=False),
        state_path=tmp_path / "state.toml",
    )

    assert (n_cut, n_skipped, n_failed) == (1, 0, 0)
    assert state.episodes["ep002"].cut is True
    assert save_calls["count"] == 1
