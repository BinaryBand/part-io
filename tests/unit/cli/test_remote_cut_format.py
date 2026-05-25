from __future__ import annotations

from pathlib import Path

from part_io.adapters.audio.ad_segments import AdSegment
from part_io.cli import audio_ad_remove as aar
from part_io.cli.remote import _cut as cutmod
from part_io.cli.remote._cut import CutSettings
from part_io.cli.remote._state import PipelineState, _Match


def test_pair_and_cut_preserves_source_extension(tmp_path: Path, monkeypatch) -> None:
    remote_dir = tmp_path / "downloads" / "remote"
    remote_dir.mkdir(parents=True)
    source = remote_dir / "ep1.opus"
    source.write_bytes(b"opus")

    state = PipelineState()
    ep = state.episode("ep1")
    # mark open and close as positive
    ep.open_candidates = [_Match(score=0.9, start=1.0, end=2.0, label="positive")]
    ep.close_candidates = [_Match(score=0.9, start=10.0, end=11.0, label="positive")]

    # ensure find_best_pair returns a simple AdSegment
    seg = AdSegment(
        open_start=1.0,
        open_end=2.0,
        close_start=10.0,
        close_end=11.0,
        open_score=0.9,
        close_score=0.9,
    )
    monkeypatch.setattr(cutmod, "_find_best_pair", lambda *args, **kwargs: [seg])

    recorded: dict[str, Path] = {}

    def fake_exec(_source, filter_complex, output_path):
        recorded["output"] = output_path
        # simulate writing file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x")
        return True

    monkeypatch.setattr(cutmod, "_execute_ffmpeg_cut", fake_exec)

    settings = CutSettings(min_gap=0.0, max_gap=600.0, yes=True, dry_run=False)
    result = cutmod._pair_and_cut(
        "ep1",
        source,
        output_dir=tmp_path / "staging",
        ep_state=ep,
        settings=settings,
    )

    assert result == "cut"
    assert recorded["output"].suffix == ".opus"


def test_run_ffmpeg_chooses_codec_by_extension(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "in.mp3"
    src.write_bytes(b"x")
    captured: dict[str, list[str]] = {}

    class FakeProc:
        returncode = 0

    def fake_run_resolved(cmd, *a, **k):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(aar, "run_resolved", fake_run_resolved)

    out_opus = tmp_path / "out.opus"
    aar._run_ffmpeg(src, "[0:a]atrim=0:1[out]", out_opus)
    joined = " ".join(map(str, captured["cmd"]))
    assert "libopus" in joined
