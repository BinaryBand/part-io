"""Tests for audio clip playback and extraction adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from partio.adapters.audio import clips
from partio.adapters.audio.clips import (
    audio_duration_seconds,
    extract_audio_clip,
    play_audio_segment,
)


class _MockCompletedProcess:
    """Simulate a subprocess result with a given returncode."""

    def __init__(self, returncode: int, stdout: bytes | str = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_play_audio_segment_builds_correct_ffplay_command(monkeypatch) -> None:
    """play_audio_segment should invoke ffplay with correct positional args."""
    captured: list[list[str]] = []

    def _mock_run_resolved(cmd: list[str], **_kwargs) -> _MockCompletedProcess:
        captured.append(cmd)
        return _MockCompletedProcess(0)

    monkeypatch.setattr(clips, "run_resolved", _mock_run_resolved)

    play_audio_segment(
        source_path=Path("/media/episode.mp3"),
        start_seconds=42.5,
        duration_seconds=3.0,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "ffplay"
    assert "-ss" in cmd
    ss_index = cmd.index("-ss")
    assert cmd[ss_index + 1] == "42.500"
    assert "-t" in cmd
    t_index = cmd.index("-t")
    assert cmd[t_index + 1] == "3.000"
    assert str(Path("/media/episode.mp3")) in cmd


def test_extract_audio_clip_builds_correct_ffmpeg_command(monkeypatch) -> None:
    """extract_audio_clip should invoke ffmpeg with correct params."""
    captured: list[list[str]] = []

    def _mock_run_resolved(cmd: list[str], **_kwargs) -> _MockCompletedProcess:
        captured.append(cmd)
        return _MockCompletedProcess(0)

    monkeypatch.setattr(clips, "run_resolved", _mock_run_resolved)

    extract_audio_clip(
        source_path=Path("/media/episode.mp3"),
        destination_path=Path("/out/clip001.mp3"),
        start_seconds=10.0,
        duration_seconds=5.5,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd
    i_index = cmd.index("-i")
    assert str(Path("/media/episode.mp3")) == cmd[i_index + 1]
    assert "-ss" in cmd
    ss_index = cmd.index("-ss")
    assert cmd[ss_index + 1] == "10.000"
    assert "-t" in cmd
    t_index = cmd.index("-t")
    assert cmd[t_index + 1] == "5.500"
    assert str(Path("/out/clip001.mp3")) in cmd
    assert "libmp3lame" in cmd


def test_play_audio_segment_raises_on_nonzero_returncode(monkeypatch) -> None:
    """A non-zero ffplay returncode should raise ValueError."""

    def _mock_run_resolved(*_args, **_kwargs) -> _MockCompletedProcess:
        return _MockCompletedProcess(1)

    monkeypatch.setattr(clips, "run_resolved", _mock_run_resolved)

    with pytest.raises(ValueError, match="ffplay failed to play segment"):
        play_audio_segment(
            source_path=Path("/media/episode.mp3"),
            start_seconds=0.0,
            duration_seconds=1.0,
        )


def test_audio_duration_seconds_parses_ffprobe_output(monkeypatch) -> None:
    """The probed duration comes back as a float."""
    captured: list[list[str]] = []

    def _mock_run_resolved(cmd: list[str], **_kwargs) -> _MockCompletedProcess:
        captured.append(cmd)
        return _MockCompletedProcess(0, "3612.480000\n")

    monkeypatch.setattr(clips, "run_resolved", _mock_run_resolved)

    assert audio_duration_seconds(Path("/media/episode.mp3")) == pytest.approx(3612.48)
    assert captured[0][0] == "ffprobe"
    assert "format=duration" in captured[0]


@pytest.mark.parametrize(
    ("returncode", "stdout", "match"),
    [
        (1, "", "ffprobe failed to read duration"),
        (0, "N/A\n", "ffprobe reported no duration"),
        (0, "0.000000\n", "ffprobe reported no duration"),
    ],
)
def test_audio_duration_seconds_raises_when_unreadable(
    monkeypatch, returncode: int, stdout: str, match: str
) -> None:
    """A failed probe or a container with no duration is a ValueError."""

    def _mock_run_resolved(*_args, **_kwargs) -> _MockCompletedProcess:
        return _MockCompletedProcess(returncode, stdout)

    monkeypatch.setattr(clips, "run_resolved", _mock_run_resolved)

    with pytest.raises(ValueError, match=match):
        audio_duration_seconds(Path("/media/episode.mp3"))


def test_extract_audio_clip_raises_on_nonzero_returncode(monkeypatch) -> None:
    """A non-zero ffmpeg returncode should raise ValueError."""

    def _mock_run_resolved(*_args, **_kwargs) -> _MockCompletedProcess:
        return _MockCompletedProcess(1)

    monkeypatch.setattr(clips, "run_resolved", _mock_run_resolved)

    with pytest.raises(ValueError, match="ffmpeg failed to write clip"):
        extract_audio_clip(
            source_path=Path("/media/episode.mp3"),
            destination_path=Path("/out/clip.mp3"),
            start_seconds=0.0,
            duration_seconds=1.0,
        )
