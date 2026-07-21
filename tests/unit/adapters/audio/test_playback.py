"""Tests for non-blocking ffplay playback handles."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from partio.adapters.audio import playback as playback_module
from partio.adapters.audio.playback import PlaybackHandle, start_audio_segment

STARTED_AT = 100.0


def _process(*, poll=None) -> MagicMock:
    """A stand-in ffplay process; *poll* is None while running, an int once exited."""
    process = MagicMock()
    process.poll.return_value = poll
    return process


def _handle(process=None, *, poll=None, duration=10.0) -> PlaybackHandle:
    """A handle for a 30.0s-in segment that started at monotonic time 100.0.

    Tests set the "current" time by patching ``time.monotonic``, so elapsed
    time is ``monotonic - STARTED_AT``.
    """
    return PlaybackHandle(
        process if process is not None else _process(poll=poll),
        start_seconds=30.0,
        duration_seconds=duration,
        started_at=STARTED_AT,
    )


# -- playhead ----------------------------------------------------------------


def test_position_tracks_elapsed_time() -> None:
    """The playhead advances with wall-clock time from the segment start."""
    handle = _handle()
    with patch.object(playback_module.time, "monotonic", return_value=STARTED_AT + 4.0):
        assert handle.position_seconds == 34.0


def test_position_clamps_at_the_segment_end() -> None:
    """Overrunning the segment does not push the playhead past its end."""
    handle = _handle(duration=10.0)
    with patch.object(playback_module.time, "monotonic", return_value=999.0):
        assert handle.position_seconds == 40.0


def test_is_playing_false_once_the_duration_elapses() -> None:
    """A segment stops counting as playing after its duration."""
    handle = _handle(duration=10.0)
    with patch.object(playback_module.time, "monotonic", return_value=115.0):
        assert handle.is_playing() is False


def test_is_playing_false_when_the_process_exited() -> None:
    """An exited ffplay is not playing even inside the nominal duration."""
    handle = _handle(poll=0, duration=10.0)
    with patch.object(playback_module.time, "monotonic", return_value=101.0):
        assert handle.is_playing() is False


def test_is_playing_true_while_running_inside_duration() -> None:
    """A live process inside its duration counts as playing."""
    handle = _handle(poll=None, duration=10.0)
    with patch.object(playback_module.time, "monotonic", return_value=102.0):
        assert handle.is_playing() is True


# -- stop --------------------------------------------------------------------


def test_stop_terminates_a_running_process() -> None:
    """stop() terminates ffplay and waits for it."""
    process = _process(poll=None)
    _handle(process).stop()

    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_stop_is_a_noop_for_an_exited_process() -> None:
    """Stopping an already-finished process does nothing."""
    process = _process(poll=0)
    _handle(process).stop()

    process.terminate.assert_not_called()


def test_stop_kills_when_terminate_is_ignored() -> None:
    """A process that ignores SIGTERM is killed."""
    process = _process(poll=None)
    process.wait.side_effect = subprocess.TimeoutExpired(cmd="ffplay", timeout=2.0)

    _handle(process).stop()

    process.kill.assert_called_once()


# -- start_audio_segment -----------------------------------------------------


def test_start_audio_segment_builds_the_ffplay_command() -> None:
    """ffplay is invoked with the resolved binary and the segment bounds."""
    with (
        patch.object(playback_module, "resolve_executable", return_value="/usr/bin/ffplay"),
        patch.object(playback_module.subprocess, "Popen") as popen,
    ):
        start_audio_segment(source_path=Path("ep.mp3"), start_seconds=12.5, duration_seconds=3.25)

    command = popen.call_args.args[0]
    assert command[0] == "/usr/bin/ffplay"
    assert command[command.index("-ss") + 1] == "12.500"
    assert command[command.index("-t") + 1] == "3.250"
    assert command[-1] == "ep.mp3"


def test_start_audio_segment_detaches_stdin() -> None:
    """ffplay must not read stdin or it would swallow the TUI's keystrokes."""
    with (
        patch.object(playback_module, "resolve_executable", return_value="/usr/bin/ffplay"),
        patch.object(playback_module.subprocess, "Popen") as popen,
    ):
        start_audio_segment(source_path=Path("ep.mp3"), start_seconds=0.0, duration_seconds=1.0)

    assert popen.call_args.kwargs["stdin"] is subprocess.DEVNULL
