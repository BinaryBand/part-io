"""Tests for the live audition UI: playback wiring, keys, and screen contents."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from partio.cli.commands.audio import _audition_ui
from partio.cli.commands.audio._audition_ui import AnsweredSpan, _Playback, _screen, run_audition
from partio.cli.commands.audio._section_bar import GLYPH_CLIP, GLYPH_NO, GLYPH_PLAYHEAD

CLIP_START = 30.0
CLIP_DURATION = 10.0
REGION_START = 0.0
REGION_END = 120.0
# History either side of the clip, so segment jumps have somewhere to land.
NEIGHBOURS = [
    AnsweredSpan(start=0.0, end=10.0, answer=False),
    AnsweredSpan(start=60.0, end=70.0, answer=True),
]


def _run(*, result=True, keys=(), answered=()):
    """Drive run_audition with a stubbed Application, optionally pressing *keys*.

    Returns ``(returned_value, start_audio_segment_mock, handles)``.
    """
    handles: list[MagicMock] = []

    def _start(**_kwargs):
        handle = MagicMock()
        handle.position_seconds = CLIP_START
        handle.is_playing.return_value = True
        handles.append(handle)
        return handle

    with (
        patch.object(_audition_ui, "start_audio_segment", side_effect=_start) as start_mock,
        patch.object(_audition_ui, "Application") as app_cls,
    ):

        def _app_run():
            bindings = app_cls.call_args.kwargs["key_bindings"]
            for key in keys:
                _press(bindings, key)
            return result

        app_cls.return_value.run.side_effect = _app_run
        returned = run_audition(
            source_path=Path("ep.mp3"),
            clip_start=CLIP_START,
            clip_duration=CLIP_DURATION,
            question="Is the jingle anywhere in this clip?",
            region_start=REGION_START,
            region_end=REGION_END,
            answered=answered,
        )
    return returned, start_mock, handles


def _press(bindings, key: str) -> None:
    """Invoke the handler bound to *key* with a dummy event."""
    for binding in bindings.bindings:
        if any(str(k) == key or getattr(k, "value", None) == key for k in binding.keys):
            binding.handler(MagicMock())
            return
    raise AssertionError(f"no binding for {key!r}")


# -- playback wiring ---------------------------------------------------------


def test_run_audition_starts_playback_at_the_clip_start() -> None:
    """The clip begins playing immediately, before any key is pressed."""
    _returned, start_mock, _handles = _run()

    assert start_mock.call_args.kwargs["start_seconds"] == CLIP_START
    assert start_mock.call_args.kwargs["duration_seconds"] == CLIP_DURATION


def test_run_audition_returns_the_app_result() -> None:
    """Yes/no/quit results pass through untouched."""
    assert _run(result=True)[0] is True
    assert _run(result=False)[0] is False
    assert _run(result=None)[0] is None


def test_run_audition_stops_playback_on_exit() -> None:
    """Answering stops ffplay rather than letting the clip run on."""
    _returned, _start_mock, handles = _run()

    assert handles[-1].stop.called


# -- keys --------------------------------------------------------------------


def test_left_key_rewinds_the_cursor() -> None:
    """Left scrubs back by the scrub interval and restarts playback there."""
    _returned, start_mock, _handles = _run(keys=["left"])

    assert start_mock.call_args.kwargs["start_seconds"] == CLIP_START - 2.0


def test_right_key_advances_the_cursor() -> None:
    """Right scrubs forward by the scrub interval."""
    _returned, start_mock, _handles = _run(keys=["right"])

    assert start_mock.call_args.kwargs["start_seconds"] == CLIP_START + 2.0


def test_scrub_back_is_clamped_to_the_region_start() -> None:
    """The cursor roams the whole region but never falls out of it."""
    _returned, start_mock, _handles = _run(keys=["left"] * 20)

    assert start_mock.call_args.kwargs["start_seconds"] == REGION_START


def test_scrub_forward_is_clamped_to_the_region_end() -> None:
    """Advancing past the clip is allowed, up to the region end."""
    _returned, start_mock, _handles = _run(keys=["right"] * 50)

    assert start_mock.call_args.kwargs["start_seconds"] == REGION_END


def test_scrubbing_stops_the_previous_playback() -> None:
    """Each scrub kills the in-flight ffplay before starting the next."""
    _returned, _start_mock, handles = _run(keys=["left"])

    assert len(handles) >= 2
    assert handles[0].stop.called


def test_replay_restarts_from_the_current_cursor() -> None:
    """[r] replays without moving the cursor."""
    _returned, start_mock, _handles = _run(keys=["left", "r"])

    assert start_mock.call_args.kwargs["start_seconds"] == CLIP_START - 2.0


def test_short_remainder_still_plays_a_minimum_length() -> None:
    """Scrubbing to the very end still yields an audible clip."""
    _returned, start_mock, _handles = _run(keys=["right"] * 50)

    assert start_mock.call_args.kwargs["duration_seconds"] == 1.0


def test_roaming_far_from_the_clip_caps_the_playback_length() -> None:
    """An unexplored stretch plays a bounded preview, not minutes of audio."""
    _returned, start_mock, _handles = _run(keys=["left"] * 20)

    # Region start, with nothing judged ahead of the clip: the window would run
    # all the way to the clip end without the cap.
    assert start_mock.call_args.kwargs["duration_seconds"] == 30.0


def test_the_question_clip_is_never_trimmed_by_the_cap() -> None:
    """The clip under question plays in full even though the cap is shorter."""
    _returned, start_mock, _handles = _run()

    assert start_mock.call_args.kwargs["duration_seconds"] == CLIP_DURATION


# -- segment jumps -----------------------------------------------------------


def test_shift_right_jumps_to_the_next_segment() -> None:
    """Shift+right lands on the start of the next judged span."""
    _returned, start_mock, _handles = _run(keys=["s-right"], answered=NEIGHBOURS)

    assert start_mock.call_args.kwargs["start_seconds"] == 60.0


def test_shift_left_jumps_to_the_previous_segment() -> None:
    """Shift+left lands on the start of the previous judged span."""
    _returned, start_mock, _handles = _run(keys=["s-left"], answered=NEIGHBOURS)

    assert start_mock.call_args.kwargs["start_seconds"] == 0.0


def test_jumping_to_a_judged_span_plays_only_that_span() -> None:
    """The span you jumped to plays out and then stops, rather than running on."""
    _returned, start_mock, _handles = _run(keys=["s-right"], answered=NEIGHBOURS)

    assert start_mock.call_args.kwargs["duration_seconds"] == 10.0


def test_shift_left_returns_to_the_clip_start_from_a_scrub() -> None:
    """After nudging off the clip, a back-jump snaps to the clip's own start."""
    _returned, start_mock, _handles = _run(keys=["right", "s-left"], answered=NEIGHBOURS)

    assert start_mock.call_args.kwargs["start_seconds"] == CLIP_START


def test_jumping_is_a_no_op_at_the_ends() -> None:
    """With nothing judged, the clip is the only stop and jumps do nothing.

    Playback must be left alone rather than restarted in place.
    """
    _returned, start_mock, _handles = _run(keys=["s-left", "s-right"])

    assert start_mock.call_count == 1


# -- screen ------------------------------------------------------------------


def _screen_text(**overrides) -> str:
    playback = overrides.pop("playback", _Playback(cursor=CLIP_START))
    kwargs = {
        "source_path": Path("episode.mp3"),
        "question": "Is the jingle anywhere in this clip?",
        "clip_start": CLIP_START,
        "clip_end": CLIP_START + CLIP_DURATION,
        "region_start": REGION_START,
        "region_end": REGION_END,
        "answered": [],
        "playback": playback,
    }
    kwargs.update(overrides)
    return "".join(text for _style, text in _screen(**kwargs))


def test_screen_shows_question_source_and_clip_bounds() -> None:
    """The screen states what is being asked and about which slice of audio."""
    text = _screen_text()

    assert "Is the jingle anywhere in this clip?" in text
    assert "episode.mp3" in text
    assert "0:30.0" in text
    assert "0:40.0" in text


def test_screen_shows_the_key_legend() -> None:
    """Every action is advertised on screen."""
    text = _screen_text()

    for hint in ("[y]", "[n]", "[←/→]", "[⇧←/⇧→]", "[r]", "[q]"):
        assert hint in text


def test_screen_flags_a_cursor_outside_the_clip() -> None:
    """Roaming off the clip is called out, so the question stays unambiguous."""
    before = _screen_text(playback=_Playback(cursor=CLIP_START - 4.0))
    after = _screen_text(playback=_Playback(cursor=CLIP_START + CLIP_DURATION + 20.0))

    assert "outside this clip" in before
    assert "outside this clip" in after


def test_screen_omits_the_outside_note_inside_the_clip() -> None:
    """No note while the cursor sits inside the clip under question."""
    assert "outside this clip" not in _screen_text()


def test_screen_fits_the_terminal_width() -> None:
    """Every rendered line fits, so the closing timestamp is never truncated."""
    with patch.object(
        _audition_ui.shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))
    ):
        text = _screen_text()

    assert max(len(line) for line in text.splitlines()) <= 80


def test_screen_aligns_the_playhead_with_the_bar() -> None:
    """The playhead marker sits in the same column as the clip it points at."""
    with patch.object(
        _audition_ui.shutil, "get_terminal_size", return_value=os.terminal_size((100, 24))
    ):
        # Playhead parked at the clip start, so it must land on the clip's first cell.
        text = _screen_text(playback=_Playback(cursor=CLIP_START))

    lines = text.splitlines()
    bar_line = next(line for line in lines if GLYPH_CLIP in line)
    head_line = lines[lines.index(bar_line) + 1]

    assert head_line.index(GLYPH_PLAYHEAD) == bar_line.index(GLYPH_CLIP)


def test_screen_renders_answered_history() -> None:
    """Previously answered spans appear as shading on the bar."""
    text = _screen_text(answered=[AnsweredSpan(start=0.0, end=20.0, answer=False)])

    assert GLYPH_NO in text
