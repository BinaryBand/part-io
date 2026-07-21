"""Live audition UI: a section bar, a moving playhead, and instant y/n keys.

Playback runs in the background while the question is on screen, so a listener
answers the moment they hear (or stop hearing) the jingle instead of waiting
out the clip.  Arrow keys rewind/advance the playback cursor for extra context
and shift+arrows jump between judged segments anywhere in the region; the
question always refers to the highlighted clip, which never moves.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from partio.adapters.audio.playback import PlaybackHandle, start_audio_segment
from partio.cli.commands.audio._section_bar import (
    AnsweredSpan,
    bar_cells,
    bar_fragments,
    format_timestamp,
    jump_target,
    play_window_end,
    playhead_row,
    segment_stops,
)

_SCRUB_SECONDS = 2.0
_MIN_PLAY_SECONDS = 1.0
# Ceiling on one playback run, so roaming far from the clip does not commit the
# listener to minutes of audio.  Never trims the clip under question itself.
_MAX_PLAY_SECONDS = 30.0
_REFRESH_SECONDS = 0.1
_MIN_BAR_WIDTH = 20

_STYLE = Style(
    [
        ("bar.base", "fg:#4e4e4e"),
        ("bar.no", "fg:#6c6c6c"),
        ("bar.yes", "fg:green"),
        ("bar.clip", "fg:cyan bold"),
        ("playhead", "fg:yellow bold"),
        ("question", "bold"),
        ("meta", "fg:#6c6c6c"),
        ("key", "fg:cyan bold"),
        ("edge", "fg:#4e4e4e"),
    ]
)


@dataclass
class _Playback:
    """Mutable playback cursor + handle for the running audition."""

    cursor: float
    handle: PlaybackHandle | None = None


def _bar_width(*, chrome: int) -> int:
    """Bar columns that fit once the surrounding labels have taken their space."""
    return max(_MIN_BAR_WIDTH, shutil.get_terminal_size().columns - chrome - 1)


def run_audition(
    *,
    source_path: Path,
    clip_start: float,
    clip_duration: float,
    question: str,
    region_start: float,
    region_end: float,
    answered: Sequence[AnsweredSpan] = (),
) -> bool | None:
    """Play the clip and ask *question* live.

    Returns ``True`` for yes, ``False`` for no, and ``None`` if the listener
    quit. Playback stops as soon as an answer is given.
    """
    clip_end = clip_start + clip_duration
    playback = _Playback(cursor=clip_start)

    def _stop() -> None:
        if playback.handle is not None:
            playback.handle.stop()
            playback.handle = None

    def _play(from_seconds: float) -> None:
        _stop()
        playback.cursor = min(max(from_seconds, region_start), region_end)
        window_end = play_window_end(
            position=playback.cursor,
            clip_start=clip_start,
            clip_end=clip_end,
            region_end=region_end,
            answered=answered,
        )
        # The cap must never trim the clip under question, however long the
        # caller made it.
        ceiling = max(_MAX_PLAY_SECONDS, clip_duration)
        duration = min(max(_MIN_PLAY_SECONDS, window_end - playback.cursor), ceiling)
        playback.handle = start_audio_segment(
            source_path=source_path,
            start_seconds=playback.cursor,
            duration_seconds=duration,
        )

    def _render() -> FormattedText:
        return FormattedText(
            _screen(
                source_path=source_path,
                question=question,
                clip_start=clip_start,
                clip_end=clip_end,
                region_start=region_start,
                region_end=region_end,
                answered=answered,
                playback=playback,
            )
        )

    bindings = _key_bindings(
        play=_play,
        playback=playback,
        stops=segment_stops(clip_start=clip_start, answered=answered),
    )
    app: Application[bool | None] = Application(
        layout=Layout(Window(FormattedTextControl(_render), wrap_lines=False)),
        key_bindings=bindings,
        style=_STYLE,
        refresh_interval=_REFRESH_SECONDS,
        full_screen=False,
    )

    _play(clip_start)
    try:
        return app.run()
    finally:
        _stop()


def _key_bindings(
    *, play: Callable[[float], None], playback: _Playback, stops: Sequence[float]
) -> KeyBindings:
    """Bind the audition keys: answer, scrub, jump, replay, quit."""
    bindings = KeyBindings()
    _add_answer_keys(bindings)
    _add_navigation_keys(bindings, play=play, playback=playback, stops=stops)
    return bindings


def _add_answer_keys(bindings: KeyBindings) -> None:
    """Bind the keys that end the audition: yes, no, quit."""

    @bindings.add("y")
    @bindings.add("Y")
    def _on_yes(event) -> None:  # noqa: ANN001 - prompt_toolkit event object
        event.app.exit(result=True)

    @bindings.add("n")
    @bindings.add("N")
    def _on_no(event) -> None:  # noqa: ANN001
        event.app.exit(result=False)

    @bindings.add("q")
    @bindings.add("c-c")
    @bindings.add("escape")
    def _on_quit(event) -> None:  # noqa: ANN001
        event.app.exit(result=None)


def _add_navigation_keys(
    bindings: KeyBindings,
    *,
    play: Callable[[float], None],
    playback: _Playback,
    stops: Sequence[float],
) -> None:
    """Bind the keys that move the playback cursor: scrub, jump, replay."""

    def _jump(*, forward: bool) -> None:
        """Move to the neighbouring segment, or stay put at the region's ends."""
        target = jump_target(position=playback.cursor, stops=stops, forward=forward)
        if target is not None:
            play(target)

    @bindings.add("left")
    def _on_back(_event) -> None:  # noqa: ANN001
        play(playback.cursor - _SCRUB_SECONDS)

    @bindings.add("right")
    def _on_forward(_event) -> None:  # noqa: ANN001
        play(playback.cursor + _SCRUB_SECONDS)

    @bindings.add("s-left")
    def _on_jump_back(_event) -> None:  # noqa: ANN001
        _jump(forward=False)

    @bindings.add("s-right")
    def _on_jump_forward(_event) -> None:  # noqa: ANN001
        _jump(forward=True)

    @bindings.add("r")
    def _on_replay(_event) -> None:  # noqa: ANN001
        play(playback.cursor)


def _screen(
    *,
    source_path: Path,
    question: str,
    clip_start: float,
    clip_end: float,
    region_start: float,
    region_end: float,
    answered: Sequence[AnsweredSpan],
    playback: _Playback,
) -> list[tuple[str, str]]:
    """Build the full audition screen as styled fragments."""
    # Derive the bar width and the playhead indent from the same prefix/suffix
    # strings, so the marker can never drift out of alignment with the bar.
    prefix = f"  {format_timestamp(region_start)} ├"
    suffix = f"┤ {format_timestamp(region_end)}"
    width = _bar_width(chrome=len(prefix) + len(suffix))
    position = playback.handle.position_seconds if playback.handle else playback.cursor
    cells = bar_cells(
        region_start=region_start,
        region_end=region_end,
        clip_start=clip_start,
        clip_end=clip_end,
        answered=answered,
        width=width,
    )

    lines: list[tuple[str, str]] = [
        ("class:meta", f"\n  {source_path.name}"),
        (
            "class:meta",
            f"   region {format_timestamp(region_start)} - {format_timestamp(region_end)}\n\n",
        ),
        ("class:edge", prefix),
    ]
    lines.extend(bar_fragments(cells))
    lines.append(("class:edge", f"{suffix}\n"))
    lines.append(
        (
            "class:playhead",
            " " * len(prefix)
            + playhead_row(
                position=position,
                region_start=region_start,
                region_end=region_end,
                width=width,
            )
            + "\n",
        )
    )
    lines.extend(
        _status_lines(
            playback=playback,
            clip_start=clip_start,
            clip_end=clip_end,
            region_end=region_end,
            answered=answered,
        )
    )
    lines.append(("class:question", f"\n  {question}\n\n"))
    lines.extend(_key_hint_fragments())
    return lines


def _status_lines(
    *,
    playback: _Playback,
    clip_start: float,
    clip_end: float,
    region_end: float,
    answered: Sequence[AnsweredSpan],
) -> list[tuple[str, str]]:
    """Render the clip bounds and the live playback readout."""
    handle = playback.handle
    playing = handle is not None and handle.is_playing()
    marker = "▶" if playing else "■"
    position = handle.position_seconds if handle else playback.cursor
    # The cursor roams the whole region, so it can sit either side of the clip.
    roaming = not (clip_start <= playback.cursor < clip_end)
    window_end = play_window_end(
        position=playback.cursor,
        clip_start=clip_start,
        clip_end=clip_end,
        region_end=region_end,
        answered=answered,
    )
    return [
        (
            "class:meta",
            f"\n  this clip  {format_timestamp(clip_start)} - {format_timestamp(clip_end)}"
            + ("   (outside this clip)" if roaming else ""),
        ),
        (
            "class:meta",
            f"\n  {marker} {format_timestamp(position)} / {format_timestamp(window_end)}\n",
        ),
    ]


def _key_hint_fragments() -> list[tuple[str, str]]:
    """Render the key legend, split so neither row outgrows a narrow terminal."""
    rows = [
        [("y", "yes"), ("n", "no"), ("r", "replay"), ("q", "quit")],
        [("←/→", f"scrub {_SCRUB_SECONDS:g}s"), ("⇧←/⇧→", "segment")],
    ]
    fragments: list[tuple[str, str]] = []
    for hints in rows:
        fragments.append(("", "  "))
        for key, label in hints:
            fragments.append(("class:key", f"[{key}]"))
            fragments.append(("class:meta", f" {label}   "))
        fragments.append(("", "\n"))
    return fragments


__all__ = ["AnsweredSpan", "run_audition"]
