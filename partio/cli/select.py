"""Interactive selection for the CLI.

Wraps :mod:`questionary` so the user navigates a list with the arrow keys
instead of typing an index.  When stdin/stdout is not a TTY -- pipes, CI,
tests -- selection degrades to a numbered Rich prompt so every menu stays
scriptable.

Every interactive prompt binds ``esc`` to :data:`GO_BACK` via :func:`bind_back`,
so callers can distinguish "step back one screen" (esc) from "abandon the whole
command" (ctrl-c, which yields ``None``).
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import questionary
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.prompt import Prompt

T = TypeVar("T")


class GoBack:
    """Sentinel type for the "step back one screen" answer."""

    def __repr__(self) -> str:
        """Render as GO_BACK for debugging/logging."""
        return "GO_BACK"

    def __bool__(self) -> bool:
        """Falsy, so a forgotten ``is GO_BACK`` check cannot read as a real answer."""
        return False


GO_BACK = GoBack()
"""Returned by any interactive prompt when the user pressed ``esc``."""


def bind_back(question: questionary.Question) -> questionary.Question:
    """Bind ``esc`` on *question* so answering it can yield :data:`GO_BACK`.

    The new binding is *merged* rather than added in place: list prompts expose
    a mutable ``KeyBindings``, but text-style prompts (text/path/confirm) expose
    an immutable ``_MergedKeyBindings`` that has no ``add``.  Merging covers
    both, and putting ours last gives esc priority over any default binding.

    Bound non-eagerly on purpose: ``esc`` also opens the escape sequences that
    encode arrow keys, so prompt_toolkit must be free to wait a moment and see
    whether more bytes follow before treating it as a bare keypress.
    """
    extra = KeyBindings()

    @extra.add("escape")
    def _on_escape(event) -> None:  # noqa: ANN001 - prompt_toolkit event object
        # A second exit() on an application that is already finishing raises,
        # so ignore repeat presses arriving during teardown.
        if not event.app.is_done:
            event.app.exit(result=GO_BACK)

    existing = question.application.key_bindings
    question.application.key_bindings = (
        merge_key_bindings([existing, extra]) if existing is not None else extra
    )
    return question


_STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:cyan"),
        ("separator", "fg:#6c6c6c"),
        ("instruction", "fg:#6c6c6c"),
        ("answer", "fg:cyan bold"),
    ]
)

_HELP_STYLE = "fg:#6c6c6c"
_INSTRUCTION = "(arrow keys, type to filter, esc to go back)"
_MULTI_INSTRUCTION = "(space toggles, a toggles all, enter confirms, esc to go back)"

# Width taken by the "? " prefix, the title/help gap, and a right-hand safety margin.
# Answering echoes "? <message> <title>", so that line -- not the pointer row -- is
# the widest thing rendered and the one the help must be sized against.
_ROW_CHROME = 8
# Gap between the title column and the help column.
_GAP = 3
# A long title must never squeeze the metadata column out entirely; reserve at
# least enough for "YYYY-MM-DD   999.9 MB", the widest metadata we render.
_MIN_HELP_WIDTH = 22
_MIN_TITLE_WIDTH = 20


@dataclass(frozen=True)
class Option(Generic[T]):
    """One selectable row.

    *title* is the bold label, *value* is what :func:`select_one` returns when
    the row is chosen, *help* is the dimmed trailing description, and *group*
    renders a separator heading whenever it changes between consecutive rows.
    """

    title: str
    value: T
    help: str = ""
    group: str | None = None
    disabled: str | None = None
    """Reason this row cannot be picked; ``None`` means selectable."""


def select_one(
    message: str,
    options: Sequence[Option[T]],
    *,
    console: Console,
) -> T | GoBack | None:
    """Ask the user to choose one of *options*.

    Returns the chosen option's ``value``, :data:`GO_BACK` if the user pressed
    esc, or ``None`` if they cancelled outright.
    """
    if not options:
        return None
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _arrow_key_select(message, options)
    return _numbered_select(message, options, console=console)


def _arrow_key_select(message: str, options: Sequence[Option[T]]) -> T | GoBack | None:
    """Render the questionary arrow-key menu."""
    width, help_width = _column_widths(message, options)
    choices: list[questionary.Choice | questionary.Separator] = []
    last_group: str | None = None

    for index, option in enumerate(options):
        if option.group != last_group:
            if index:
                choices.append(questionary.Separator(" "))
            if option.group:
                choices.append(questionary.Separator(f"  {option.group}"))
            last_group = option.group
        choices.append(
            questionary.Choice(title=_title(option, width, help_width), value=option.value)
        )

    return bind_back(
        questionary.select(
            message,
            choices=choices,
            style=_STYLE,
            instruction=_INSTRUCTION,
            use_search_filter=True,
            use_jk_keys=False,
        )
    ).ask()


def select_many(
    message: str,
    options: Sequence[Option[T]],
    *,
    console: Console,
) -> list[T] | GoBack | None:
    """Ask the user to check any number of *options*.

    Returns the chosen values in listed order, :data:`GO_BACK` if the user
    pressed esc, or ``None`` if they cancelled outright.  An empty list means
    "confirmed, but nothing checked".
    """
    if not options:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _checkbox_select(message, options)
    return _numbered_multi_select(message, options, console=console)


def _checkbox_select(message: str, options: Sequence[Option[T]]) -> list[T] | GoBack | None:
    """Render the questionary checkbox menu."""
    width, help_width = _column_widths(message, options)
    choices = [
        questionary.Choice(
            title=_title(option, width, help_width),
            value=option.value,
            disabled=option.disabled,
        )
        for option in options
    ]
    return bind_back(
        questionary.checkbox(
            message,
            choices=choices,
            style=_STYLE,
            instruction=_MULTI_INSTRUCTION,
            use_search_filter=True,
            use_jk_keys=False,
        )
    ).ask()


def _numbered_multi_select(
    message: str,
    options: Sequence[Option[T]],
    *,
    console: Console,
) -> list[T] | None:
    """Non-TTY fallback: print a numbered list and read comma-separated indexes."""
    selectable = [option for option in options if option.disabled is None]
    for index, option in enumerate(options, start=1):
        label = f"  [bold]{index}[/bold]. [green]{option.title}[/green]"
        if option.disabled is not None:
            label = f"  [dim]{index}. {option.title} ({option.disabled})[/dim]"
        suffix = f"  -- {option.help}" if option.help else ""
        console.print(f"{label}{suffix}")
    console.print()

    try:
        raw = Prompt.ask(f"{message} (comma-separated numbers)", console=console, default="")
    except (KeyboardInterrupt, EOFError):
        return None

    chosen: list[T] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped.isdigit():
            continue
        position = int(stripped) - 1
        if 0 <= position < len(options) and options[position] in selectable:
            chosen.append(options[position].value)
    return chosen


def _column_widths(message: str, options: Sequence[Option[T]]) -> tuple[int, int]:
    """Split the usable row width into ``(title_width, help_width)``.

    Both columns shrink to fit the terminal, but a long title is capped before
    the help column is starved -- otherwise episode titles would push their
    date/size metadata off the row entirely.
    """
    longest_title = max(len(option.title) for option in options)
    available = shutil.get_terminal_size().columns - len(message) - _ROW_CHROME
    if not any(option.help for option in options):
        return min(longest_title, max(_MIN_TITLE_WIDTH, available)), 0
    title_width = min(longest_title, max(_MIN_TITLE_WIDTH, available - _GAP - _MIN_HELP_WIDTH))
    return title_width, max(0, available - title_width - _GAP)


def _clip(text: str, width: int) -> str:
    """Hard-truncate *text* to *width*, marking the cut with an ellipsis."""
    if len(text) <= width:
        return text
    return text[: max(1, width - 3)] + "..."


def _title(option: Option[T], width: int, help_width: int) -> list[tuple[str, str]]:
    """Build the prompt_toolkit formatted title: name + dimmed help.

    Both columns are clipped so each row stays on one line -- a wrapped row
    also wraps the echoed answer once the choice is made.  Clipping is done by
    hand rather than with :func:`textwrap.shorten`, which collapses runs of
    spaces and would break the aligned ``date   size`` metadata column.
    """
    title = _clip(option.title, width)
    if not option.help or help_width <= 0:
        return [("", title)]
    return [
        ("", title.ljust(width)),
        (_HELP_STYLE, f"{' ' * _GAP}{_clip(option.help, help_width)}"),
    ]


def _numbered_select(
    message: str,
    options: Sequence[Option[T]],
    *,
    console: Console,
) -> T | None:
    """Non-TTY fallback: print a numbered list and read an index."""
    last_group: str | None = None
    for index, option in enumerate(options, start=1):
        if option.group != last_group:
            console.print(f"\n[dim]{option.group}[/dim]" if option.group else "")
            last_group = option.group
        suffix = f"  -- {option.help}" if option.help else ""
        console.print(f"  [bold]{index}[/bold]. [green]{option.title}[/green]{suffix}")
    console.print()

    numbers = [str(i) for i in range(1, len(options) + 1)]
    try:
        choice = Prompt.ask(
            message,
            console=console,
            choices=numbers,
            default="1",
            show_choices=False,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        return None
    return options[int(choice) - 1].value


__all__ = ["GO_BACK", "GoBack", "Option", "bind_back", "select_many", "select_one"]
