"""Interactive single-choice selection for the CLI.

Wraps :mod:`questionary` so the user navigates a list with the arrow keys
instead of typing an index.  When stdin/stdout is not a TTY -- pipes, CI,
tests -- selection degrades to a numbered Rich prompt so every menu stays
scriptable.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import questionary
from prompt_toolkit.styles import Style
from rich.prompt import Prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

T = TypeVar("T")

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
_INSTRUCTION = "(arrow keys, type to filter, ctrl-c to quit)"

# Width taken by the "? " prefix, the title/help gap, and a right-hand safety margin.
# Answering echoes "? <message> <title>", so that line -- not the pointer row -- is
# the widest thing rendered and the one the help must be sized against.
_ROW_CHROME = 8
# Never squeeze help below this, even in a very narrow terminal -- drop it instead.
_MIN_HELP_WIDTH = 16


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


def select_one(
    message: str,
    options: Sequence[Option[T]],
    *,
    console: Console,
) -> T | None:
    """Ask the user to choose one of *options*.

    Returns the chosen option's ``value``, or ``None`` if the user cancelled.
    """
    if not options:
        return None
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _arrow_key_select(message, options)
    return _numbered_select(message, options, console=console)


def _arrow_key_select(message: str, options: Sequence[Option[T]]) -> T | None:
    """Render the questionary arrow-key menu."""
    width = max(len(option.title) for option in options)
    help_width = shutil.get_terminal_size().columns - width - len(message) - _ROW_CHROME
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

    return questionary.select(
        message,
        choices=choices,
        style=_STYLE,
        instruction=_INSTRUCTION,
        use_search_filter=True,
        use_jk_keys=False,
    ).ask()


def _title(option: Option[T], width: int, help_width: int) -> list[tuple[str, str]]:
    """Build the prompt_toolkit formatted title: name + dimmed help.

    The help is shortened to *help_width* so each row stays on one line -- a
    wrapped row also wraps the echoed answer once the choice is made.
    """
    if not option.help or help_width < _MIN_HELP_WIDTH:
        return [("", option.title)]
    help_text = textwrap.shorten(option.help, width=help_width, placeholder="...")
    return [("", option.title.ljust(width)), (_HELP_STYLE, f"   {help_text}")]


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


__all__ = ["Option", "select_one"]
