"""Per-arg walkthrough prompting for the interactive picker.

Introspects command functions to discover required options, then walks the
user through each one with a questionary prompt that matches the option's
type annotation.  ``esc`` steps back through the walkthrough.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin, get_type_hints

import questionary
import typer
from rich.console import Console

from partio.cli.commands.library._store import default_store
from partio.cli.select import GO_BACK, GoBack, Option, bind_back, select_one
from partio.core.ports import AudioPathKind

if TYPE_CHECKING:
    from partio.cli.registry import CommandEntry
    from partio.core.ports import AudioPathEntry

console = Console()

_REQUIRED = inspect.Parameter.empty
_MIN_ANNOTATION_ARGS = 2
_CUSTOM_PATH_CHOICE = "c"


def required_options(fn: Callable[..., Any]) -> list[tuple[str, type, typer.models.OptionInfo]]:
    """Find required ``Annotated`` options on *fn*.

    A parameter is considered *required* when it has an ``Annotated[T,
    typer.models.OptionInfo(...)]`` annotation **and** no default value was
    supplied (``param.default is inspect.Parameter.empty``).

    Returns a list of ``(flag_name, inner_type, option_info)`` triples.
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    results: list[tuple[str, type, typer.models.OptionInfo]] = []

    for name, param in sig.parameters.items():
        if name == "ctx":
            continue

        hint = hints.get(name)
        if hint is None or get_origin(hint) is not Annotated:
            continue

        args = get_args(hint)
        if len(args) < _MIN_ANNOTATION_ARGS or not isinstance(args[1], typer.models.OptionInfo):
            continue

        if param.default is not _REQUIRED:
            continue

        inner_type = args[0]
        option_info: typer.models.OptionInfo = args[1]
        flag_name = _extract_flag(option_info, name)
        results.append((flag_name, inner_type, option_info))

    return results


def prompt_for_args(entry: CommandEntry) -> list[str] | GoBack | None:
    """Walk the user through every required option of *entry*.

    For each required option, prints its ``help`` text, then prompts with a
    questionary helper matching the annotation type (``Path`` also offers the
    remembered library as a picker).

    ``esc`` steps back to the previous option; pressing it on the first option
    returns :data:`GO_BACK` so the caller can redisplay whatever came before.
    ``ctrl-c`` returns ``None`` to abandon the command outright.  Otherwise
    returns a flat ``["--flag", "value", ...]`` list ready to be appended to a
    Typer invocation.
    """
    options = required_options(entry.fn)
    answers: list[str] = [""] * len(options)
    index = 0

    while index < len(options):
        flag_name, inner_type, option_info = options[index]
        console.print(f"[bold]{option_info.help or flag_name}[/bold]")
        value = _ask(inner_type, flag_name)
        if value is None:
            return None
        if isinstance(value, GoBack):
            if index == 0:
                return GO_BACK
            index -= 1
            continue
        answers[index] = str(value)
        index += 1

    flags = [flag for flag, _type, _info in options]
    return [part for pair in zip(flags, answers, strict=True) for part in pair]


def _extract_flag(
    option_info: typer.models.OptionInfo,
    param_name: str,
) -> str:
    """Derive the primary ``--flag`` string from an ``OptionInfo``."""
    if isinstance(option_info.default, str) and option_info.default.startswith("--"):
        return option_info.default
    if option_info.param_decls:
        return option_info.param_decls[0]
    return f"--{param_name.replace('_', '-')}"


def _ask(inner_type: type, flag_name: str) -> str | bool | int | float | GoBack | None:
    """Dispatch to the questionary prompt matching *inner_type*.

    Returns :data:`GO_BACK` when the user pressed esc and ``None`` when they
    cancelled.
    """
    prompt_text = flag_name.lstrip("-").replace("-", " ")
    if inner_type is bool:
        return bind_back(questionary.confirm(f"{prompt_text}?")).ask()
    if inner_type is int:
        return _ask_number(prompt_text, cast=int, name="integer")
    if inner_type is float:
        return _ask_number(prompt_text, cast=float, name="number")
    if inner_type is Path:
        return _prompt_path(prompt_text)
    return bind_back(questionary.text(prompt_text)).ask()


def _ask_number(
    prompt_text: str, *, cast: Callable[[str], int | float], name: str
) -> int | float | GoBack | None:
    """Prompt for a number, re-asking until the text parses."""

    def _validate(text: str) -> bool | str:
        try:
            cast(text)
        except ValueError:
            return f"Enter a valid {name}"
        return True

    answer = bind_back(questionary.text(prompt_text, validate=_validate)).ask()
    if answer is None or isinstance(answer, GoBack):
        return answer
    return cast(answer)


def _library_entries() -> list[AudioPathEntry]:
    """Return remembered audio paths, or an empty list if the store is unreadable."""
    try:
        return default_store().list_items()
    except (OSError, ValueError):  # A broken/missing library never blocks a prompt.
        return []


def _kind_for(prompt_text: str) -> AudioPathKind | None:
    """The library kind a prompt wants, inferred from its flag name.

    The audio commands name their options after the thing they take --
    ``--source`` / ``--sample`` -- which is exactly the ``AudioPathKind``
    vocabulary, so the mapping needs no separate table. Any other flag name
    yields ``None``, meaning "offer everything".
    """
    try:
        return AudioPathKind(prompt_text)
    except ValueError:
        return None


def _prompt_path(prompt_text: str) -> str | GoBack | None:
    """Prompt for a filesystem path, offering the remembered library as a picker.

    Entries are narrowed to the kind the flag asks for, so ``--sample`` offers
    bootstrapped seed clips rather than whole episodes. With nothing suitable
    remembered -- or on choosing "enter a path manually" -- this falls back to a
    plain path prompt.
    """
    kind = _kind_for(prompt_text)
    entries = [e for e in _library_entries() if kind is None or e.kind is kind]
    if not entries:
        return bind_back(questionary.path(prompt_text)).ask()

    heading = f"remembered {kind.value}s" if kind is not None else "remembered audio"
    options = [
        Option(
            title=entry.label,
            value=str(entry.path),
            help=f"({entry.kind.value}) {entry.path}",
            group=heading,
        )
        for entry in entries
    ]
    options.append(Option(title="enter a path manually", value=_CUSTOM_PATH_CHOICE))

    chosen = select_one(prompt_text, options, console=console)
    if chosen == _CUSTOM_PATH_CHOICE:
        # esc at the manual prompt returns to this picker rather than skipping past it.
        typed = bind_back(questionary.path("path")).ask()
        return _prompt_path(prompt_text) if isinstance(typed, GoBack) else typed
    return chosen
