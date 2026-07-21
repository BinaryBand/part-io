"""Per-arg walkthrough prompting for the interactive picker.

Introspects command functions to discover required options, then walks the
user through each one with a Rich prompt that matches the option's type
annotation.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin, get_type_hints

import typer
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from partio.cli.commands.library._store import default_store

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


def prompt_for_args(entry: CommandEntry) -> list[str]:
    """Walk the user through every required option of *entry*.

    For each required option, prints its ``help`` text, then prompts with a
    Rich helper that matches the annotation type:

    * ``Path`` / ``str`` → ``Prompt.ask``
    * ``int``             → ``IntPrompt.ask``
    * ``float``           → ``FloatPrompt.ask``
    * ``bool``            → ``Confirm.ask``

    Returns a flat ``["--flag", "value", ...]`` list ready to be appended
    to a Typer invocation.
    """
    args: list[str] = []
    for flag_name, inner_type, option_info in required_options(entry.fn):
        label = option_info.help or flag_name
        console.print(f"[bold]{label}[/bold]")
        value = _rich_prompt(inner_type, flag_name)
        args.append(flag_name)
        args.append(str(value))
    return args


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


def _rich_prompt(inner_type: type, flag_name: str) -> str | bool | int | float:
    """Dispatch to the correct Rich prompt class for *inner_type*."""
    from pathlib import Path

    prompt_text = flag_name.lstrip("-").replace("-", " ")
    if inner_type is bool:
        return Confirm.ask(f"{prompt_text}?", console=console)
    if inner_type is int:
        return IntPrompt.ask(prompt_text, console=console)
    if inner_type is float:
        return FloatPrompt.ask(prompt_text, console=console)
    if inner_type is Path:
        return _prompt_path(prompt_text)
    # Plain string prompt.
    return Prompt.ask(prompt_text, console=console)


def _library_entries() -> list[AudioPathEntry]:
    """Return remembered audio paths, or an empty list if the store is unreadable."""
    try:
        return default_store().list_items()
    except (OSError, ValueError):  # A broken/missing library never blocks a prompt.
        return []


def _prompt_path(prompt_text: str) -> str:
    """Prompt for a filesystem path, offering the remembered library as a picker.

    When the library has entries, they are listed as a numbered menu so the user
    can pick a remembered file (e.g. an episode just downloaded) instead of
    typing a path. Choosing ``c`` -- or an empty library -- falls back to a plain
    text prompt.
    """
    entries = _library_entries()
    if not entries:
        return Prompt.ask(prompt_text, console=console)

    console.print("[dim]Remembered audio (from your library):[/dim]")
    for idx, entry in enumerate(entries, start=1):
        console.print(
            f"  [bold]{idx}[/bold]. [green]{entry.label}[/green] "
            f"[dim]({entry.kind.value}) {entry.path}[/dim]"
        )
    console.print(f"  [bold]{_CUSTOM_PATH_CHOICE}[/bold]. [yellow]enter a path manually[/yellow]")

    numbers = [str(i) for i in range(1, len(entries) + 1)]
    choice = Prompt.ask(
        prompt_text,
        console=console,
        choices=[*numbers, _CUSTOM_PATH_CHOICE],
        default="1",
        show_choices=False,
    ).strip()

    if choice == _CUSTOM_PATH_CHOICE:
        return Prompt.ask("path", console=console)
    return str(entries[int(choice) - 1].path)
