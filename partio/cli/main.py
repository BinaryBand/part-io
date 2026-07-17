"""cli.main: the central command-line interface.

Assembles the Typer app from the registry via :func:`discover`, builds one
sub-app per command group, provides a global ``--json`` flag and a Rich
numbered picker for bare invocation.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from partio.cli.registry import CommandEntry, discover

app = typer.Typer(add_completion=False, invoke_without_command=True, rich_markup_mode="rich")
console = Console()

# -- assemble the command tree from auto-discovery --------------------------


def _build_app() -> None:
    """Populate *app* with commands discovered from ``cli.commands``."""
    groups: dict[str, list[CommandEntry]] = {}
    roots: list[CommandEntry] = []

    for entry in discover():
        if entry.group:
            groups.setdefault(entry.group, []).append(entry)
        else:
            roots.append(entry)

    for group_name, entries in groups.items():
        sub = typer.Typer(
            rich_markup_mode="rich",
            no_args_is_help=True,
            help=f"{group_name} commands",
        )
        for e in entries:
            sub.command(e.name, help=e.help)(e.fn)
        app.add_typer(sub, name=group_name)

    for entry in roots:
        app.command(entry.name, help=entry.help)(entry.fn)


_build_app()


# -- picker ----------------------------------------------------------------


_QUIT_CHOICE = "q"


def _label_for(entry: CommandEntry) -> str:
    """Render an entry's ``group name`` (or bare ``name``) invocation label."""
    return f"{entry.group} {entry.name}" if entry.group else entry.name


def _show_picker() -> None:
    """Display a numbered menu, grouped by command group, and dispatch the choice."""
    commands = discover()

    console.print(Panel("[bold]partio[/bold] -- audio tooling CLI", style="cyan", expand=False))

    last_group: str | None = None
    for idx, entry in enumerate(commands, start=1):
        if entry.group != last_group:
            console.print(f"\n[dim]{entry.group or 'commands'}[/dim]")
            last_group = entry.group
        console.print(f"  [bold]{idx}[/bold]. [green]{entry.name}[/green]  -- {entry.help}")
    console.print(f"\n  [bold]{_QUIT_CHOICE}[/bold]. [yellow]quit[/yellow]")
    console.print()

    labels = [_label_for(e) for e in commands]
    numbers = [str(i) for i in range(1, len(commands) + 1)]
    try:
        choice = Prompt.ask(
            f"Pick a command [1-{len(commands)}], or '{_QUIT_CHOICE}' to quit",
            console=console,
            choices=[*numbers, *labels, _QUIT_CHOICE],
            default=_QUIT_CHOICE,
            show_choices=False,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\nCancelled.", style="yellow")
        raise typer.Exit(code=0) from None

    if choice == _QUIT_CHOICE:
        raise typer.Exit(code=0)

    # ``choices`` above guarantees a number or a valid label reaches here.
    selected = labels[int(choice) - 1] if choice.isdigit() else choice

    # Re-invoke the app with the chosen subcommand, walking through required
    # args when the picker is used (non-interactive fallback via prompt= on
    # each Option still works for direct terminal invocation).
    from partio.cli.prompting import prompt_for_args

    entry = commands[labels.index(selected)]
    extra_args = prompt_for_args(entry)
    app(selected.split() + extra_args, standalone_mode=False)


# -- callback (runs on every invocation) ------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON."),
    ] = False,
) -> None:
    """partio: audio tooling CLI."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output

    if version:
        from importlib.metadata import version as pkg_version

        console.print(f"partio {pkg_version('partio')}")
        raise typer.Exit(code=0)

    if ctx.invoked_subcommand is None:
        _show_picker()
