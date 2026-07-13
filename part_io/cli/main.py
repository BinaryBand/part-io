"""cli.main: the command-line interface.

When invoked with no arguments ``part-io`` presents a numbered picker of
available commands.  Commands are registered via the :mod:`part_io.cli.registry`
decorator at each function's definition site and assembled here automatically.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Import command modules to trigger @command registration.
from part_io.cli import audio_bootstrap, audio_locate, audio_review, audio_search  # noqa: F401
from part_io.cli.registry import get_commands

app = typer.Typer(add_completion=False, invoke_without_command=True)
console = Console()

# -- assemble the command tree from the registry ---------------------------
for _entry in get_commands():
    app.command(_entry.name, help=_entry.help)(_entry.fn)


# -- picker ----------------------------------------------------------------


def _show_picker() -> None:
    """Display a numbered menu and dispatch the selected command."""
    commands = get_commands()

    console.print(Panel("[bold]part-io[/bold] — audio tooling CLI", style="cyan", expand=False))
    for idx, entry in enumerate(commands, start=1):
        console.print(f"  [bold]{idx}[/bold]. [green]{entry.name}[/green]  — {entry.help}")
    console.print()

    valid_names = [entry.name for entry in commands]
    range_hint = f"1-{len(commands)}"
    raw = Prompt.ask(
        f"Pick a command [{range_hint}]",
        console=console,
        choices=[str(i) for i in range(1, len(commands) + 1)] + valid_names,
        show_choices=False,
    )
    choice = raw.strip()

    if not choice:
        console.print("No selection.", style="yellow")
        raise typer.Exit(code=0)

    # Accept a number or a command name.
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(commands):
            selected = commands[idx].name
        else:
            console.print(f"Invalid choice: {choice}", style="red")
            raise typer.Exit(code=1)
    elif choice in valid_names:
        selected = choice
    else:
        console.print(f"Unknown command: {choice}", style="red")
        raise typer.Exit(code=1)

    # Re-invoke the app with the chosen subcommand.
    app([selected], standalone_mode=False)


# -- callback (runs on every invocation) ------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit."),
    ] = False,
) -> None:
    """part-io: audio tooling CLI."""
    if version:
        from importlib.metadata import version as pkg_version

        console.print(f"part-io {pkg_version('part-io')}")
        raise typer.Exit(code=0)

    if ctx.invoked_subcommand is None:
        _show_picker()
