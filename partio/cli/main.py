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


def _show_picker() -> None:
    """Display a numbered menu and dispatch the selected command."""
    commands = discover()

    console.print(Panel("[bold]partio[/bold] -- audio tooling CLI", style="cyan", expand=False))
    for idx, entry in enumerate(commands, start=1):
        label = f"{entry.group} {entry.name}" if entry.group else entry.name
        console.print(f"  [bold]{idx}[/bold]. [green]{label}[/green]  -- {entry.help}")
    console.print()

    valid_labels = [(f"{e.group} {e.name}" if e.group else e.name) for e in commands]
    label_to_entry: dict[str, CommandEntry] = dict(zip(valid_labels, commands, strict=True))
    range_hint = f"1-{len(commands)}"
    raw = Prompt.ask(
        f"Pick a command [{range_hint}]",
        console=console,
        choices=[str(i) for i in range(1, len(commands) + 1)] + valid_labels,
        show_choices=False,
    )
    choice = raw.strip()

    if not choice:
        console.print("No selection.", style="yellow")
        raise typer.Exit(code=0)

    # Accept a number or a command label.
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(commands):
            selected = valid_labels[idx]
        else:
            console.print(f"Invalid choice: {choice}", style="red")
            raise typer.Exit(code=1)
    elif choice in valid_labels:
        selected = choice
    else:
        console.print(f"Unknown command: {choice}", style="red")
        raise typer.Exit(code=1)

    # Re-invoke the app with the chosen subcommand, walking through required
    # args when the picker is used (non-interactive fallback via prompt= on
    # each Option still works for direct terminal invocation).
    from partio.cli.prompting import prompt_for_args

    entry = label_to_entry[selected]
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
