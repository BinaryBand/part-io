"""cli.main: the central command-line interface.

Assembles the Typer app from the registry via :func:`discover`, builds one
sub-app per command group, provides a global ``--json`` flag and a Rich
numbered picker for bare invocation.
"""

from __future__ import annotations

import contextlib
from importlib.metadata import version as pkg_version
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from partio.cli.prompting import prompt_for_args
from partio.cli.registry import CommandEntry, discover
from partio.cli.select import GoBack, Option, select_one

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


_QUIT = "__quit__"


def _label_for(entry: CommandEntry) -> str:
    """Render an entry's ``group name`` (or bare ``name``) invocation label."""
    return f"{entry.group} {entry.name}" if entry.group else entry.name


def _run_command(selected: str, extra_args: list[str]) -> None:
    """Run *selected* as a subcommand and return once it has finished.

    ``standalone_mode=False`` keeps click from calling ``sys.exit`` on the
    command's behalf, and swallowing ``SystemExit`` turns a command that ends
    itself -- :func:`~partio.cli.output.fail`, a "nothing to do" exit -- into a
    return to the menu rather than the end of the session.  The command has
    already reported whatever it exited over, so nothing is printed here.

    ``Abort`` (ctrl-c outside a prompt) is deliberately left to propagate: it is
    the one interrupt that should still stop partio.
    """
    with contextlib.suppress(SystemExit):
        app(selected.split() + extra_args, standalone_mode=False)


def _show_picker() -> None:
    """Display the arrow-key command menu and dispatch the choice.

    Loops rather than running once: pressing esc during the argument
    walkthrough steps back out to this menu instead of abandoning the session,
    and a finished command returns here too, so the session ends only when the
    user asks it to.
    """
    commands = discover()

    console.print(Panel("[bold]partio[/bold] -- audio tooling CLI", style="cyan", expand=False))

    options = [
        Option(title=entry.name, value=_label_for(entry), help=entry.help, group=entry.group)
        for entry in commands
    ]
    options.append(Option(title="quit", value=_QUIT, help="Exit partio."))
    labels = [_label_for(entry) for entry in commands]

    while True:
        selected = select_one("Pick a command", options, console=console)
        if selected is None or isinstance(selected, GoBack) or selected == _QUIT:
            # Nothing to go back to at the top level, so esc quits like ctrl-c.
            raise typer.Exit(code=0)

        entry = commands[labels.index(selected)]
        extra_args = prompt_for_args(entry)
        if isinstance(extra_args, GoBack):
            continue
        if extra_args is None:
            raise typer.Exit(code=0)

        # Re-invoke the app with the chosen subcommand (the prompt= fallback on
        # each Option still covers direct terminal invocation).
        _run_command(selected, extra_args)
        console.print()


# -- callback (runs on every invocation) ------------------------------------


@app.callback(invoke_without_command=True)
def _callback(
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
        console.print(f"partio {pkg_version('partio')}")
        raise typer.Exit(code=0)

    if ctx.invoked_subcommand is None:
        _show_picker()


def main() -> None:
    """Run the command-line interface."""
    app()
