"""cli.main: the command-line interface.

When invoked with no arguments ``part-io`` presents a numbered picker of
available commands.  Individual commands are registered via Typer's
``@app.command()`` / ``app.add_typer()`` APIs so ``ty`` can inspect the
real signatures.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False, invoke_without_command=True)
console = Console()

_COMMANDS: list[tuple[str, str]] = [
    ("audio-search", "Find repeated occurrences of an audio sample in a longer file."),
    ("audio-locate", "Locate the single best occurrence of an audio sample."),
    ("audio-review", "Generate review bundles (clips + manifest) for manual labeling."),
    ("audio-bootstrap", "Interactively locate a jingle and write a seed clip."),
]


# -- subcommand registration (imported here to avoid circular deps) --------
from part_io.cli.audio_bootstrap import (  # noqa: E402
    bootstrap as _bootstrap_cmd,
)
from part_io.cli.audio_locate import locate as _locate_cmd  # noqa: E402
from part_io.cli.audio_review import review as _review_cmd  # noqa: E402
from part_io.cli.audio_search import search as _search_cmd  # noqa: E402

app.command("audio-search")(_search_cmd)
app.command("audio-locate")(_locate_cmd)
app.command("audio-review")(_review_cmd)
app.command("audio-bootstrap")(_bootstrap_cmd)


# -- picker ----------------------------------------------------------------


def _show_picker() -> None:
    """Display a numbered menu and dispatch the selected command."""
    console.print(Panel("[bold]part-io[/bold] — audio tooling CLI", style="cyan", expand=False))
    for idx, (name, desc) in enumerate(_COMMANDS, start=1):
        console.print(f"  [bold]{idx}[/bold]. [green]{name}[/green]  — {desc}")
    console.print()

    try:
        choice = input("Pick a command [1-4]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\nAborted.", style="yellow")
        raise typer.Exit(code=0) from None

    if not choice:
        console.print("No selection.", style="yellow")
        raise typer.Exit(code=0)

    # Accept a number or a command name.
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(_COMMANDS):
            selected = _COMMANDS[idx][0]
        else:
            console.print(f"Invalid choice: {choice}", style="red")
            raise typer.Exit(code=1)
    elif choice in {name for name, _ in _COMMANDS}:
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
