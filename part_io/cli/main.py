"""cli.main: the command-line interface.

Keep it thin -- parse arguments, call into app, format results. Typer is the
standard framework: declare commands with @app.command() and describe
arguments/options with typing.Annotated so `ty` sees real signatures. The
`hello` command below is a worked example -- replace or delete it.
"""

from __future__ import annotations

from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def hello(name: Annotated[str, typer.Argument(help="Who to greet.")]) -> None:
    """Greet someone by name -- worked example, replace or delete."""
    typer.echo(f"Hello, {name}")


def main() -> None:
    """Run the command-line interface."""
    app()
