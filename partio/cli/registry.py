"""Central registry for CLI commands.

Commands are registered via the :func:`command` decorator at their
definition site.  :func:`discover` imports every module under
``cli.commands`` so their decorators fire, then returns the accumulated
list so that ``main.py`` can assemble the Typer app without hand-wiring
each entry.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandEntry:
    """One registered CLI command."""

    name: str
    group: str | None
    help: str
    fn: Callable[..., Any]


_registry: list[CommandEntry] = []


def command(
    name: str,
    verb: str | None = None,
    *,
    help: str = "",  # noqa: A002 -- mirrors Typer/Click parameter name
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a CLI command function.

    For grouped commands::

        @command("audio", "search", help="Find repeated occurrences of an audio sample.")
        def search(...) -> None:
            ...

    For root (flat) commands::

        @command("health-check", help="Check system health.")
        def health_check(...) -> None:
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if verb is not None:
            # Grouped command: command("audio", "search", help=...)
            _registry.append(CommandEntry(name=verb, group=name, help=help, fn=fn))
        else:
            # Root command: command("health-check", help=...)
            _registry.append(CommandEntry(name=name, group=None, help=help, fn=fn))
        return fn

    return decorator


def discover() -> list[CommandEntry]:
    """Import every module under ``cli.commands`` so ``@command`` runs.

    Returns the accumulated registry.  Uses ``pkgutil.walk_packages`` for
    recursive discovery; no ``try/except`` around imports (satisfies the
    ``no-guarded-imports`` ast-grep rule).  A missing or broken command
    module fails loudly at startup.
    """
    from partio.cli import commands

    for info in pkgutil.walk_packages(commands.__path__, f"{commands.__name__}."):
        importlib.import_module(info.name)
    return get_commands()


def get_commands() -> list[CommandEntry]:
    """Return all registered commands (copy)."""
    return list(_registry)


__all__ = ["CommandEntry", "command", "discover", "get_commands"]
