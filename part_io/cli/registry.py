"""Central registry for CLI commands.

Commands are registered via the :func:`command` decorator at their
definition site.  :func:`get_commands` returns the accumulated list so
that ``main.py`` can assemble the Typer app without hand-wiring each
entry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandEntry:
    """One registered CLI command."""

    name: str
    help: str
    fn: Callable[..., Any]


_registry: list[CommandEntry] = []


def command(
    name: str,
    *,
    help: str = "",  # noqa: A002 — mirrors Typer/Click parameter name
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a CLI command function.

    Usage::

        @command("search-audio", help="Find repeated occurrences of an audio sample.")
        def search(...) -> None:
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _registry.append(CommandEntry(name=name, help=help, fn=fn))
        return fn

    return decorator


def get_commands() -> list[CommandEntry]:
    """Return all registered commands (copy)."""
    return list(_registry)


__all__ = ["CommandEntry", "command", "get_commands"]
