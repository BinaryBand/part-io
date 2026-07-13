"""Ports: the Protocol interfaces that adapters must satisfy.

A port is defined here in core; its concrete implementation lives in adapters
and is wired to the application in app. Because core may not import adapters,
the dependency always points inward. `ty` verifies each adapter structurally
satisfies its port at the point where app wires them together.

The Clock below is a worked example of the pattern -- replace or delete it.
"""

from __future__ import annotations

from typing import Protocol


class Clock(Protocol):
    """A source of the current time, injected so tests can supply a fake."""

    def now(self) -> float:
        """Return the current time as a POSIX timestamp."""
