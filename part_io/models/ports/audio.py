"""Callable ports for audio human-in-the-loop review."""

from __future__ import annotations

from collections.abc import Callable

AuditorFn = Callable[[float, float, str], bool]


__all__ = ["AuditorFn"]
