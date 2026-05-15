"""Callable ports for process execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ResolveExecutableFn = Callable[[str], str]
RunResolvedFn = Callable[..., Any]


__all__ = ["ResolveExecutableFn", "RunResolvedFn"]
