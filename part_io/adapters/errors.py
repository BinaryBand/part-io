"""Typed boundary errors for lint adapters and execution helpers."""

from __future__ import annotations


class LintBoundaryError(Exception):
    """Base class for lint boundary failures."""


class LintConfigError(LintBoundaryError):
    """Raised when lint config cannot be loaded or parsed."""


class LintProcessError(LintBoundaryError):
    """Raised when a lint command cannot be constructed or executed."""
