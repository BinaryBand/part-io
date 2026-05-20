"""Project-wide lightweight exception types used by adapters and utils.

Keep these simple and importable from library code; adapters may raise them
and services can map them to user-facing messages at entrypoints.
"""

from __future__ import annotations


class ProjectError(Exception):
    """Base class for project-specific boundary errors."""


class CacheError(ProjectError):
    """Raised when on-disk cache operations fail in a non-recoverable way."""


class ExternalProcessError(ProjectError):
    """Raised when an external process fails to launch or exits unexpectedly."""


class AudioProcessError(ProjectError):
    """Raised when audio playback/streaming helper encounters an error."""
