"""Audio subprocess lifecycle manager.

This is a small, test-friendly manager for launching and cleaning up
audio player subprocesses (ffplay). It intentionally mirrors the
responsibilities of the prior ad-hoc `_AUDIO_IO` dict, but encapsulates
resource handling so callers can rely on deterministic cleanup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from subprocess import Popen  # nosemgrep: no-direct-subprocess-import-except-in-utils

from part_io.utils.exec import launch_resolved


class AudioProcessManager:
    """Manage launched audio processes and associated file handles.

    Example:
        mgr = AudioProcessManager()
        proc = mgr.start_player(path)
        mgr.stop(proc)
        mgr.stop_all()
    """

    def __init__(self) -> None:
        """Create a new manager with an empty registry."""
        self._store: dict[int, tuple[Popen, tuple[Any, ...]]] = {}

    def start_player(self, path: Path, *, args: list[str] | None = None) -> Popen:
        """Start an `ffplay` process to play *path* and register its handles.

        Returns the `Popen` instance for the launched process.
        """
        stdin_f = open("/dev/null", "rb")
        stdout_f = open("/dev/null", "wb")
        stderr_f = open("/dev/null", "wb")
        # Pre-path args (e.g. -ss/-t) go before the filename so ffplay uses
        # fast input-side seeking rather than decoding up to the target point.
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if args:
            cmd = [*cmd, *args]
        cmd.append(str(path))
        try:
            proc = launch_resolved(cmd, stdin=stdin_f, stdout=stdout_f, stderr=stderr_f)
        except Exception:
            stdin_f.close()
            stdout_f.close()
            stderr_f.close()
            raise
        self._store[id(proc)] = (proc, (stdin_f, stdout_f, stderr_f))
        return proc

    def stop(self, proc: Popen | None) -> None:
        """Stop *proc* if running and close associated handles."""
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                logging.debug("Failed to kill process %s", proc, exc_info=True)

        entry = self._store.pop(id(proc), None)
        handles = entry[1] if entry is not None else None
        if handles is not None:
            for h in handles:
                try:
                    h.close()
                except Exception:
                    logging.debug("Failed to close handle: %s", h, exc_info=True)

    def stop_all(self) -> None:
        """Stop and cleanup all registered processes and handles."""
        for proc, handles in list(self._store.values()):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    logging.debug("Failed to kill process %s", proc, exc_info=True)

            for h in handles:
                try:
                    h.close()
                except Exception:
                    logging.debug("Failed to close handle: %s", h, exc_info=True)

        self._store.clear()
