"""Small helpers to resolve executables and run subprocesses safely.

These helpers resolve the executable name to an absolute path using
``shutil.which`` before invoking ``subprocess.run``. This satisfies
linters that warn about starting processes with a partial executable
path and makes failures clearer when the executable is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from os import getenv
from pathlib import Path
from typing import Any, Callable, Iterable, cast


def resolve_executable(name: str) -> str:
    """Return an absolute path to *name* if it exists and is executable.

    If *name* already contains a path separator it is treated as a path
    and validated. Otherwise ``shutil.which`` is used.
    """
    if os.sep in name or name.startswith("."):
        path_obj = Path(name).expanduser()
        if path_obj.exists() and os.access(path_obj, os.X_OK):
            # Keep symlink paths (e.g. .venv/bin/python) to preserve venv behavior.
            return str(path_obj.absolute())
        raise FileNotFoundError(f"Executable not found or not executable: {name}")

    path = shutil.which(name)
    if not path:
        if os.name == "nt":
            win_get_links = Path(getenv("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
            candidate = win_get_links / f"{name}.exe"
            if candidate.exists():
                return str(candidate)
            win_get_root = Path(getenv("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
            for found in win_get_root.rglob(f"{name}.exe"):
                if found.is_file():
                    return str(found)
        raise FileNotFoundError(f"Executable not found in PATH: {name}")
    return path


def run_resolved(cmd: Iterable[str], /, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Resolve the command's executable and call ``subprocess.run``.

    The first element of *cmd* is resolved with :func:`resolve_executable`.
    All other positional and keyword arguments are forwarded to
    :func:`subprocess.run`.
    """
    cmd_list: list[str] = list(cmd)
    if not cmd_list:
        raise ValueError("empty command")

    executable = cmd_list[0]
    resolved = resolve_executable(executable)
    full_cmd = [resolved, *cmd_list[1:]]
    return cast(subprocess.CompletedProcess[Any], subprocess.run(full_cmd, **kwargs))  # noqa: S603


def launch_resolved(cmd: Iterable[str], /, **kwargs: Any) -> "subprocess.Popen[Any]":
    """Resolve the command's executable and launch a persistent subprocess.

    The first element of *cmd* is resolved with :func:`resolve_executable`.
    Returns a :class:`subprocess.Popen` object for long-running processes.
    """
    cmd_list: list[str] = list(cmd)
    if not cmd_list:
        raise ValueError("empty command")

    executable = cmd_list[0]
    resolved = resolve_executable(executable)
    full_cmd = [resolved, *cmd_list[1:]]
    return subprocess.Popen(full_cmd, **kwargs)  # noqa: S603


def run_resolved_with_stderr_callback(
    cmd: Iterable[str],
    /,
    *,
    on_stderr_line: Callable[[str], None] | None = None,
) -> tuple[int, bytes]:
    """Run a resolved command, capturing stdout while optionally streaming stderr.

    Returns a tuple of ``(returncode, stdout_bytes)``. When ``on_stderr_line``
    is provided, stderr is consumed line-by-line in a background thread and
    each decoded line is forwarded to the callback.
    """
    proc = launch_resolved(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    callback_thread: threading.Thread | None = None

    if on_stderr_line is not None and proc.stderr is not None:

        def _read_stderr() -> None:
            stderr = proc.stderr
            if stderr is None:
                return
            for raw in stderr:
                on_stderr_line(raw.decode(errors="replace").strip())

        callback_thread = threading.Thread(target=_read_stderr, daemon=True)
        callback_thread.start()

    stdout_data = proc.stdout.read() if proc.stdout is not None else b""
    proc.wait()

    if callback_thread is not None:
        callback_thread.join(timeout=2)

    return proc.returncode, stdout_data


__all__ = [
    "resolve_executable",
    "run_resolved",
    "launch_resolved",
    "run_resolved_with_stderr_callback",
]
