"""Process execution adapter: resolve executables and run subprocesses safely.

Callable ports implemented here are defined in partio.core.ports. Executable
names are resolved to an absolute path using ``shutil.which`` before invoking
``subprocess.run``. This satisfies linters that warn about starting processes
with a partial executable path and makes failures clearer when the executable
is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from os import getenv
from pathlib import Path
from typing import Any, cast


def resolve_executable(name: str) -> str:
    """Return an absolute path to *name* if it exists and is executable.

    If *name* already contains a path separator it is treated as a path
    and validated. Otherwise ``shutil.which`` is used. Symlinks are kept
    intact: dereferencing a venv's ``bin/python`` link would escape the
    virtual environment.
    """
    if os.sep in name or name.startswith("."):
        path_obj = Path(name).expanduser()
        if path_obj.exists() and os.access(path_obj, os.X_OK):
            return os.path.abspath(path_obj)  # noqa: PTH100 - resolve() would dereference symlinks
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


def run_resolved(
    cmd: Iterable[str],
    /,
    **kwargs: Any,  # noqa: ANN401 - kwargs forwarded to subprocess.run
) -> subprocess.CompletedProcess[Any]:
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
    return cast(
        "subprocess.CompletedProcess[Any]",
        subprocess.run(full_cmd, check=False, **kwargs),
    )


__all__ = ["resolve_executable", "run_resolved"]
