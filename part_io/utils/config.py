from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import cast

_CONFIG_CACHE: dict[str, object] | None = None


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while True:
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            return Path.cwd()
        p = p.parent


def load_config() -> dict[str, object]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    pyproject = _repo_root() / "pyproject.toml"
    if pyproject.exists():
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        tool = data.get("tool", {})
        part_cfg = tool.get("part_io") if isinstance(tool, dict) else None
        if isinstance(part_cfg, dict):
            _CONFIG_CACHE = part_cfg
            return _CONFIG_CACHE
    _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_profile_cache_dir(remote_dir: Path | None = None) -> Path:
    """Return the configured profile cache directory.

    Resolution order:
    - Environment variable ``PART_IO_PROFILE_CACHE_DIR`` if set.
    - ``defaults.profile_cache_dir`` from ``[tool.part_io]`` in pyproject.toml.
    - ``remote_dir.parent / '.profile_cache'`` if *remote_dir* is given.
    - ``<repo_root>/downloads/.profile_cache`` otherwise.
    """
    env = os.getenv("PART_IO_PROFILE_CACHE_DIR")
    if env:
        return Path(env).expanduser()

    cfg = load_config()
    defaults = cfg.get("defaults")
    default_val = (
        cast(dict[str, object], defaults).get("profile_cache_dir")
        if isinstance(defaults, dict)
        else None
    )
    if default_val:
        p = Path(str(default_val))
        if not p.is_absolute():
            p = _repo_root() / p
        return p

    if remote_dir is not None:
        return remote_dir.parent / ".profile_cache"

    return _repo_root() / "downloads" / ".profile_cache"


def get_codec_args_for_extension(ext: str) -> list[str] | None:
    """Return ffmpeg codec argument list for a given extension (with or without dot).

    Returns None when no configured codec is found.
    """
    if not ext:
        return None
    if ext.startswith("."):
        ext = ext[1:]
    cfg = load_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else None
    codecs = None
    if isinstance(defaults, dict):
        codecs = cast(dict[str, object], defaults).get("codecs")
    if not isinstance(codecs, dict):
        return None
    entry = cast(dict[str, object], codecs).get(ext)
    if not isinstance(entry, dict):
        return None
    codec = cast(dict[str, object], entry).get("codec")
    bitrate = cast(dict[str, object], entry).get("bitrate")
    if not codec:
        return None
    args: list[str] = ["-c:a", str(codec)]
    if bitrate:
        args += ["-b:a", str(bitrate)]
    return args


__all__ = ["load_config", "get_profile_cache_dir", "get_codec_args_for_extension"]
