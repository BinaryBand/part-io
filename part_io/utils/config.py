from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

_tomllib: Any = None
try:
    import tomllib as _tomllib  # type: ignore[no-redef]
except ImportError:
    try:
        import tomli as _tomllib  # type: ignore[no-redef]
    except ImportError:
        pass

_CONFIG_CACHE: Dict[str, Any] | None = None


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while True:
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            return Path.cwd()
        p = p.parent


def _config_path() -> Path:
    return _repo_root() / "config" / "part_io.toml"


def load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    # Prefer pyproject.toml [tool.part_io] when available; fallback to
    # config/part_io.toml for backwards compatibility.
    repo = _repo_root()
    pyproject = repo / "pyproject.toml"
    if pyproject.exists() and _tomllib is not None:
        try:
            with pyproject.open("rb") as fh:
                data = _tomllib.load(fh)
        except Exception:
            data = {}
        else:
            # Expect structure: {"tool": {"part_io": { ... }}}
            tool = data.get("tool", {})
            part_cfg = tool.get("part_io") if isinstance(tool, dict) else None
            if isinstance(part_cfg, dict):
                _CONFIG_CACHE = part_cfg
                return _CONFIG_CACHE

    # Fallback to legacy config file
    cfg_file = _config_path()
    if cfg_file.exists() and _tomllib is not None:
        try:
            with cfg_file.open("rb") as fh:
                data = _tomllib.load(fh)
        except Exception:
            data = {}
    else:
        data = {}

    _CONFIG_CACHE = data
    return _CONFIG_CACHE


def get_profile_cache_dir(remote_dir: Path | None = None) -> Path:
    """Return the configured profile cache directory.

    Resolution order:
    - Environment variable `PART_IO_PROFILE_CACHE_DIR` if set
    - `defaults.profile_cache_dir` from `config/part_io.toml` if present
      (relative paths are resolved against the repo root)
    - If `remote_dir` is provided, use `remote_dir.parent / '.profile_cache'`
    - Otherwise fall back to `<repo_root>/downloads/.profile_cache`
    """
    env = os.getenv("PART_IO_PROFILE_CACHE_DIR")
    if env:
        return Path(env).expanduser()

    cfg = load_config()
    default_val = None
    if isinstance(cfg, dict):
        default_val = cfg.get("defaults", {}).get("profile_cache_dir")

    if default_val:
        p = Path(default_val)
        if not p.is_absolute():
            p = _repo_root() / p
        return p

    if remote_dir is not None:
        return remote_dir.parent / ".profile_cache"

    return _repo_root() / "downloads" / ".profile_cache"


__all__ = ["load_config", "get_profile_cache_dir"]
