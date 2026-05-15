"""Lint tool configuration loader adapter.

Callable ports implemented here are defined in part_io.models.ports.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from part_io.adapters.errors import LintConfigError

DEFAULT_LINT_CONFIG_PATH = Path("config/lint.toml")


def load_lint_config(
    tool_name: str,
    config_section: str | None = None,
    *,
    config_path: Path = DEFAULT_LINT_CONFIG_PATH,
) -> dict:
    """Load tool-specific config from lint TOML and return a section mapping."""
    if not config_section:
        return {}

    try:
        config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LintConfigError(
            f"Error loading {tool_name} config from {config_path}: {exc}"
        ) from exc

    return config_data.get(config_section, {})


__all__ = ["DEFAULT_LINT_CONFIG_PATH", "load_lint_config"]


