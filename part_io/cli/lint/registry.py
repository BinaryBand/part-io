"""Central lint tool registry and command builder catalog."""

from __future__ import annotations

from collections.abc import Callable

from part_io.models.lint import ToolSpec

Builder = Callable[[dict], list[str]]


def get_tool_spec(tool_key: str) -> ToolSpec:
    """Return the registered tool spec for *tool_key*."""
    if tool_key not in TOOL_SPECS:
        raise ValueError(f"Unknown lint tool key: {tool_key}")
    return TOOL_SPECS[tool_key]


def get_tool_builder(tool_key: str) -> Builder:
    """Return the registered command builder for *tool_key*."""
    if tool_key not in BUILDERS:
        raise ValueError(f"Unknown lint tool key: {tool_key}")
    return BUILDERS[tool_key]


def _build_coverage_cmd(cfg: dict) -> list[str]:
    floor = cfg.get("floor", 80)
    return [
        "poetry",
        "run",
        "pytest",
        "-q",
        "tests/",
        "--ignore=tests/integration/test_lint.py",
        "--ignore-glob=*/test_lint.py",
        "--cov=part_io",
        "--cov-report=term",
        f"--cov-fail-under={floor}",
    ]


def _build_cpd_cmd(cfg: dict) -> list[str]:
    config_path = cfg.get("config_path", "config/jscpd.json")
    jscpd_version = cfg.get("version", "4.0.5")
    package = f"jscpd@{jscpd_version}"
    return ["npx", "--yes", package, "--config", str(config_path)]


def _build_lizard_cmd(cfg: dict) -> list[str]:
    cmd = ["lizard"]
    if ccn := cfg.get("ccn"):
        cmd += ["--CCN", str(ccn)]
    if length := cfg.get("length"):
        cmd += ["--length", str(length)]
    if cfg.get("warnings_only"):
        cmd += ["--warnings_only"]
    cmd += cfg.get("paths", ["part_io"])
    return cmd


def _build_ruff_cmd(cfg: dict) -> list[str]:
    mode = cfg.get("mode", "check")
    cmd = ["ruff"]
    if mode == "format":
        cmd += ["format"]
        if cfg.get("line_length"):
            cmd += ["--line-length", str(cfg["line_length"])]
    else:
        cmd += ["check"]
        if cfg.get("select"):
            cmd += ["--select", cfg["select"]]
        if cfg.get("ignore"):
            cmd += ["--ignore", cfg["ignore"]]
        if cfg.get("line_length"):
            cmd += ["--line-length", str(cfg["line_length"])]
    cmd += cfg.get("paths", ["part_io", "tests"])
    return cmd


def _build_semgrep_cmd(cfg: dict) -> list[str]:
    _ = cfg
    return ["semgrep", "scan", "--config", "config/semgrep/", "--error"]


def _build_ty_cmd(cfg: dict) -> list[str]:
    _ = cfg
    return ["ty", "check"]


def _build_vulture_cmd(cfg: dict) -> list[str]:
    cmd = ["vulture"]
    if min_confidence := cfg.get("min_confidence"):
        cmd += ["--min-confidence", str(min_confidence)]
    cmd += cfg.get("paths", ["part_io", "tests"])
    return cmd


TOOL_SPECS: dict[str, ToolSpec] = {
    "coverage": ToolSpec(key="coverage", executable="pytest", config_section="coverage"),
    "cpd": ToolSpec(key="cpd", executable="npx", config_section=None),
    "lizard": ToolSpec(key="lizard", executable="lizard", config_section="lizard"),
    "ruff": ToolSpec(key="ruff", executable="ruff", config_section="ruff"),
    "semgrep": ToolSpec(key="semgrep", executable="semgrep", config_section=None),
    "ty": ToolSpec(key="ty", executable="ty", config_section=None),
    "vulture": ToolSpec(key="vulture", executable="vulture", config_section="vulture"),
}

BUILDERS: dict[str, Builder] = {
    "coverage": _build_coverage_cmd,
    "cpd": _build_cpd_cmd,
    "lizard": _build_lizard_cmd,
    "ruff": _build_ruff_cmd,
    "semgrep": _build_semgrep_cmd,
    "ty": _build_ty_cmd,
    "vulture": _build_vulture_cmd,
}


def build_tool_cmd(tool_key: str, cfg: dict) -> list[str]:
    """Build command args for a registered tool key."""
    return get_tool_builder(tool_key)(cfg)


__all__ = ["TOOL_SPECS", "BUILDERS", "get_tool_spec", "get_tool_builder", "build_tool_cmd"]


