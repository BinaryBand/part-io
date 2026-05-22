"""Guardrail tests for dependency direction and process boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT_PATHS = {
    ROOT / "part_io" / "cli" / "tasks.py",
    ROOT / "part_io" / "cli" / "generate" / "tasks.py",
    ROOT / "part_io" / "cli" / "audio_search.py",
    ROOT / "part_io" / "cli" / "audio_review.py",
    ROOT / "part_io" / "cli" / "audio_review_batch.py",
    ROOT / "part_io" / "cli" / "audio_detect_ads.py",
    ROOT / "part_io" / "cli" / "audio_detect_ads_batch.py",
    ROOT / "part_io" / "cli" / "audio_ad_detect.py",
    ROOT / "part_io" / "cli" / "audio_ad_remove.py",
    ROOT / "part_io" / "cli" / "download_unmatched.py",
    ROOT / "part_io" / "cli" / "remote_pipeline.py",
    ROOT / "part_io" / "cli" / "remote_promote.py",
    ROOT / "part_io" / "cli" / "audio_snippet_profile.py",
}
ENTRYPOINT_DIRS = {
    ROOT / "part_io" / "cli" / "lint",
}
CORE_DIRS = [
    ROOT / "part_io" / "adapters",
    ROOT / "part_io" / "models",
    ROOT / "part_io" / "services",
    ROOT / "part_io" / "utils",
]


def _python_files(base: Path):
    return [path for path in base.rglob("*.py") if path.name != "__init__.py"]


def _is_entrypoint(path: Path) -> bool:
    if path in ENTRYPOINT_PATHS:
        return True
    return any(parent == entry_dir for entry_dir in ENTRYPOINT_DIRS for parent in path.parents)


def _is_cli_import(module: str | None) -> bool:
    return module is not None and module.startswith("part_io.cli")


def test_core_modules_do_not_import_cli() -> None:
    """Core and support modules should not depend on CLI packages."""
    for base in CORE_DIRS:
        for path in _python_files(base):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and _is_cli_import(node.module):
                    raise AssertionError(f"{path} imports CLI module {node.module}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("part_io.cli"):
                            raise AssertionError(f"{path} imports CLI module {alias.name}")


def test_non_entrypoints_do_not_call_print_or_sys_exit() -> None:
    """Only entrypoint modules may print or terminate the process."""
    for path in _python_files(ROOT / "part_io"):
        if _is_entrypoint(path):
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Name) and node.func.id == "print":
                raise AssertionError(f"{path} calls print() outside an entrypoint")

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "exit"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sys"
            ):
                raise AssertionError(f"{path} calls sys.exit() outside an entrypoint")
