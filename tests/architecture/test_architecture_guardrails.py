"""Guardrail tests for dependency direction and process boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Modules that are allowed to call print() or sys.exit().
PRINT_ALLOWED_PATHS = {
    ROOT / "partio" / "cli" / "main.py",
    ROOT / "partio" / "cli" / "output.py",
}
CORE_DIRS = [
    ROOT / "partio" / "adapters",
    ROOT / "partio" / "core",
    ROOT / "partio" / "app",
    ROOT / "partio" / "lib",
]


def _python_files(base: Path):
    return [path for path in base.rglob("*.py") if path.name != "__init__.py"]


def _is_print_allowed(path: Path) -> bool:
    return path in PRINT_ALLOWED_PATHS


def _is_cli_import(module: str | None) -> bool:
    return module is not None and module.startswith("partio.cli")


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
                        if alias.name.startswith("partio.cli"):
                            raise AssertionError(f"{path} imports CLI module {alias.name}")


def test_non_entrypoints_do_not_call_print_or_sys_exit() -> None:
    """Only allowed modules may print or terminate the process."""
    for path in _python_files(ROOT / "partio"):
        if _is_print_allowed(path):
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Name) and node.func.id == "print":
                raise AssertionError(f"{path} calls print() outside an allowed module")

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "exit"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sys"
            ):
                raise AssertionError(f"{path} calls sys.exit() outside an allowed module")
