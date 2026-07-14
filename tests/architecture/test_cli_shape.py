"""Conformance test enforcing the standard CLI shape.

Asserts, over :func:`~part_io.cli.registry.discover`, that every registered
command satisfies the naming, help, docstring, and reachability rules.  This
test is the actual enforcer -- it makes the CLI shape portable across projects.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from part_io.cli.main import app
from part_io.cli.registry import discover

ROOT = Path(__file__).resolve().parents[2]

_KEBAB_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def _command_label(entry) -> str:
    return f"{entry.group} {entry.name}" if entry.group else entry.name


def test_all_entries_have_nonempty_help() -> None:
    """Every registered command must carry a non-empty help string."""
    for entry in discover():
        assert entry.help.strip(), f"Command {_command_label(entry)!r} has an empty help string"


def test_all_entry_names_are_lower_kebab() -> None:
    """Every entry name (the leaf) must be lower-kebab-case."""
    for entry in discover():
        pattern = _KEBAB_RE
        assert pattern.match(entry.name), (
            f"Command name {entry.name!r} (group={entry.group!r}) is not lower-kebab-case"
        )


def test_grouped_commands_have_verb_last() -> None:
    """Grouped command leaves should be a single verb segment."""
    for entry in discover():
        if entry.group:
            assert "-" not in entry.name, (
                f"Grouped command {entry.group} {entry.name!r} should be a single verb (no hyphens)"
            )


def test_command_functions_have_docstrings() -> None:
    """Every command function must have a non-empty docstring."""
    for entry in discover():
        doc = entry.fn.__doc__
        fn_name = getattr(entry.fn, "__name__", "<unknown>")
        assert doc is not None, (
            f"Command {_command_label(entry)!r} function {fn_name} has no docstring"
        )
        assert doc.strip(), (
            f"Command {_command_label(entry)!r} function {fn_name} has empty docstring"
        )


def test_no_command_module_calls_print_or_sys_exit() -> None:
    """Command modules must not call print() or sys.exit() directly."""
    commands_dir = ROOT / "part_io" / "cli" / "commands"
    for path in commands_dir.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                raise AssertionError(
                    f"{path.relative_to(ROOT)} calls print() -- use output.emit() instead"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "exit"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sys"
            ):
                raise AssertionError(
                    f"{path.relative_to(ROOT)} calls sys.exit() -- use output.fail() instead"
                )


def test_all_registered_commands_are_reachable() -> None:
    """Every registered command must be reachable from the assembled app."""
    registered = {_command_label(e) for e in discover()}

    # Walk the Typer command tree to find all reachable commands.
    reachable: set[str] = set()

    for cmd_info in app.registered_commands:
        if cmd_info.name:
            reachable.add(cmd_info.name)

    for group_info in app.registered_groups:
        group_name = group_info.name
        sub = group_info.typer_instance
        if sub is None:
            continue
        for cmd_info in sub.registered_commands:
            if cmd_info.name:
                reachable.add(f"{group_name} {cmd_info.name}")

    missing = registered - reachable
    assert not missing, f"Commands registered but not reachable from app: {missing}"
