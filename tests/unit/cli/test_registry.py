"""Tests for the cli.registry module."""

from __future__ import annotations

from part_io.cli.registry import CommandEntry, command, get_commands


def test_command_decorator_adds_entry() -> None:
    """A function decorated with @command should appear in the registry."""

    @command("test-fake-cmd", help="A fake command for testing.")
    def _fake_cmd() -> None:  # vulture: ignore
        pass

    after = get_commands()
    names = {e.name for e in after}
    assert "test-fake-cmd" in names
    # Clean up so other tests aren't polluted.
    from part_io.cli import registry as _reg

    _reg._registry[:] = [e for e in _reg._registry if e.name != "test-fake-cmd"]


def test_get_commands_returns_copy() -> None:
    """Mutating the return value of get_commands() must not affect the registry."""
    cmds = get_commands()
    cmds.clear()
    assert len(get_commands()) > 0


def test_registry_entry_is_frozen_dataclass() -> None:
    """CommandEntry instances must be immutable."""
    entry = CommandEntry(name="x", help="y", fn=lambda: None)
    try:
        entry.name = "z"  # ty: ignore[invalid-assignment]
    except AttributeError:
        return  # expected
    msg = "CommandEntry should be frozen"
    raise AssertionError(msg)
