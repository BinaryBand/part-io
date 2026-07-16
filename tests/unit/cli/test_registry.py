"""Tests for the cli.registry module."""

from __future__ import annotations

from partio.cli.registry import CommandEntry, command, discover, get_commands


def test_command_decorator_adds_grouped_entry() -> None:
    """A function decorated with @command(group, verb) should appear in the registry."""

    @command("testgroup", "testverb", help="A fake grouped command for testing.")
    def _fake_cmd() -> None:  # vulture: ignore
        pass

    after = get_commands()
    entries = [e for e in after if e.group == "testgroup" and e.name == "testverb"]
    assert len(entries) == 1
    # Clean up so other tests aren't polluted.
    from partio.cli import registry as _reg

    _reg._registry[:] = [
        e for e in _reg._registry if not (e.group == "testgroup" and e.name == "testverb")
    ]


def test_command_decorator_adds_root_entry() -> None:
    """A function decorated with @command(name) (no verb) should be a root entry."""

    @command("test-root-cmd", help="A fake root command for testing.")  # vulture: ignore
    def _fake_root() -> None:
        pass

    after = get_commands()
    entries = [e for e in after if e.name == "test-root-cmd"]
    assert len(entries) == 1
    assert entries[0].group is None
    # Clean up.
    from partio.cli import registry as _reg

    _reg._registry[:] = [e for e in _reg._registry if e.name != "test-root-cmd"]


def test_get_commands_returns_copy() -> None:
    """Mutating the return value of get_commands() must not affect the registry."""
    cmds = get_commands()
    cmds.clear()
    assert len(get_commands()) > 0


def test_registry_entry_is_frozen_dataclass() -> None:
    """CommandEntry instances must be immutable."""
    entry = CommandEntry(name="x", group=None, help="y", fn=lambda: None)
    try:
        entry.name = "z"  # ty: ignore[invalid-assignment]
    except AttributeError:
        return  # expected
    msg = "CommandEntry should be frozen"
    raise AssertionError(msg)


def test_discover_returns_audio_commands() -> None:
    """discover() should find the four audio commands."""
    entries = discover()
    names = {(e.group, e.name) for e in entries}
    assert ("audio", "search") in names
    assert ("audio", "locate") in names
    assert ("audio", "review") in names
    assert ("audio", "bootstrap") in names
