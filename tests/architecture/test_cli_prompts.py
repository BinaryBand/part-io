"""Guardrail test: every required CLI option must carry a prompt= fallback.

Iterates every :class:`~part_io.cli.registry.CommandEntry` from
:func:`~part_io.cli.registry.discover`, re-uses
:func:`~part_io.cli.prompting.required_options` to find required args, and
asserts each one's ``OptionInfo.prompt`` is truthy.  This turns "remember to
add ``prompt=``" into an enforced rule that automatically covers new commands.
"""

from __future__ import annotations

from part_io.cli.prompting import required_options
from part_io.cli.registry import discover


def _command_label(entry) -> str:
    return f"{entry.group} {entry.name}" if entry.group else entry.name


def test_all_required_options_have_prompt() -> None:
    """Every required option must specify a prompt= fallback."""
    for entry in discover():
        for flag_name, _inner_type, option_info in required_options(entry.fn):
            assert option_info.prompt, (
                f"Command {_command_label(entry)!r} required option {flag_name!r} "
                f"is missing a prompt= fallback"
            )
