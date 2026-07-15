"""Tests for cli.prompting: per-arg walkthrough from the picker."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import typer

from part_io.cli.prompting import prompt_for_args, required_options
from part_io.cli.registry import CommandEntry

# -- stub command functions for introspection tests --------------------------


def _stub_search(
    ctx: typer.Context,
    source: Annotated[
        Path, typer.Option("--source", prompt="Source audio file", help="Longer audio file.")
    ],
    sample: Annotated[
        Path, typer.Option("--sample", prompt="Reference sample", help="Reference sample.")
    ],
    threshold: Annotated[float, typer.Option(help="Match score threshold.")] = 0.8,
) -> None:
    """Stub search command."""


def _stub_all_types(
    ctx: typer.Context,
    name: Annotated[str, typer.Option("--name", help="A name.")],
    count: Annotated[int, typer.Option("--count", help="A count.")],
    ratio: Annotated[float, typer.Option("--ratio", help="A ratio.")],
    flag: Annotated[bool, typer.Option("--flag", help="A boolean flag.")],
    path: Annotated[Path, typer.Option("--path", help="A path.")],
) -> None:
    """Stub command with all supported types."""


def _stub_no_required(
    ctx: typer.Context,
    threshold: Annotated[float, typer.Option(help="Threshold.")] = 0.5,
    name: Annotated[str, typer.Option(help="Name.")] = "default",
) -> None:
    """Stub command where every option has a default."""


# -- tests for required_options ---------------------------------------------


def test_required_options_finds_required_args() -> None:
    """Required options (no default) should be discovered with correct metadata."""
    results = required_options(_stub_search)
    flags = [f for f, _, _ in results]
    assert "--source" in flags
    assert "--sample" in flags
    assert len(results) == 2


def test_required_options_skips_optional_args() -> None:
    """Options with defaults should not appear in the results."""
    results = required_options(_stub_search)
    flags = [f for f, _, _ in results]
    assert "--threshold" not in flags


def test_required_options_skips_ctx() -> None:
    """typer.Context parameters must be ignored."""
    results = required_options(_stub_search)
    for flag, _, _ in results:
        assert flag != "ctx"


def test_required_options_extracts_correct_types() -> None:
    """Each result should carry the inner (unwrapped) type."""
    results = required_options(_stub_all_types)
    type_map = {f: t for f, t, _ in results}
    assert type_map["--name"] is str
    assert type_map["--count"] is int
    assert type_map["--ratio"] is float
    assert type_map["--flag"] is bool
    assert type_map["--path"] is Path


def test_required_options_extracts_help_text() -> None:
    """OptionInfo.help should be propagated."""
    results = required_options(_stub_search)
    help_map = {f: info.help for f, _, info in results}
    assert help_map["--source"] == "Longer audio file."
    assert help_map["--sample"] == "Reference sample."


def test_required_options_no_required_args() -> None:
    """A function with only optional args should yield an empty list."""
    assert required_options(_stub_no_required) == []


def test_required_options_synthesises_flag_name() -> None:
    """When no explicit --flag is provided, one is synthesised from the param name."""

    def _fn(
        ctx: typer.Context,
        my_arg: Annotated[str, typer.Option(help="Synthesised.")],
    ) -> None:
        """Stub."""

    results = required_options(_fn)
    assert len(results) == 1
    assert results[0][0] == "--my-arg"


# -- tests for prompt_for_args ----------------------------------------------


def test_prompt_for_args_returns_flat_flag_list() -> None:
    """prompt_for_args should return ["--flag", "value", ...] pairs."""
    entry = CommandEntry(name="search", group="audio", help="Search.", fn=_stub_search)
    with (
        patch("part_io.cli.prompting.Prompt.ask", return_value="./source.mp3"),
        patch("part_io.cli.prompting.Console.print"),
    ):
        result = prompt_for_args(entry)

    assert len(result) == 4
    assert result == ["--source", "./source.mp3", "--sample", "./source.mp3"]
    for i in range(0, len(result), 2):
        assert result[i].startswith("--")


def test_prompt_for_args_no_required_args() -> None:
    """Commands with no required options should return an empty list."""
    entry = CommandEntry(name="stub", group=None, help="Stub.", fn=_stub_no_required)
    assert prompt_for_args(entry) == []


def test_prompt_for_args_uses_correct_prompt_for_int() -> None:
    """IntPrompt.ask should be called for int-typed required options."""
    entry = CommandEntry(name="all-types", group="test", help="All types.", fn=_stub_all_types)
    with (
        patch("part_io.cli.prompting.IntPrompt.ask", return_value=42),
        patch("part_io.cli.prompting.Prompt.ask", side_effect=["name", "./path"]),
        patch("part_io.cli.prompting.FloatPrompt.ask", return_value=1.5),
        patch("part_io.cli.prompting.Confirm.ask", return_value=True),
        patch("part_io.cli.prompting.Console.print"),
    ):
        result = prompt_for_args(entry)

    assert "--count" in result
    count_idx = result.index("--count")
    assert result[count_idx + 1] == "42"


def test_prompt_for_args_uses_confirm_for_bool() -> None:
    """Confirm.ask should be called for bool-typed required options."""
    entry = CommandEntry(name="all-types", group="test", help="All types.", fn=_stub_all_types)
    with (
        patch("part_io.cli.prompting.Prompt.ask", side_effect=["name", "./path"]),
        patch("part_io.cli.prompting.IntPrompt.ask", return_value=1),
        patch("part_io.cli.prompting.FloatPrompt.ask", return_value=0.5),
        patch("part_io.cli.prompting.Confirm.ask", return_value=True),
        patch("part_io.cli.prompting.Console.print"),
    ):
        result = prompt_for_args(entry)

    assert "--flag" in result
    flag_idx = result.index("--flag")
    assert result[flag_idx + 1] == "True"


def test_prompt_for_args_uses_float_prompt_for_float() -> None:
    """FloatPrompt.ask should be called for float-typed required options."""
    entry = CommandEntry(name="all-types", group="test", help="All types.", fn=_stub_all_types)
    with (
        patch("part_io.cli.prompting.Prompt.ask", side_effect=["name", "./path"]),
        patch("part_io.cli.prompting.IntPrompt.ask", return_value=1),
        patch("part_io.cli.prompting.FloatPrompt.ask", return_value=3.14),
        patch("part_io.cli.prompting.Confirm.ask", return_value=True),
        patch("part_io.cli.prompting.Console.print"),
    ):
        result = prompt_for_args(entry)

    assert "--ratio" in result
    ratio_idx = result.index("--ratio")
    assert result[ratio_idx + 1] == "3.14"
