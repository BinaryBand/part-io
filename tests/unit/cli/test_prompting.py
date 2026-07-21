"""Tests for cli.prompting: per-arg walkthrough from the picker."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import typer
from prompt_toolkit.key_binding import KeyBindings

from partio.cli.prompting import _ask, prompt_for_args, required_options
from partio.cli.registry import CommandEntry
from partio.cli.select import GO_BACK

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


# Bound to a name so ruff does not read it as a boolean positional argument.
_TRUE = True


def _as_args(result) -> list[str]:
    """Narrow a prompt_for_args result to the success case."""
    assert isinstance(result, list)
    return result


def _ask_returning(*values):
    """Patch the per-arg prompt to answer with *values* in order."""
    return patch("partio.cli.prompting._ask", side_effect=list(values))


def test_prompt_for_args_returns_flat_flag_list() -> None:
    """prompt_for_args should return ["--flag", "value", ...] pairs."""
    entry = CommandEntry(name="search", group="audio", help="Search.", fn=_stub_search)
    with (
        _ask_returning("./source.mp3", "./sample.mp3"),
        patch("partio.cli.prompting.Console.print"),
    ):
        result = _as_args(prompt_for_args(entry))
    assert result == ["--source", "./source.mp3", "--sample", "./sample.mp3"]
    for i in range(0, len(result), 2):
        assert result[i].startswith("--")


def test_prompt_for_args_no_required_args() -> None:
    """Commands with no required options should return an empty list."""
    entry = CommandEntry(name="stub", group=None, help="Stub.", fn=_stub_no_required)
    assert prompt_for_args(entry) == []


def test_prompt_for_args_stringifies_every_answer() -> None:
    """Typed answers are rendered back to strings for the Typer invocation."""
    entry = CommandEntry(name="all-types", group="test", help="All types.", fn=_stub_all_types)
    with (
        _ask_returning("bob", 42, 3.14, _TRUE, "./path"),
        patch("partio.cli.prompting.Console.print"),
    ):
        result = _as_args(prompt_for_args(entry))

    assert result[result.index("--count") + 1] == "42"
    assert result[result.index("--ratio") + 1] == "3.14"
    assert result[result.index("--flag") + 1] == "True"
    assert result[result.index("--name") + 1] == "bob"


# -- esc / go back -----------------------------------------------------------


def test_esc_steps_back_to_the_previous_option() -> None:
    """esc on the 2nd option re-asks the 1st, and the re-answer is the one kept."""
    entry = CommandEntry(name="search", group="audio", help="Search.", fn=_stub_search)
    answers = ["./first.mp3", GO_BACK, "./corrected.mp3", "./sample.mp3"]
    with _ask_returning(*answers), patch("partio.cli.prompting.Console.print"):
        result = _as_args(prompt_for_args(entry))
    assert result == ["--source", "./corrected.mp3", "--sample", "./sample.mp3"]


def test_esc_on_the_first_option_returns_go_back() -> None:
    """Nothing left to step back to, so the caller is told to go back."""
    entry = CommandEntry(name="search", group="audio", help="Search.", fn=_stub_search)
    with _ask_returning(GO_BACK), patch("partio.cli.prompting.Console.print"):
        assert prompt_for_args(entry) is GO_BACK


def test_cancel_returns_none() -> None:
    """ctrl-c abandons the whole walkthrough rather than stepping back."""
    entry = CommandEntry(name="search", group="audio", help="Search.", fn=_stub_search)
    with _ask_returning("./source.mp3", None), patch("partio.cli.prompting.Console.print"):
        assert prompt_for_args(entry) is None


# -- _ask dispatch -----------------------------------------------------------


def test_ask_uses_confirm_for_bool() -> None:
    """bool options get a yes/no confirm, with esc bound."""
    with patch("partio.cli.prompting.questionary.confirm") as confirm:
        confirm.return_value.application.key_bindings = KeyBindings()
        confirm.return_value.ask.return_value = True
        assert _ask(bool, "--flag") is True


def test_ask_casts_numbers() -> None:
    """int/float options are cast back from the text prompt."""
    with patch("partio.cli.prompting.questionary.text") as text:
        text.return_value.application.key_bindings = KeyBindings()
        text.return_value.ask.return_value = "42"
        assert _ask(int, "--count") == 42
        text.return_value.ask.return_value = "3.5"
        assert _ask(float, "--ratio") == 3.5


def test_ask_number_validator_rejects_junk() -> None:
    """The numeric prompt validates before accepting, instead of crashing later."""
    with patch("partio.cli.prompting.questionary.text") as text:
        text.return_value.application.key_bindings = KeyBindings()
        text.return_value.ask.return_value = "7"
        _ask(int, "--count")

    validate = text.call_args.kwargs["validate"]
    assert validate("12") is True
    assert isinstance(validate("banana"), str)


def test_ask_propagates_go_back_from_a_number_prompt() -> None:
    """esc at a numeric prompt is not mistaken for an unparsable answer."""
    with patch("partio.cli.prompting.questionary.text") as text:
        text.return_value.application.key_bindings = KeyBindings()
        text.return_value.ask.return_value = GO_BACK
        assert _ask(int, "--count") is GO_BACK


# -- library path picker -----------------------------------------------------


def _entry(label: str, path: str):
    from partio.core.ports import AudioPathEntry, AudioPathKind

    return AudioPathEntry(id=label, path=Path(path), label=label, kind=AudioPathKind.SOURCE)


def test_prompt_path_offers_library_entries_to_the_picker() -> None:
    """Remembered entries become picker options, and the chosen value is used."""
    entry = CommandEntry(name="search", group="audio", help="Search.", fn=_stub_search)
    library = [_entry("Ep A", "static/downloads/a.mp3"), _entry("Ep B", "static/downloads/b.mp3")]
    with (
        patch("partio.cli.prompting._library_entries", return_value=library),
        patch(
            "partio.cli.prompting.select_one", return_value="static/downloads/b.mp3"
        ) as select_mock,
        patch("partio.cli.prompting.Console.print"),
    ):
        result = _as_args(prompt_for_args(entry))
    assert result == [
        "--source",
        "static/downloads/b.mp3",
        "--sample",
        "static/downloads/b.mp3",
    ]
    options = select_mock.call_args.args[1]
    assert [o.value for o in options[:2]] == ["static/downloads/a.mp3", "static/downloads/b.mp3"]
    assert [o.title for o in options[:2]] == ["Ep A", "Ep B"]
    assert options[-1].title == "enter a path manually"


def test_prompt_path_empty_library_asks_for_a_path() -> None:
    """With nothing remembered, the picker is skipped for a path prompt."""
    with (
        patch("partio.cli.prompting._library_entries", return_value=[]),
        patch("partio.cli.prompting.questionary.path") as path_mock,
    ):
        path_mock.return_value.application.key_bindings = KeyBindings()
        path_mock.return_value.ask.return_value = "/typed.mp3"
        assert _ask(Path, "--source") == "/typed.mp3"


def test_prompt_path_custom_falls_back_to_free_text() -> None:
    """Choosing "enter a path manually" prompts for a typed path."""
    from partio.cli.prompting import _CUSTOM_PATH_CHOICE

    with (
        patch("partio.cli.prompting._library_entries", return_value=[_entry("Ep A", "a.mp3")]),
        patch("partio.cli.prompting.select_one", return_value=_CUSTOM_PATH_CHOICE),
        patch("partio.cli.prompting.questionary.path") as path_mock,
    ):
        path_mock.return_value.application.key_bindings = KeyBindings()
        path_mock.return_value.ask.return_value = "/my/own.mp3"
        assert _ask(Path, "--source") == "/my/own.mp3"


def test_esc_at_the_manual_path_prompt_returns_to_the_picker() -> None:
    """esc while typing a path reopens the library picker, not the previous arg."""
    from partio.cli.prompting import _CUSTOM_PATH_CHOICE

    library = [_entry("Ep A", "a.mp3")]
    # First pass: pick "manual", press esc. Second pass: pick the library entry.
    with (
        patch("partio.cli.prompting._library_entries", return_value=library),
        patch(
            "partio.cli.prompting.select_one", side_effect=[_CUSTOM_PATH_CHOICE, "a.mp3"]
        ) as select_mock,
        patch("partio.cli.prompting.questionary.path") as path_mock,
    ):
        path_mock.return_value.application.key_bindings = KeyBindings()
        path_mock.return_value.ask.return_value = GO_BACK
        result = _ask(Path, "--source")

    assert result == "a.mp3"
    assert select_mock.call_count == 2


def test_prompt_path_cancelled_picker_propagates_none() -> None:
    """ctrl-c at the library picker abandons rather than falling through."""
    with (
        patch("partio.cli.prompting._library_entries", return_value=[_entry("Ep A", "a.mp3")]),
        patch("partio.cli.prompting.select_one", return_value=None),
    ):
        assert _ask(Path, "--source") is None
