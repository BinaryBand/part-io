"""Tests for cli.select: arrow-key selection with a numbered non-TTY fallback."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import questionary
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from partio.cli.select import GO_BACK, GoBack, Option, select_many, select_one

CONSOLE = Console()


def _options() -> list[Option[str]]:
    return [
        Option(title="bootstrap", value="audio bootstrap", help="Locate a jingle.", group="audio"),
        Option(title="locate", value="audio locate", help="Locate a sample.", group="audio"),
        Option(title="list", value="library list", help="List paths.", group="library"),
    ]


# -- fallback (non-TTY) behaviour -------------------------------------------


def test_select_one_empty_options_returns_none() -> None:
    """An empty option list yields None without prompting."""
    assert select_one("Pick", [], console=CONSOLE) is None


def test_fallback_returns_chosen_value() -> None:
    """Outside a TTY the numbered prompt maps the index back to its value."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=False),
        patch("partio.cli.select.Prompt.ask", return_value="2"),
        patch("rich.console.Console.print"),
    ):
        assert select_one("Pick", _options(), console=CONSOLE) == "audio locate"


def test_fallback_cancel_returns_none() -> None:
    """Ctrl-C / EOF at the numbered prompt cancels the selection."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=False),
        patch("partio.cli.select.Prompt.ask", side_effect=KeyboardInterrupt),
        patch("rich.console.Console.print"),
    ):
        assert select_one("Pick", _options(), console=CONSOLE) is None


def test_fallback_used_when_stdout_is_not_a_tty() -> None:
    """A TTY stdin but piped stdout still uses the numbered fallback."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=False),
        patch("partio.cli.select.Prompt.ask", return_value="1"),
        patch("rich.console.Console.print"),
    ):
        assert select_one("Pick", _options(), console=CONSOLE) == "audio bootstrap"


# -- arrow-key (TTY) behaviour ----------------------------------------------


def test_tty_uses_questionary_select() -> None:
    """On a TTY the questionary arrow-key menu drives the choice."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "library list"
        result = select_one("Pick", _options(), console=CONSOLE)

    assert result == "library list"
    assert select_mock.call_args.args[0] == "Pick"


def test_tty_cancel_returns_none() -> None:
    """questionary returns None when the user hits ctrl-c; that propagates."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = None
        assert select_one("Pick", _options(), console=CONSOLE) is None


def test_tty_inserts_group_separators() -> None:
    """Each new group contributes a separator heading above its rows."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "audio bootstrap"
        select_one("Pick", _options(), console=CONSOLE)

    choices = select_mock.call_args.kwargs["choices"]
    # Separator subclasses Choice, so filter it out before reading real values.
    separators = [c for c in choices if isinstance(c, questionary.Separator)]
    values = [c.value for c in choices if not isinstance(c, questionary.Separator)]

    assert values == ["audio bootstrap", "audio locate", "library list"]
    # Two group headings ("audio", "library") plus one blank spacer row.
    assert len(separators) == 3


def test_tty_titles_carry_dimmed_help() -> None:
    """Titles render as (style, text) pairs: padded name then dimmed help."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "audio bootstrap"
        select_one("Pick", _options(), console=CONSOLE)

    choices = select_mock.call_args.kwargs["choices"]
    title = next(c for c in choices if getattr(c, "value", None) == "audio bootstrap").title

    assert title[0][1].startswith("bootstrap")
    assert "Locate a jingle." in title[1][1]


def test_long_titles_do_not_starve_the_metadata_column() -> None:
    """A very long title is clipped so its date/size metadata still renders."""
    options = [
        Option(
            title="Zohran Mamdani Knows He Has Political Capital. And He Intends to Spend It.",
            value="a",
            help="2026-07-19   38.8 MB",
        )
    ]
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch(
            "partio.cli.select.shutil.get_terminal_size",
            return_value=os.terminal_size((110, 24)),
        ),
        patch("partio.cli.select.questionary.checkbox") as checkbox,
    ):
        checkbox.return_value.application.key_bindings = KeyBindings()
        checkbox.return_value.ask.return_value = []
        select_many("Select episodes to download", options, console=CONSOLE)

    title = checkbox.call_args.kwargs["choices"][0].title
    assert title[0][1].rstrip().endswith("...")  # title clipped
    assert "2026-07-19   38.8 MB" in title[1][1]  # metadata intact
    assert len("".join(text for _style, text in title)) <= 110


def test_short_rows_keep_their_full_help() -> None:
    """When everything fits, neither column is truncated."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch(
            "partio.cli.select.shutil.get_terminal_size",
            return_value=os.terminal_size((100, 24)),
        ),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "audio bootstrap"
        select_one("Pick a command", _options(), console=CONSOLE)

    title = select_mock.call_args.kwargs["choices"][1].title
    assert title[0][1].startswith("bootstrap")
    assert "Locate a jingle." in title[1][1]
    assert "..." not in title[1][1]


# -- esc / go back -----------------------------------------------------------


def _press_escape(prompt_mock) -> MagicMock:
    """Fire the esc binding registered on a mocked questionary prompt."""
    bindings = prompt_mock.return_value.application.key_bindings
    binding = next(b for b in bindings.bindings if getattr(b.keys[0], "value", None) == "escape")
    event = MagicMock()
    event.app.is_done = False
    binding.handler(event)
    return event


def test_esc_binding_is_registered_on_select() -> None:
    """select_one binds esc so the caller can tell "back" from "cancel"."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "audio locate"
        select_one("Pick", _options(), console=CONSOLE)

    event = _press_escape(select_mock)
    event.app.exit.assert_called_once_with(result=GO_BACK)


def test_esc_binding_is_registered_on_checkbox() -> None:
    """select_many binds esc the same way."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.checkbox") as checkbox,
    ):
        checkbox.return_value.application.key_bindings = KeyBindings()
        checkbox.return_value.ask.return_value = []
        select_many("Pick", _options(), console=CONSOLE)

    event = _press_escape(checkbox)
    event.app.exit.assert_called_once_with(result=GO_BACK)


def test_esc_is_not_bound_eagerly() -> None:
    """A non-eager binding lets prompt_toolkit still assemble arrow-key sequences."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "audio locate"
        select_one("Pick", _options(), console=CONSOLE)

    bindings = select_mock.return_value.application.key_bindings
    binding = next(b for b in bindings.bindings if getattr(b.keys[0], "value", None) == "escape")
    assert binding.eager() is False


def test_repeat_escape_during_teardown_is_ignored() -> None:
    """A second esc while the prompt is already finishing must not re-exit."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "audio locate"
        select_one("Pick", _options(), console=CONSOLE)

    bindings = select_mock.return_value.application.key_bindings
    binding = next(b for b in bindings.bindings if getattr(b.keys[0], "value", None) == "escape")
    event = MagicMock()
    event.app.is_done = True
    binding.handler(event)

    event.app.exit.assert_not_called()


def test_go_back_sentinel_is_falsy_and_readable() -> None:
    """GO_BACK is falsy so a missed check cannot pass for a real answer."""
    assert not GO_BACK
    assert repr(GO_BACK) == "GO_BACK"
    assert isinstance(GO_BACK, GoBack)


def test_multi_select_empty_options_returns_empty_list() -> None:
    """Nothing to check yields an empty selection, not None (which means cancelled)."""
    assert select_many("Pick", [], console=CONSOLE) == []


def test_multi_select_tty_returns_checked_values() -> None:
    """On a TTY the questionary checkbox drives the selection."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.checkbox") as checkbox,
    ):
        checkbox.return_value.application.key_bindings = KeyBindings()
        checkbox.return_value.ask.return_value = ["audio locate", "library list"]
        result = select_many("Pick", _options(), console=CONSOLE)

    assert result == ["audio locate", "library list"]


def test_multi_select_passes_disabled_reason_through() -> None:
    """A disabled option reaches questionary with its reason attached."""
    options = [
        Option(title="new", value="a"),
        Option(title="done", value="b", disabled="already in library"),
    ]
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.checkbox") as checkbox,
    ):
        checkbox.return_value.application.key_bindings = KeyBindings()
        checkbox.return_value.ask.return_value = []
        select_many("Pick", options, console=CONSOLE)

    choices = checkbox.call_args.kwargs["choices"]
    assert choices[0].disabled is None
    assert choices[1].disabled == "already in library"


def test_multi_select_fallback_parses_comma_separated_indexes() -> None:
    """Outside a TTY, comma-separated numbers select the matching rows."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=False),
        patch("partio.cli.select.Prompt.ask", return_value="1, 3"),
        patch("rich.console.Console.print"),
    ):
        result = select_many("Pick", _options(), console=CONSOLE)

    assert result == ["audio bootstrap", "library list"]


def test_multi_select_fallback_ignores_junk_and_out_of_range() -> None:
    """Non-numeric or out-of-range tokens are skipped rather than raising."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=False),
        patch("partio.cli.select.Prompt.ask", return_value="2, nope, 99, "),
        patch("rich.console.Console.print"),
    ):
        result = select_many("Pick", _options(), console=CONSOLE)

    assert result == ["audio locate"]


def test_multi_select_fallback_refuses_disabled_rows() -> None:
    """A disabled row cannot be selected by index in the fallback either."""
    options = [
        Option(title="new", value="a"),
        Option(title="done", value="b", disabled="already in library"),
    ]
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=False),
        patch("partio.cli.select.Prompt.ask", return_value="1,2"),
        patch("rich.console.Console.print"),
    ):
        result = select_many("Pick", options, console=CONSOLE)

    assert result == ["a"]


def test_multi_select_fallback_cancel_returns_none() -> None:
    """Ctrl-C at the fallback prompt cancels rather than selecting nothing."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=False),
        patch("partio.cli.select.Prompt.ask", side_effect=KeyboardInterrupt),
        patch("rich.console.Console.print"),
    ):
        assert select_many("Pick", _options(), console=CONSOLE) is None


def test_option_without_help_renders_bare_title() -> None:
    """An option with no help text produces a single-fragment title."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.application.key_bindings = KeyBindings()
        select_mock.return_value.ask.return_value = "q"
        select_one("Pick", [Option(title="quit", value="q")], console=CONSOLE)

    choices = select_mock.call_args.kwargs["choices"]
    assert choices[0].title == [("", "quit")]
