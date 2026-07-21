"""Tests for cli.select: arrow-key selection with a numbered non-TTY fallback."""

from __future__ import annotations

from unittest.mock import patch

from rich.console import Console

from partio.cli.select import Option, select_one

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
        select_mock.return_value.ask.return_value = None
        assert select_one("Pick", _options(), console=CONSOLE) is None


def test_tty_inserts_group_separators() -> None:
    """Each new group contributes a separator heading above its rows."""
    import questionary

    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
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
        select_mock.return_value.ask.return_value = "audio bootstrap"
        select_one("Pick", _options(), console=CONSOLE)

    choices = select_mock.call_args.kwargs["choices"]
    title = next(c for c in choices if getattr(c, "value", None) == "audio bootstrap").title

    assert title[0][1].startswith("bootstrap")
    assert "Locate a jingle." in title[1][1]


def test_option_without_help_renders_bare_title() -> None:
    """An option with no help text produces a single-fragment title."""
    with (
        patch("partio.cli.select.sys.stdin.isatty", return_value=True),
        patch("partio.cli.select.sys.stdout.isatty", return_value=True),
        patch("partio.cli.select.questionary.select") as select_mock,
    ):
        select_mock.return_value.ask.return_value = "q"
        select_one("Pick", [Option(title="quit", value="q")], console=CONSOLE)

    choices = select_mock.call_args.kwargs["choices"]
    assert choices[0].title == [("", "quit")]
