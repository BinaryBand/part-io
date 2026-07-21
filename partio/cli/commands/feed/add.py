"""CLI for remembering a podcast feed."""

from __future__ import annotations

import uuid
from typing import Annotated

import httpx
import typer

from partio.adapters.feed import fetch_feed_title
from partio.cli.library import feed_store
from partio.cli.output import _json_flag, emit, fail
from partio.cli.registry import command
from partio.core.ports import FeedEntry


@command("feed", "add", help="Remember a podcast feed.")
def add(
    ctx: typer.Context,
    url: Annotated[
        str,
        typer.Option("--url", prompt="Feed URL", help="Podcast RSS/Atom feed URL."),
    ],
    label: Annotated[
        str | None,
        typer.Option(help="Friendly name (defaults to the feed's own title)."),
    ] = None,
) -> None:
    """Remember a podcast feed so its episodes join the library.

    The feed is fetched once to confirm it resolves and to pick up its title,
    so a typo fails here rather than at download time.  Nothing is downloaded:
    every episode simply becomes selectable, and only the ones actually chosen
    are ever fetched.
    """
    store = feed_store()
    if any(entry.url == url for entry in store.list_items()):
        fail(ValueError(f"Feed already remembered: {url}"))

    resolved_label = label
    if resolved_label is None:
        try:
            resolved_label = fetch_feed_title(url)
        except httpx.HTTPError as exc:
            fail(exc)

    entry = FeedEntry(id=uuid.uuid4().hex[:8], url=url, label=resolved_label or url)
    store.add_item(entry)
    emit(f"Remembered feed {entry.label} as {entry.id}", as_json=_json_flag(ctx))
