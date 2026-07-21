"""CLI for listing the library: remembered feeds and the audio they offer."""

from __future__ import annotations

from itertools import groupby

import typer
from rich.console import Console

from partio.cli.library import MARK_LEGEND, Track, feeds, tracks
from partio.cli.output import ExitCode, _json_flag, emit
from partio.cli.registry import command

# The spinner is drawn on stderr so it never mingles with --json output.
console = Console(stderr=True)

# A long episode title must not push its date/size column off the line.
_MAX_LABEL_WIDTH = 60
# A back catalogue runs to thousands of episodes; a listing nobody can scroll
# is worse than a short one.  The pickers still offer every episode, and
# --json still emits the lot.
_MAX_PER_GROUP = 20


@command("feed", "list", help="List remembered feeds and the episodes they offer.")
def list_feeds(ctx: typer.Context) -> None:
    """List every remembered feed alongside the episodes it offers.

    Episodes are listed whether or not they have been downloaded, because that
    is what the pickers offer; the glyph says which ones are already here.
    This is the one place that reads each feed in full -- showing everything is
    the product here, where in a prompt it would just be a wait.
    """
    as_json = _json_flag(ctx)
    remembered = feeds()
    with console.status("Reading feeds"):
        available = tracks(full=True)
    if not remembered and not available:
        emit("Nothing in the library yet -- add a feed with `partio feed add`.", as_json=as_json)
        raise SystemExit(ExitCode.NO_RESULT)

    if as_json:
        emit(
            {
                "feeds": [{"id": e.id, "label": e.label, "url": e.url} for e in remembered],
                "tracks": [_as_dict(track) for track in available],
            },
            as_json=True,
        )
        return
    emit(_render(available))


def _as_dict(track: Track) -> dict:
    return {
        "label": track.label,
        "path": str(track.path),
        "kind": track.kind.value,
        "group": track.group,
        "on_disk": track.on_disk,
    }


def _render(available: list[Track]) -> list[str]:
    """Render the listing grouped by feed, newest first, truncated per feed."""
    if not available:
        return ["No episodes available -- the remembered feeds returned nothing."]

    width = min(max(len(track.label) for track in available), _MAX_LABEL_WIDTH)
    lines = [MARK_LEGEND]
    for group, group_tracks in groupby(available, key=lambda track: track.group):
        shown = list(group_tracks)
        lines.extend(["", group])
        lines.extend(_row(track, width) for track in shown[:_MAX_PER_GROUP])
        hidden = len(shown) - _MAX_PER_GROUP
        if hidden > 0:
            lines.append(f"    ... and {hidden} more")
    return lines


def _row(track: Track, width: int) -> str:
    """One listing row: availability glyph, label, then the dimmed detail."""
    label = track.label if len(track.label) <= width else track.label[: width - 3] + "..."
    return f"  {track.mark} {label:<{width}}  {track.detail}".rstrip()
