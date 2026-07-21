"""Pure path arithmetic for where a feed episode lands on disk."""

from __future__ import annotations

from pathlib import Path

from partio.core.models import FeedEpisode
from partio.core.text import slugify


def destination_for(episode: FeedEpisode, *, dest_dir: Path) -> Path:
    """Return where *episode* would land inside *dest_dir*.

    The library is virtual: an episode is offered before anything is
    downloaded, so its destination has to be knowable without I/O.  Deriving it
    from the title also makes the answer stable, which is what lets a second
    look tell "already downloaded" from "not yet".
    """
    return dest_dir / f"{slugify(episode.title)}.mp3"


__all__ = ["destination_for"]
