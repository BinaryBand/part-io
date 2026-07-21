"""The library: every episode of every remembered feed, downloaded or not.

There is one audio concept in partio and this is it.  A *feed* is how audio
enters the library; a *track* is one piece of audio in it, whether or not the
bytes are on this machine yet.  Callers ask for :func:`tracks` and hand the
chosen one to :func:`ensure_local`, which downloads it only if it has to -- so
every picker can offer a whole back catalogue while the disk stays as empty as
the user leaves it.

The on-disk index (``static/library.json``) is a cache, not a user-managed
list: it is written here and by ``audio bootstrap``, and no command exposes it.
"""

from partio.cli.library._cache import cached, remember
from partio.cli.library._feeds import feed_store, feeds
from partio.cli.library._fetch import ensure_local
from partio.cli.library._tracks import MARK_LEGEND, Track, has_more, refresh, tracks

__all__ = [
    "MARK_LEGEND",
    "Track",
    "cached",
    "ensure_local",
    "feed_store",
    "feeds",
    "has_more",
    "refresh",
    "remember",
    "tracks",
]
