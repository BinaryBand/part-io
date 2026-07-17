"""Small text-normalization helpers."""

from __future__ import annotations

import re

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, fallback: str = "untitled") -> str:
    """Return a filesystem-safe, lowercase, hyphen-separated slug of *text*.

    Runs of non-alphanumeric characters collapse to a single hyphen and leading
    and trailing hyphens are trimmed. Returns *fallback* when *text* has no
    alphanumeric content.

    >>> slugify("The Green River Killer: Part 1!")
    'the-green-river-killer-part-1'
    """
    slug = _NON_SLUG.sub("-", text.lower()).strip("-")
    return slug or fallback


__all__ = ["slugify"]
