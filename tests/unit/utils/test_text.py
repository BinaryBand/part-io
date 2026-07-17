"""Tests for the slugify text helper."""

from __future__ import annotations

import pytest

from partio.utils.text import slugify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Green River Killer: Part 1!", "the-green-river-killer-part-1"),
        ("  Trim & Collapse   spaces  ", "trim-collapse-spaces"),
        ("already-a-slug", "already-a-slug"),
        ("MiXeD CaSe", "mixed-case"),
    ],
)
def test_slugify_normalizes(text, expected):
    """slugify() lowercases, hyphenates, collapses runs, and trims."""
    assert slugify(text) == expected


def test_slugify_uses_fallback_for_empty_result():
    """slugify() returns the fallback when there is no alphanumeric content."""
    assert slugify("!!!") == "untitled"
    assert slugify("", fallback="episode") == "episode"
