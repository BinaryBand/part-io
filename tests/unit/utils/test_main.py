"""Tests for the utils layer."""

from __future__ import annotations

from part_io.utils import main


def test_utils_main_imports() -> None:
    assert main is not None
