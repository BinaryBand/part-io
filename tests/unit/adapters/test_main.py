"""Tests for the adapters layer."""

from __future__ import annotations

from part_io.adapters import main


def test_adapters_main_imports() -> None:
    assert main is not None
