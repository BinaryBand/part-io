"""Tests for the app layer."""

from __future__ import annotations

from part_io.app import main


def test_app_main_imports() -> None:
    assert main is not None
