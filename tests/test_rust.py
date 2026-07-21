"""Conditional gate: enforce a fixed shape for Rust once a rust/ tree appears.

Skips entirely for pure-Python projects with no rust/ directory. The moment
one exists, this asserts Cargo.toml/*.rs never leak outside rust/ and that the
cargo toolchain (fmt, clippy, test) is clean -- the Rust equivalent of
ruff/ty/vulture.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUST = ROOT / "rust"


def _rust_or_skip() -> Path:
    if not RUST.exists():
        pytest.skip("no rust/ tree")
    if shutil.which("cargo") is None:
        pytest.fail("rust/ present but cargo not on PATH; the Rust shape is unverifiable")
    return RUST


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=RUST, check=False)


def test_rust_location() -> None:
    """Cargo.toml and *.rs must live only under rust/, never elsewhere in the project."""
    _rust_or_skip()
    offenders: list[str] = []
    for path in sorted({*ROOT.rglob("Cargo.toml"), *ROOT.rglob("*.rs")}):
        parts = path.relative_to(ROOT).parts
        if any(part.startswith(".") for part in parts):
            continue
        if RUST not in path.parents:
            offenders.append(str(path.relative_to(ROOT)))
    listing = "\n".join(offenders)
    assert not offenders, f"Rust files found outside rust/:\n\n{listing}"


def test_rust_format() -> None:
    """cargo fmt --check must report no reformats needed."""
    _rust_or_skip()
    result = _run(["cargo", "fmt", "--all", "--", "--check"])
    assert result.returncode == 0, (
        f"cargo fmt --check failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


def test_rust_clippy() -> None:
    """cargo clippy must produce zero warnings."""
    _rust_or_skip()
    result = _run(["cargo", "clippy", "--all-targets", "--all-features", "--", "-D", "warnings"])
    assert result.returncode == 0, (
        f"cargo clippy failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )


def test_rust_test() -> None:
    """cargo test must pass."""
    _rust_or_skip()
    result = _run(["cargo", "test"])
    assert result.returncode == 0, (
        f"cargo test failed (exit {result.returncode}):\n\n{result.stdout}\n{result.stderr}"
    )
