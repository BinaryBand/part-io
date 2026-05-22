"""CLI: regenerate .profile.toml for every audio file in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from part_io.adapters.audio.snippet_profile import write_snippet_profile

_AUDIO_EXTENSIONS = frozenset({".mp3", ".opus"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overwrite .profile.toml for every audio snippet in DIRECTORY."
    )
    parser.add_argument("directory", type=Path, help="Directory containing snippet audio files")
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    snippets = sorted(p for p in directory.iterdir() if p.suffix.lower() in _AUDIO_EXTENSIONS)
    if not snippets:
        print(f"No audio files found in {directory}")
        sys.exit(0)

    failures = 0
    for path in snippets:
        print(f"  {path.name} ...", end=" ", flush=True)
        try:
            out = write_snippet_profile(path)
            print(f"{out.stat().st_size:,} bytes")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}", file=sys.stderr)
            failures += 1

    written = len(snippets) - failures
    print(f"\n{written}/{len(snippets)} profile(s) written.")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
