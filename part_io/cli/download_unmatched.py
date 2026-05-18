"""Download unmatched episode audio files from a JSON manifest.

Reads ``downloads/unmatched.json``, streams each ``content`` URL to
``downloads/remote/{slug}/{id}.mp3``, and skips files that already exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

_CHUNK = 1 << 16  # 64 KiB read buffer


def _download_one(url: str, dest: Path, *, client: httpx.Client) -> None:
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes(_CHUNK):
                fh.write(chunk)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download unmatched episode MP3s from a JSON manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("downloads") / "unmatched.json",
        help="Path to the unmatched references JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads") / "remote",
        help="Root directory for downloaded files",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Subdirectory name under output-dir to store all downloads (overrides podcast slug)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.manifest.exists():
        parser.exit(2, f"Manifest not found: {args.manifest}\n")

    podcasts: list[dict] = json.loads(args.manifest.read_text(encoding="utf-8"))

    jobs: list[tuple[str, str, str]] = []  # (slug, safe_id, url)
    for pod in podcasts:
        slug: str = pod["slug"]
        for ref in pod.get("unmatched_references", []):
            url: str = ref.get("content", "")
            if url:
                safe_id = ref["id"].replace("/", "_").replace(":", "_")
                jobs.append((slug, safe_id, url))

    total = len(jobs)
    if total == 0:
        sys.stderr.write("No downloadable references found.\n")
        return

    skipped = 0
    done = 0
    failed = 0

    with httpx.Client(timeout=args.timeout) as client:
        for slug, ep_id, url in jobs:
            if args.root:
                dest_dir = args.output_dir / args.root
            else:
                dest_dir = args.output_dir / slug
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{ep_id}.mp3"

            if dest.exists():
                skipped += 1
                continue

            try:
                _download_one(url, dest, client=client)
                done += 1
                sys.stderr.write(f"[{done + skipped + failed}/{total}] OK   {dest.name}\n")
                sys.stderr.flush()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                sys.stderr.write(f"[{done + skipped + failed}/{total}] FAIL {dest.name}: {exc}\n")
                sys.stderr.flush()
                if dest.exists():
                    dest.unlink()

    sys.stderr.write(f"\nDone — {done} downloaded, {skipped} skipped, {failed} failed.\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
