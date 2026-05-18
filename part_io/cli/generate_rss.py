"""Generate a podcast RSS feed from an unmatched references JSON manifest.

Reads ``downloads/unmatched.json`` and writes one ``.rss`` file per podcast
to ``--output-dir``.  Enclosure URLs are built from ``--base-url``; filenames
match the sanitised names produced by ``download_unmatched`` (slashes and
colons in episode IDs are replaced with underscores).
"""

from __future__ import annotations

import argparse
import json
import sys
from email.utils import formatdate
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


def _safe_id(episode_id: str) -> str:
    """Return a filesystem-safe version of *episode_id* (same as download_unmatched)."""
    return episode_id.replace("/", "_").replace(":", "_")


def _rfc2822(iso: str) -> str:
    """Convert an ISO-8601 UTC string to an RFC 2822 date string."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return formatdate(dt.timestamp(), usegmt=True)
    except ValueError:
        return iso


def _build_channel(pod: dict, base_url: str) -> ET.Element:
    name: str = pod["name"]
    channel = ET.Element("channel")
    ET.SubElement(channel, "title").text = name
    ET.SubElement(channel, "link").text = base_url
    ET.SubElement(channel, "description").text = f"Generated feed for {name}"
    for ref in pod.get("unmatched_references", []):
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = ref.get("title", "")
        ET.SubElement(item, "guid").text = ref["id"]
        ET.SubElement(item, "pubDate").text = _rfc2822(ref.get("pub_date", ""))
        ET.SubElement(item, "description").text = ref.get("description", "")
        filename = f"{_safe_id(ref['id'])}.mp3"
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", f"{base_url}/{filename}")
        enclosure.set("type", "audio/mpeg")
    return channel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RSS feeds from an unmatched references JSON manifest."
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
        help="Directory to write <slug>.rss files into",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="s3://Media/Podcasts/Morbid",
        help="Base URL for enclosure hrefs (no trailing slash)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.manifest.exists():
        parser.exit(2, f"Manifest not found: {args.manifest}\n")

    podcasts: list[dict] = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pod in podcasts:
        slug: str = pod["slug"]
        rss_root = ET.Element("rss")
        rss_root.set("version", "2.0")
        rss_root.append(_build_channel(pod, args.base_url))

        tree = ET.ElementTree(rss_root)
        ET.indent(tree, space="  ")
        dest = args.output_dir / f"{slug}.rss"
        tree.write(dest, encoding="unicode", xml_declaration=True)
        sys.stderr.write(f"Wrote {dest}\n")


if __name__ == "__main__":
    main()
