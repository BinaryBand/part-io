"""Unit tests for part_io.cli.generate_rss."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from part_io.cli.generate_rss import _build_channel, _build_parser, _rfc2822, _safe_id, main


class TestSafeId:
    def test_replaces_slashes(self):
        assert _safe_id("a/b/c") == "a_b_c"

    def test_replaces_colons(self):
        assert _safe_id("a:b:c") == "a_b_c"

    def test_mixed(self):
        assert _safe_id("2024/01:ep") == "2024_01_ep"

    def test_no_special_chars(self):
        assert _safe_id("episode001") == "episode001"


class TestRfc2822:
    def test_valid_iso_utc(self):
        result = _rfc2822("2024-01-15T10:00:00Z")
        assert "2024" in result
        assert "Jan" in result

    def test_valid_iso_offset(self):
        result = _rfc2822("2024-06-01T12:00:00+00:00")
        assert "2024" in result

    def test_invalid_falls_back_to_input(self):
        result = _rfc2822("not-a-date")
        assert result == "not-a-date"


class TestBuildChannel:
    def _sample_pod(self):
        return {
            "name": "Test Podcast",
            "slug": "test-pod",
            "unmatched_references": [
                {
                    "id": "ep/001",
                    "title": "Episode 1",
                    "pub_date": "2024-01-01T00:00:00Z",
                    "description": "First episode",
                    "content": "https://example.com/ep1.mp3",
                }
            ],
        }

    def test_channel_title(self):
        channel = _build_channel(self._sample_pod(), "https://cdn.example.com")
        title = channel.find("title")
        assert title is not None and title.text == "Test Podcast"

    def test_item_count(self):
        channel = _build_channel(self._sample_pod(), "https://cdn.example.com")
        items = channel.findall("item")
        assert len(items) == 1

    def test_enclosure_url_uses_safe_id(self):
        channel = _build_channel(self._sample_pod(), "https://cdn.example.com")
        enclosure = channel.find("item/enclosure")
        assert enclosure is not None
        assert "ep_001.mp3" in enclosure.get("url", "")

    def test_enclosure_type(self):
        channel = _build_channel(self._sample_pod(), "https://cdn")
        enclosure = channel.find("item/enclosure")
        assert enclosure is not None
        assert enclosure.get("type") == "audio/mpeg"

    def test_empty_references(self):
        pod = {"name": "Empty", "slug": "empty", "unmatched_references": []}
        channel = _build_channel(pod, "https://cdn")
        assert channel.findall("item") == []


class TestBuildParser:
    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.manifest == Path("downloads") / "unmatched.json"
        assert args.output_dir == Path("downloads") / "remote"

    def test_custom_args(self, tmp_path):
        parser = _build_parser()
        args = parser.parse_args([
            "--manifest", str(tmp_path / "m.json"),
            "--output-dir", str(tmp_path / "out"),
            "--base-url", "https://example.com",
        ])
        assert args.base_url == "https://example.com"


class TestMain:
    def _write_manifest(self, path: Path) -> None:
        data = [
            {
                "name": "My Show",
                "slug": "my-show",
                "unmatched_references": [
                    {
                        "id": "ep:1",
                        "title": "Ep 1",
                        "pub_date": "2024-01-01T00:00:00Z",
                        "description": "Desc",
                        "content": "https://example.com/ep1.mp3",
                    }
                ],
            }
        ]
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_writes_rss_file(self, tmp_path):
        manifest = tmp_path / "unmatched.json"
        self._write_manifest(manifest)
        out_dir = tmp_path / "out"

        import sys
        from unittest.mock import patch

        with patch("sys.argv", [
            "generate_rss",
            "--manifest", str(manifest),
            "--output-dir", str(out_dir),
            "--base-url", "https://cdn.example.com",
        ]):
            main()

        rss_file = out_dir / "my-show.rss"
        assert rss_file.exists()
        tree = ET.parse(rss_file)
        root = tree.getroot()
        assert root.tag == "rss"

    def test_exits_when_manifest_missing(self, tmp_path):
        import sys
        from unittest.mock import patch

        with patch("sys.argv", [
            "generate_rss",
            "--manifest", str(tmp_path / "missing.json"),
            "--output-dir", str(tmp_path),
        ]):
            with pytest.raises(SystemExit):
                main()
