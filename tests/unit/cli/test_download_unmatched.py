"""Unit tests for part_io.cli.download_unmatched."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from part_io.cli.download_unmatched import _build_parser, _download_one, main


class TestBuildParser:
    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.manifest == Path("downloads") / "unmatched.json"
        assert args.output_dir == Path("downloads") / "remote"
        assert args.root is None
        assert args.timeout == pytest.approx(120.0)

    def test_custom_manifest(self, tmp_path):
        parser = _build_parser()
        args = parser.parse_args(["--manifest", str(tmp_path / "m.json")])
        assert args.manifest == tmp_path / "m.json"

    def test_custom_root(self):
        parser = _build_parser()
        args = parser.parse_args(["--root", "my-show"])
        assert args.root == "my-show"


class TestDownloadOne:
    def test_streams_content_to_file(self, tmp_path):
        dest = tmp_path / "ep.mp3"
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response

        _download_one("https://example.com/ep.mp3", dest, client=mock_client)

        assert dest.read_bytes() == b"chunk1chunk2"
        mock_response.raise_for_status.assert_called_once()

    def test_raises_on_http_error(self, tmp_path):
        import httpx

        dest = tmp_path / "ep.mp3"
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            _download_one("https://example.com/ep.mp3", dest, client=mock_client)


class TestMain:
    def _write_manifest(self, path: Path, jobs: list[dict] | None = None) -> None:
        if jobs is None:
            jobs = [
                {
                    "slug": "my-show",
                    "unmatched_references": [
                        {"id": "ep/001", "content": "https://example.com/ep1.mp3"},
                        {"id": "ep:002", "content": "https://example.com/ep2.mp3"},
                    ],
                }
            ]
        path.write_text(json.dumps(jobs), encoding="utf-8")

    def test_missing_manifest_exits(self, tmp_path):
        with patch(
            "sys.argv",
            [
                "download_unmatched",
                "--manifest",
                str(tmp_path / "missing.json"),
            ],
        ):
            with pytest.raises(SystemExit):
                main()

    def test_skips_existing_files(self, tmp_path, capsys):
        manifest = tmp_path / "unmatched.json"
        self._write_manifest(manifest)
        out_dir = tmp_path / "out"
        dest_dir = out_dir / "my-show"
        dest_dir.mkdir(parents=True)
        (dest_dir / "ep_001.mp3").write_bytes(b"existing")
        (dest_dir / "ep_002.mp3").write_bytes(b"existing")

        with patch(
            "sys.argv",
            [
                "download_unmatched",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out_dir),
            ],
        ):
            main()

        err = capsys.readouterr().err
        assert "2 skipped" in err

    def test_downloads_new_files(self, tmp_path, capsys):
        manifest = tmp_path / "unmatched.json"
        self._write_manifest(
            manifest,
            [
                {
                    "slug": "my-show",
                    "unmatched_references": [
                        {"id": "ep001", "content": "https://example.com/ep1.mp3"},
                    ],
                }
            ],
        )
        out_dir = tmp_path / "out"

        with patch("part_io.cli.download_unmatched._download_one") as mock_dl:
            with patch(
                "sys.argv",
                [
                    "download_unmatched",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(out_dir),
                ],
            ):
                main()

        mock_dl.assert_called_once()

    def test_custom_root_overrides_slug(self, tmp_path):
        manifest = tmp_path / "unmatched.json"
        self._write_manifest(manifest)
        out_dir = tmp_path / "out"
        (out_dir / "custom").mkdir(parents=True)
        # Pre-create files so _download_one is never called
        (out_dir / "custom" / "ep_001.mp3").write_bytes(b"x")
        (out_dir / "custom" / "ep_002.mp3").write_bytes(b"x")

        with patch(
            "sys.argv",
            [
                "download_unmatched",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out_dir),
                "--root",
                "custom",
            ],
        ):
            main()

        # Files would land under "custom" not "my-show"
        assert not (out_dir / "my-show").exists()

    def test_no_downloadable_refs_returns_early(self, tmp_path, capsys):
        manifest = tmp_path / "unmatched.json"
        manifest.write_text(
            json.dumps([{"slug": "empty", "unmatched_references": [{"id": "x", "content": ""}]}]),
            encoding="utf-8",
        )

        with patch(
            "sys.argv",
            [
                "download_unmatched",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            main()

        assert "No downloadable" in capsys.readouterr().err

    def test_failed_download_exits_1(self, tmp_path):
        manifest = tmp_path / "unmatched.json"
        self._write_manifest(
            manifest,
            [{"slug": "s", "unmatched_references": [{"id": "x", "content": "https://x.com/x"}]}],
        )

        def _fail(url, dest, *, client):
            raise OSError("network error")

        with patch("part_io.cli.download_unmatched._download_one", side_effect=_fail):
            with patch(
                "sys.argv",
                [
                    "download_unmatched",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(tmp_path / "out"),
                ],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
        assert exc_info.value.code == 1
