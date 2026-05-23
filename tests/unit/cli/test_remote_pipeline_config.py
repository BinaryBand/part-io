from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w

from part_io.adapters.audio.snippet_profile import SnippetProfileModel, encode_matrix
from part_io.cli import remote_pipeline as rp
from part_io.cli.remote_pipeline import _load_snippets


def test_load_snippets_rejects_seed_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "__config__.toml"

    config_path.write_text(
        tomli_w.dumps(
            {
                "snippets": [
                    {
                        "name": "open",
                        "seed_path": "downloads/snippets/open.mp3",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        _load_snippets(config_path)


def test_load_snippets_supports_inline_profiles(tmp_path: Path) -> None:
    config_path = tmp_path / "__config__.toml"
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    config_path.write_text(
        tomli_w.dumps(
            {
                "snippets": [
                    {
                        "name": "open",
                        "profile": {
                            "source_hash": "abc123",
                            "n_frames": int(matrix.shape[0]),
                            "analysis_rate": 16000,
                            "hop_size": 1024,
                            "band_count": 1,
                            "data": encode_matrix(matrix),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_snippets(config_path)

    assert "open" in loaded.profiles
    np.testing.assert_allclose(loaded.profiles["open"], matrix)


def test_config_init_writes_profile_only_config(tmp_path: Path, monkeypatch) -> None:
    open_seed = tmp_path / "open.mp3"
    close_seed = tmp_path / "close.mp3"
    open_seed.write_bytes(b"open")
    close_seed.write_bytes(b"close")

    def fake_snapshot(_path: Path) -> SnippetProfileModel:
        return SnippetProfileModel(
            source_hash="abc123",
            n_frames=2,
            analysis_rate=16000,
            hop_size=1024,
            band_count=1,
            data="encoded",
        )

    monkeypatch.setattr(rp, "snapshot_snippet_profile", fake_snapshot)

    args = argparse.Namespace(
        remote_dir=tmp_path,
        config=None,
        open_seed=open_seed,
        close_seed=close_seed,
        intro_seed=None,
        outro_seed=None,
        force=False,
    )

    rp._cmd_config_init(args)

    config_path = tmp_path / "__config__.toml"
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert len(parsed["snippets"]) == 2
    assert all("seed_path" not in snippet for snippet in parsed["snippets"])
    assert all("profile" in snippet for snippet in parsed["snippets"])
