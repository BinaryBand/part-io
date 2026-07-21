"""Tests for the cli.commands.audio.bootstrap module."""

from __future__ import annotations

import pytest

from partio.cli.commands.audio import bootstrap as audio_bootstrap
from partio.cli.library import _cache as library_store
from partio.core.ports import AudioPathKind

_STUB_DURATION = 1800.0


@pytest.fixture(autouse=True)
def _library_path(tmp_path, monkeypatch):
    """Keep `audio bootstrap` from registering its seeds in the real library."""
    monkeypatch.setattr(library_store, "DEFAULT_LIBRARY_PATH", tmp_path / "library.json")


@pytest.fixture(autouse=True)
def _probed_duration(monkeypatch):
    """Stub the ffprobe duration probe backing the default search region.

    `audio bootstrap` now searches to the end of the file unless told otherwise,
    and the fixture sources here are stub bytes that no real ffprobe can read.
    """
    monkeypatch.setattr(audio_bootstrap, "audio_duration_seconds", lambda _path: _STUB_DURATION)


def test_audio_bootstrap_main_writes_seed_clip(monkeypatch, capsys, tmp_path):
    """The bootstrap CLI should locate the jingle and write a seed clip."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    output = tmp_path / "episode_seed.mp3"
    extracted: list[dict] = []

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", lambda **_kwargs: (5.0, 6.5))
    monkeypatch.setattr(
        audio_bootstrap, "extract_audio_clip", lambda **kwargs: extracted.append(kwargs)
    )

    audio_bootstrap.bootstrap(ctx=None, source=source, output=output)

    assert extracted == [
        {
            "source_path": source,
            "destination_path": output,
            "start_seconds": 5.0,
            "duration_seconds": 1.5,
        }
    ]
    assert "jingle 5.000s -> 6.500s" in capsys.readouterr().out


def test_audio_bootstrap_registers_the_seed_as_a_sample(monkeypatch, tmp_path):
    """The seed lands in the library as a SAMPLE, so --sample pickers can offer it."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    output = tmp_path / "episode_seed.mp3"

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", lambda **_kwargs: (5.0, 6.5))
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)

    audio_bootstrap.bootstrap(ctx=None, source=source, output=output)

    entries = library_store.cache_store().list_items()
    assert [(e.path, e.kind) for e in entries] == [(output, AudioPathKind.SAMPLE)]
    assert entries[0].label == "episode_seed"


def test_audio_bootstrap_does_not_duplicate_an_existing_seed(monkeypatch, tmp_path):
    """Re-running over the same destination refreshes the clip without a second entry."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    output = tmp_path / "episode_seed.mp3"

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", lambda **_kwargs: (5.0, 6.5))
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)

    audio_bootstrap.bootstrap(ctx=None, source=source, output=output)
    audio_bootstrap.bootstrap(ctx=None, source=source, output=output)

    assert len(library_store.cache_store().list_items()) == 1


def test_audio_bootstrap_registers_every_seed_of_a_multi_run(monkeypatch, tmp_path):
    """Each numbered seed from --max-occurrences is remembered separately."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")

    monkeypatch.setattr(
        audio_bootstrap, "locate_jingle_spans", lambda **_kwargs: [(1.0, 2.0), (5.0, 6.0)]
    )
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)

    audio_bootstrap.bootstrap(ctx=None, source=source, output=tmp_path / "seeds", max_occurrences=2)

    labels = [e.label for e in library_store.cache_store().list_items()]
    assert labels == ["episode_seed_01", "episode_seed_02"]


def test_audio_bootstrap_searches_the_whole_file_by_default(monkeypatch, tmp_path):
    """With no --region-end the search region runs to the probed duration."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    tuning: list[dict] = []

    def _spy(**kwargs):
        tuning.append(kwargs)
        return (5.0, 6.5)

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", _spy)
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)

    audio_bootstrap.bootstrap(ctx=None, source=source, output=tmp_path / "seed.mp3")

    assert tuning[0]["region_start"] == 0.0
    assert tuning[0]["region_end"] == _STUB_DURATION


def test_audio_bootstrap_region_end_still_narrows_the_search(monkeypatch, tmp_path):
    """An explicit --region-end wins over the probed duration."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    tuning: list[dict] = []

    def _spy(**kwargs):
        tuning.append(kwargs)
        return (5.0, 6.5)

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", _spy)
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)

    audio_bootstrap.bootstrap(
        ctx=None, source=source, output=tmp_path / "seed.mp3", region_end=90.0
    )

    assert tuning[0]["region_end"] == 90.0


def test_audio_bootstrap_main_exits_when_no_jingle_found(monkeypatch, capsys, tmp_path):
    """An empty span should print the no-jingle message and exit non-zero."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", lambda **_kwargs: None)

    with pytest.raises(SystemExit) as excinfo:
        audio_bootstrap.bootstrap(ctx=None, source=source)

    assert excinfo.value.code == 1
    assert "No jingle found in the search region." in capsys.readouterr().out


def test_audio_bootstrap_main_multi_writes_numbered_seed_clips(monkeypatch, capsys, tmp_path):
    """With --max-occurrences > 1 each located span gets a numbered seed clip."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    output_dir = tmp_path / "seeds"
    extracted: list[dict] = []

    monkeypatch.setattr(
        audio_bootstrap, "locate_jingle_spans", lambda **_kwargs: [(10.0, 20.0), (50.0, 62.0)]
    )
    monkeypatch.setattr(
        audio_bootstrap, "extract_audio_clip", lambda **kwargs: extracted.append(kwargs)
    )

    audio_bootstrap.bootstrap(
        ctx=None,
        source=source,
        output=output_dir,
        max_occurrences=3,
    )

    assert [kwargs["destination_path"] for kwargs in extracted] == [
        output_dir / "episode_seed_01.mp3",
        output_dir / "episode_seed_02.mp3",
    ]
    assert [(kwargs["start_seconds"], kwargs["duration_seconds"]) for kwargs in extracted] == [
        (10.0, 10.0),
        (50.0, 12.0),
    ]
    output = capsys.readouterr().out
    assert f"jingle 10.000s -> 20.000s written to {output_dir / 'episode_seed_01.mp3'}" in output
    assert f"jingle 50.000s -> 62.000s written to {output_dir / 'episode_seed_02.mp3'}" in output


def test_audio_bootstrap_main_multi_exits_when_no_jingle_found(monkeypatch, capsys, tmp_path):
    """An empty span list should print the no-jingle message and exit non-zero."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_spans", lambda **_kwargs: [])

    with pytest.raises(SystemExit) as excinfo:
        audio_bootstrap.bootstrap(ctx=None, source=source, max_occurrences=5)

    assert excinfo.value.code == 1
    assert "No jingle found in the search region." in capsys.readouterr().out


def test_audio_bootstrap_main_auditor_plays_segment_and_reads_input(monkeypatch, capsys, tmp_path):
    """The interactive auditor should audition via ffplay and parse the answer."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    output = tmp_path / "episode_seed.mp3"
    played: list[dict] = []

    def _fake_locate(*, auditor, **_kwargs):
        return (5.0, 6.5) if auditor(5.0, 1.5, "Is the jingle anywhere in this clip?") else None

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", _fake_locate)
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)
    monkeypatch.setattr(
        "partio.cli.commands.audio._auditor.play_audio_segment",
        lambda **kwargs: played.append(kwargs),
    )
    answers = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    audio_bootstrap.bootstrap(ctx=None, source=source, output=output)

    assert played == [{"source_path": source, "start_seconds": 5.0, "duration_seconds": 1.5}]
    assert "jingle 5.000s -> 6.500s" in capsys.readouterr().out


def test_audio_bootstrap_main_rejects_missing_source(capsys, tmp_path):
    """A missing source file should exit via fail."""
    with pytest.raises(SystemExit) as excinfo:
        audio_bootstrap.bootstrap(ctx=None, source=tmp_path / "missing.opus")

    assert excinfo.value.code == 2
    assert "Source not found" in capsys.readouterr().err
