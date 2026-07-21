"""Tests for audio CLI entrypoints."""

from __future__ import annotations

import json

import pytest

from partio.adapters.audio.matcher import AudioMatch, BestMatch
from partio.cli.commands.audio import bootstrap as audio_bootstrap
from partio.cli.commands.audio import locate as audio_locate
from partio.cli.commands.audio import review as audio_review
from partio.cli.commands.audio import search as audio_search
from partio.cli.commands.library import _store as library_store

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


def test_audio_search_main_prints_matches(monkeypatch, capsys, tmp_path):
    """The search CLI should print detected match windows."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_search,
        "find_audio_sample_matches",
        lambda **_kwargs: [
            AudioMatch(start_seconds=1.23, end_seconds=4.56, duration_seconds=3.33, score=0.91)
        ],
    )

    audio_search.search(source=source, sample=sample, ctx=None)

    output = capsys.readouterr().out
    assert "1.230s -> 4.560s" in output
    assert "score=0.9100" in output


def test_audio_review_main_writes_bundle(monkeypatch, capsys, tmp_path):
    """The review CLI should generate a bundle, manifest, and labels template."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_review,
        "find_audio_sample_matches",
        lambda **_kwargs: [
            AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9),
            AudioMatch(start_seconds=7.0, end_seconds=10.0, duration_seconds=3.0, score=0.8),
        ],
    )
    monkeypatch.setattr(audio_review, "_extract_clip", lambda **_kwargs: None)

    audio_review.review(
        ctx=None,
        source=source,
        sample=sample,
        output_root=tmp_path / "review",
        bundle_name="bundle",
        max_clips=1,
    )

    bundle_dir = tmp_path / "review" / "bundle"
    manifest_path = bundle_dir / "matches_manifest.csv"
    labels_path = bundle_dir / "match_labels.json"
    output = capsys.readouterr().out

    assert bundle_dir.exists()
    assert manifest_path.exists()
    assert labels_path.exists()
    assert "Exported clips: 1 (from 2 total matches)" in output


def test_audio_review_main_writes_interactive_labels(monkeypatch, capsys, tmp_path):
    """With --interactive, the review CLI should write a completed labels file."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_review,
        "find_audio_sample_matches",
        lambda **_kwargs: [
            AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9),
            AudioMatch(start_seconds=7.0, end_seconds=10.0, duration_seconds=3.0, score=0.8),
            AudioMatch(start_seconds=12.0, end_seconds=15.0, duration_seconds=3.0, score=0.7),
        ],
    )
    monkeypatch.setattr(audio_review, "_extract_clip", lambda **_kwargs: None)
    monkeypatch.setattr(
        "partio.cli.commands.audio._auditor.play_audio_segment", lambda **_kwargs: None
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    audio_review.review(
        ctx=None,
        source=source,
        sample=sample,
        output_root=tmp_path / "review",
        bundle_name="interactive",
        interactive=True,
    )

    labels_path = tmp_path / "review" / "interactive" / "match_labels.json"
    assert labels_path.exists()


def test_audio_review_main_default_writes_empty_template(monkeypatch, capsys, tmp_path):
    """Without --interactive, the review CLI writes an empty labels template."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_review,
        "find_audio_sample_matches",
        lambda **_kwargs: [
            AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9),
        ],
    )
    monkeypatch.setattr(audio_review, "_extract_clip", lambda **_kwargs: None)

    audio_review.review(
        ctx=None,
        source=source,
        sample=sample,
        output_root=tmp_path / "review",
        bundle_name="default",
    )

    labels_path = tmp_path / "review" / "default" / "match_labels.json"
    data = json.loads(labels_path.read_text())
    assert data["true_positive_indices"] == []
    assert data["false_positive_indices"] == []


def test_audio_locate_main_prints_best_match(monkeypatch, capsys, tmp_path):
    """The locate CLI should print the best match with prominence."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_locate,
        "find_best_sample_match",
        lambda **_kwargs: BestMatch(
            start_seconds=3.0, end_seconds=6.0, duration_seconds=3.0, score=0.95, prominence=3.2
        ),
    )

    audio_locate.locate(source=source, sample=sample, ctx=None)

    output = capsys.readouterr().out
    assert "3.000s -> 6.000s" in output
    assert "score=0.9500" in output
    assert "prominence=3.20" in output


def test_audio_locate_main_rejects_low_prominence(monkeypatch, capsys, tmp_path):
    """Low prominence should exit with code 1."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_locate,
        "find_best_sample_match",
        lambda **_kwargs: BestMatch(
            start_seconds=3.0, end_seconds=6.0, duration_seconds=3.0, score=0.5, prominence=0.1
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        audio_locate.locate(source=source, sample=sample, min_prominence=2.0, ctx=None)

    assert excinfo.value.code == 1
    assert "No confident match found" in capsys.readouterr().out


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
    from partio.core.ports import AudioPathKind

    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")
    output = tmp_path / "episode_seed.mp3"

    monkeypatch.setattr(audio_bootstrap, "locate_jingle_span", lambda **_kwargs: (5.0, 6.5))
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)

    audio_bootstrap.bootstrap(ctx=None, source=source, output=output)

    entries = library_store.default_store().list_items()
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

    assert len(library_store.default_store().list_items()) == 1


def test_audio_bootstrap_registers_every_seed_of_a_multi_run(monkeypatch, tmp_path):
    """Each numbered seed from --max-occurrences is remembered separately."""
    source = tmp_path / "episode.opus"
    source.write_bytes(b"source")

    monkeypatch.setattr(
        audio_bootstrap, "locate_jingle_spans", lambda **_kwargs: [(1.0, 2.0), (5.0, 6.0)]
    )
    monkeypatch.setattr(audio_bootstrap, "extract_audio_clip", lambda **_kwargs: None)

    audio_bootstrap.bootstrap(ctx=None, source=source, output=tmp_path / "seeds", max_occurrences=2)

    labels = [e.label for e in library_store.default_store().list_items()]
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


def test_audio_bootstrap_main_rejects_missing_source(monkeypatch, capsys, tmp_path):
    """A missing source file should exit via fail."""
    with pytest.raises(SystemExit) as excinfo:
        audio_bootstrap.bootstrap(ctx=None, source=tmp_path / "missing.opus")

    assert excinfo.value.code == 2
    assert "Source not found" in capsys.readouterr().err
