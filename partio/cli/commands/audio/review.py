"""CLI for generating manual audio review bundles.

Given a source episode and a reference sample, this tool finds candidate
matches, extracts MP3 clips, and writes a manifest plus labels template under
``downloads/review`` by default.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from part_io.adapters.audio.clips import extract_audio_clip
from part_io.adapters.audio.matcher import AudioMatch, find_audio_sample_matches
from part_io.cli.commands.audio._auditor import build_interactive_auditor
from part_io.cli.output import _json_flag, bundle_summary, emit, fail
from part_io.cli.registry import command

if TYPE_CHECKING:
    from part_io.core.ports.audio import AuditorFn


def _format_clip_name(index: int, match: AudioMatch) -> str:
    score = f"{match.score:.4f}".replace(".", "_")
    start = f"{match.start_seconds:.3f}".replace(".", "_")
    return f"{index:03d}_score-{score}_start-{start}.mp3"


def _render_manifest_clip_path(clip_path: Path) -> str:
    try:
        return str(clip_path.relative_to(Path.cwd()))
    except ValueError:
        return str(clip_path)


def _extract_clip(*, source_path: Path, destination_path: Path, match: AudioMatch) -> None:
    extract_audio_clip(
        source_path=source_path,
        destination_path=destination_path,
        start_seconds=match.start_seconds,
        duration_seconds=match.duration_seconds,
    )


def _write_manifest(*, bundle_dir: Path, source_path: Path, matches: list[AudioMatch]) -> Path:
    manifest_path = bundle_dir / "matches_manifest.csv"
    fieldnames = [
        "index",
        "score",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "clip_path",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, match in enumerate(matches, start=1):
            clip_path = bundle_dir / _format_clip_name(index, match)
            _extract_clip(source_path=source_path, destination_path=clip_path, match=match)
            writer.writerow(
                {
                    "index": index,
                    "score": round(match.score, 4),
                    "start_seconds": round(match.start_seconds, 3),
                    "end_seconds": round(match.end_seconds, 3),
                    "duration_seconds": round(match.duration_seconds, 3),
                    "clip_path": _render_manifest_clip_path(clip_path),
                }
            )

    return manifest_path


def _write_labels_template(
    *, bundle_dir: Path, source_path: Path, sample_path: Path, threshold: float
) -> Path:
    labels_path = bundle_dir / "match_labels.json"
    payload = {
        "source_path": str(source_path),
        "sample_path": str(sample_path),
        "true_positive_indices": [],
        "false_positive_indices": [],
        "threshold": threshold,
        "notes": "Fill true_positive_indices / false_positive_indices after manual listening.",
    }
    labels_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return labels_path


def _write_interactive_labels(
    *,
    bundle_dir: Path,
    source_path: Path,
    sample_path: Path,
    threshold: float,
    matches: list[AudioMatch],
) -> Path:
    from part_io.adapters.audio.clips import play_audio_segment

    true_indices: list[int] = []
    false_indices: list[int] = []

    for index, match in enumerate(matches, start=1):
        play_audio_segment(
            source_path=source_path,
            start_seconds=match.start_seconds,
            duration_seconds=match.duration_seconds,
        )
        answer = input(f"Match {index}/{len(matches)} (score={match.score:.4f})? [y/n]: ")
        if answer.strip().lower() in ("y", "yes"):
            true_indices.append(index)
        else:
            false_indices.append(index)

    labels_path = bundle_dir / "match_labels.json"
    payload = {
        "source_path": str(source_path),
        "sample_path": str(sample_path),
        "true_positive_indices": true_indices,
        "false_positive_indices": false_indices,
        "threshold": threshold,
    }
    labels_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return labels_path


def _write_labels(
    *,
    bundle_dir: Path,
    source_path: Path,
    sample_path: Path,
    threshold: float,
    matches: list[AudioMatch],
    interactive: bool,
    auditor: AuditorFn | None,
) -> Path:
    if interactive and auditor is not None:
        return _write_interactive_labels(
            bundle_dir=bundle_dir,
            source_path=source_path,
            sample_path=sample_path,
            threshold=threshold,
            matches=matches,
        )
    return _write_labels_template(
        bundle_dir=bundle_dir,
        source_path=source_path,
        sample_path=sample_path,
        threshold=threshold,
    )


def _validate_paths(*, source: Path, sample: Path, max_clips: int) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not sample.exists():
        raise FileNotFoundError(f"Sample not found: {sample}")
    if max_clips < 0:
        raise ValueError("max_clips must be non-negative")


def _generate_bundle(
    *,
    source_path: Path,
    sample_path: Path,
    threshold: float,
    step_seconds: float,
    dedupe_overlap: float,
    max_clips: int,
    output_root: Path,
    bundle_name: str | None,
    overwrite: bool,
    interactive: bool,
    auditor: AuditorFn | None,
) -> tuple[Path, Path, Path, int, int]:
    source_stem = source_path.stem
    sample_stem = sample_path.stem
    dir_name = bundle_name or f"{source_stem}/{sample_stem}"
    bundle_dir = output_root / dir_name

    if bundle_dir.exists() and not overwrite:
        raise ValueError(f"Bundle already exists: {bundle_dir} (use --overwrite)")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    matches = find_audio_sample_matches(
        source_path=source_path,
        sample_path=sample_path,
        score_threshold=threshold,
        step_seconds=step_seconds,
        dedupe_overlap=dedupe_overlap,
    )
    selected_matches = matches if max_clips == 0 else matches[:max_clips]
    manifest_path = _write_manifest(
        bundle_dir=bundle_dir,
        source_path=source_path,
        matches=selected_matches,
    )
    labels_path = _write_labels(
        bundle_dir=bundle_dir,
        source_path=source_path,
        sample_path=sample_path,
        threshold=threshold,
        matches=selected_matches,
        interactive=interactive,
        auditor=auditor,
    )
    return bundle_dir, manifest_path, labels_path, len(matches), len(selected_matches)


@command("audio", "review", help="Generate review clips + manifest for manual labeling.")
def review(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Option("--source", prompt="Source audio file", help="Longer audio file to scan."),
    ],
    sample: Annotated[
        Path,
        typer.Option("--sample", prompt="Reference sample", help="Reference sample to search for."),
    ],
    threshold: Annotated[float, typer.Option(help="Match score threshold.")] = 0.8,
    step_seconds: Annotated[float, typer.Option(help="Sliding-window step in seconds.")] = 0.1,
    dedupe_overlap: Annotated[
        float, typer.Option(help="Suppress overlapping matches above this ratio.")
    ] = 0.5,
    max_clips: Annotated[
        int, typer.Option(help="Maximum top-scored matches to export (0 = all).")
    ] = 25,
    output_root: Annotated[
        Path, typer.Option(help="Root folder where review bundles are written.")
    ] = Path("downloads") / "review",
    bundle_name: Annotated[
        str | None,
        typer.Option(help="Optional bundle directory name under output root."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Allow writing into an existing bundle."),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive/--no-interactive",
            help="Audition each clip and label interactively.",
        ),
    ] = False,
) -> None:
    """Generate review clips + manifest for manual labeling."""
    try:
        _validate_paths(source=source, sample=sample, max_clips=max_clips)
        auditor = build_interactive_auditor(source_path=source) if interactive else None
        bundle_dir, manifest_path, labels_path, total_matches, selected_count = _generate_bundle(
            source_path=source,
            sample_path=sample,
            threshold=threshold,
            step_seconds=step_seconds,
            dedupe_overlap=dedupe_overlap,
            max_clips=max_clips,
            output_root=output_root,
            bundle_name=bundle_name,
            overwrite=overwrite,
            interactive=interactive,
            auditor=auditor,
        )
    except (FileNotFoundError, ValueError) as exc:
        fail(exc)

    emit(
        bundle_summary(
            bundle_dir=bundle_dir,
            selected_count=selected_count,
            total_matches=total_matches,
            manifest_path=manifest_path,
            labels_path=labels_path,
        ),
        as_json=_json_flag(ctx),
    )
