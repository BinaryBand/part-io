"""CLI for generating manual audio review bundles.

Given a source episode and a reference sample, this tool finds candidate
matches, extracts MP3 clips, and writes a manifest plus labels template under
``downloads/review`` by default.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import typer

from part_io.adapters.audio.clips import extract_audio_clip, play_audio_segment
from part_io.adapters.audio.matcher import AudioMatch, find_audio_sample_matches
from part_io.cli import handle_cli_error
from part_io.cli.output import bundle_summary
from part_io.cli.registry import command
from part_io.core.ports.audio import AuditorFn  # noqa: TC001


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


def build_interactive_auditor(*, source_path: Path) -> AuditorFn:
    def _audition(start_seconds: float, duration_seconds: float, question: str) -> bool:
        play_audio_segment(
            source_path=source_path, start_seconds=start_seconds, duration_seconds=duration_seconds
        )
        answer = input(f"{question} [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    return _audition


def _write_interactive_labels(
    *,
    bundle_dir: Path,
    source_path: Path,
    sample_path: Path,
    threshold: float,
    matches: list[AudioMatch],
    auditor: AuditorFn,
) -> Path:
    labels_path = bundle_dir / "match_labels.json"
    true_positive_indices: list[int] = []
    false_positive_indices: list[int] = []

    for index, match in enumerate(matches, start=1):
        is_match = auditor(match.start_seconds, match.duration_seconds, "Is this a true match?")
        (true_positive_indices if is_match else false_positive_indices).append(index)

    payload = {
        "source_path": str(source_path),
        "sample_path": str(sample_path),
        "true_positive_indices": true_positive_indices,
        "false_positive_indices": false_positive_indices,
        "threshold": threshold,
        "notes": "Labeled interactively via --interactive.",
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
    if not interactive:
        return _write_labels_template(
            bundle_dir=bundle_dir,
            source_path=source_path,
            sample_path=sample_path,
            threshold=threshold,
        )
    if auditor is None:
        raise ValueError("auditor is required when interactive=True")
    return _write_interactive_labels(
        bundle_dir=bundle_dir,
        source_path=source_path,
        sample_path=sample_path,
        threshold=threshold,
        matches=matches,
        auditor=auditor,
    )


def _resolve_bundle_dir(
    *, output_root: Path, source_path: Path, sample_path: Path, bundle_name: str | None
) -> Path:
    if bundle_name:
        return output_root / bundle_name
    return output_root / source_path.stem / sample_path.stem


def _validate_paths(*, source: Path, sample: Path, max_clips: int) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not sample.exists():
        raise FileNotFoundError(f"Sample not found: {sample}")
    if max_clips < 0:
        raise ValueError("--max-clips must be >= 0")


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
    interactive: bool = False,
    auditor: AuditorFn | None = None,
) -> tuple[Path, Path, Path, int, int]:
    _validate_paths(source=source_path, sample=sample_path, max_clips=max_clips)

    bundle_dir = _resolve_bundle_dir(
        output_root=output_root,
        source_path=source_path,
        sample_path=sample_path,
        bundle_name=bundle_name,
    )
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


@command("review-audio", help="Generate review clips + manifest for manual labeling.")
def review(
    source: Annotated[Path, typer.Argument(help="Longer audio file to scan.")],
    sample: Annotated[Path, typer.Argument(help="Reference sample to search for.")],
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
        handle_cli_error(exc)

    for line in bundle_summary(
        bundle_dir=bundle_dir,
        selected_count=selected_count,
        total_matches=total_matches,
        manifest_path=manifest_path,
        labels_path=labels_path,
    ):
        print(line)


def main() -> None:
    """Run as a standalone script."""
    typer.run(review)
