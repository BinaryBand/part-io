"""CLI: remove detected ad segments from an episode MP3.

Reads ad_segments.json (produced by audio-ad-detect) and uses a single ffmpeg
filter_complex command to stitch together the non-ad spans of the source file.
No temporary files are written — the atrim+concat filter graph handles
everything in one pass.

Use --dry-run to print the cut plan without touching any audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from part_io.adapters.audio.ad_segments import AdSegment
from part_io.adapters.process.runner import run_resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove detected ad segments from an episode MP3.")
    parser.add_argument("--source", type=Path, required=True, help="Source episode MP3")
    parser.add_argument(
        "--segments",
        type=Path,
        required=True,
        help="Path to ad_segments.json produced by audio-ad-detect",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads") / "cleaned",
        help="Directory to write cleaned MP3 (default: downloads/cleaned)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output path (overrides --output-dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cut plan without running ffmpeg",
    )
    return parser


def _load_segments(segments_path: Path) -> list[AdSegment]:
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")
    data = json.loads(segments_path.read_text(encoding="utf-8"))
    return [
        AdSegment(
            open_start=s["open_start"],
            open_end=s["open_end"],
            close_start=s["close_start"],
            close_end=s["close_end"],
            open_score=s["open_score"],
            close_score=s["close_score"],
        )
        for s in data.get("segments", [])
    ]


def _validate_segments(segments: list[AdSegment]) -> None:
    """Raise if segments overlap or are not sorted."""
    sorted_segs = sorted(segments, key=lambda s: s.cut_start)
    for i in range(len(sorted_segs) - 1):
        if sorted_segs[i].cut_end > sorted_segs[i + 1].cut_start:
            raise ValueError(
                f"Overlapping ad segments: [{sorted_segs[i].cut_start}, {sorted_segs[i].cut_end}]"
                f" overlaps [{sorted_segs[i + 1].cut_start}, {sorted_segs[i + 1].cut_end}]"
            )


def _build_keep_spans(segments: list[AdSegment]) -> list[tuple[float, float | None]]:
    """Return [(start, end), ...] spans of audio to keep, end=None means 'to EOF'."""
    sorted_segs = sorted(segments, key=lambda s: s.cut_start)
    spans: list[tuple[float, float | None]] = []
    cursor = 0.0
    for seg in sorted_segs:
        if seg.cut_start > cursor:
            spans.append((cursor, seg.cut_start))
        cursor = seg.cut_end
    spans.append((cursor, None))
    return spans


def _build_filter_complex(spans: "Sequence[tuple[float, float | None]]") -> tuple[str, int]:
    """Build ffmpeg filter_complex string and return (filter_str, n_segments)."""
    parts: list[str] = []
    labels: list[str] = []
    for i, (start, end) in enumerate(spans):
        label = f"s{i}"
        if end is None:
            trim = f"atrim={start:.3f}"
        else:
            trim = f"atrim={start:.3f}:{end:.3f}"
        parts.append(f"[0:a]{trim},asetpts=PTS-STARTPTS[{label}]")
        labels.append(f"[{label}]")
    concat_inputs = "".join(labels)
    n = len(spans)
    parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[out]")
    return ";".join(parts), n


def _run_ffmpeg(
    source: Path,
    filter_complex: str,
    output: Path,
) -> int:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(output),
    ]
    result = run_resolved(command)
    return int(result.returncode)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.source.exists():
        parser.exit(2, f"Source file not found: {args.source}\n")

    try:
        segments = _load_segments(args.segments)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
        return

    if not segments:
        print("No ad segments found in segments file — nothing to cut.")
        return

    try:
        _validate_segments(segments)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
        return

    sorted_segs = sorted(segments, key=lambda s: s.cut_start)
    spans = _build_keep_spans(sorted_segs)

    print(f"Source: {args.source}")
    print(f"Ad segments to cut: {len(sorted_segs)}")
    for i, seg in enumerate(sorted_segs, 1):
        print(
            f"  {i}. cut [{seg.cut_start:.3f}s, {seg.cut_end:.3f}s]"
            f"  ({seg.cut_end - seg.cut_start:.1f}s)"
        )
    print(f"Keep spans: {len(spans)}")
    for start, end in spans:
        end_str = f"{end:.3f}s" if end is not None else "EOF"
        print(f"  [{start:.3f}s, {end_str}]")

    if args.dry_run:
        print("Dry run — no output written.")
        return

    output = args.output
    if output is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / args.source.name

    filter_complex, _ = _build_filter_complex(spans)
    exit_code = _run_ffmpeg(args.source, filter_complex, output)
    if exit_code != 0:
        print(f"ffmpeg failed (exit {exit_code})", file=sys.stderr)
        sys.exit(exit_code)

    print(f"Written: {output}")


if __name__ == "__main__":
    main()
