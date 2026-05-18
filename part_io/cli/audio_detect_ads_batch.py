"""Batch ad-break detection across a directory of MP3 files.

Runs ``audio_detect_ads`` logic in parallel and writes a Markdown report.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch, find_audio_sample_matches
from part_io.cli.audio_detect_ads import _pair_matches

_DEFAULT_MIN_GAP = 25.0
_DEFAULT_MAX_GAP = 300.0


@dataclass
class _EpisodeResult:
    path: Path
    opens: list[AudioMatch]
    closes: list[AudioMatch]
    pairs: list[tuple[AudioMatch, AudioMatch]]
    error: str | None = None


def _scan_one(
    path: Path,
    open_sample: Path,
    close_sample: Path,
    *,
    threshold: float,
    step: float,
    min_gap: float,
    max_gap: float,
    z_threshold: float | None,
) -> _EpisodeResult:
    try:
        opens = find_audio_sample_matches(
            source_path=path,
            sample_path=open_sample,
            score_threshold=threshold,
            step_seconds=step,
            z_threshold=z_threshold,
        )
        closes = find_audio_sample_matches(
            source_path=path,
            sample_path=close_sample,
            score_threshold=threshold,
            step_seconds=step,
            z_threshold=z_threshold,
        )
        pairs = _pair_matches(opens, closes, min_gap=min_gap, max_gap=max_gap)
        return _EpisodeResult(path=path, opens=opens, closes=closes, pairs=pairs)
    except Exception as exc:  # noqa: BLE001
        return _EpisodeResult(path=path, opens=[], closes=[], pairs=[], error=str(exc))


def _format_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _write_report(results: list[_EpisodeResult], out: Path) -> None:
    lines: list[str] = ["# Ad Detection Report\n"]
    ads_found = [r for r in results if r.pairs]
    no_ads = [r for r in results if not r.pairs and not r.error]
    errors = [r for r in results if r.error]

    lines.append(
        f"**{len(results)} episodes scanned** — "
        f"{len(ads_found)} with ads, "
        f"{len(no_ads)} clean, "
        f"{len(errors)} errors\n"
    )

    if ads_found:
        lines.append("\n## Episodes with Ad Breaks\n")
        for r in sorted(ads_found, key=lambda x: x.path.name):
            lines.append(f"\n### {r.path.name}\n")
            lines.append(
                f"Open tags: {len(r.opens)} · Close tags: {len(r.closes)} · "
                f"**Breaks: {len(r.pairs)}**\n"
            )
            for i, (o, c) in enumerate(r.pairs, 1):
                gap = c.start_seconds - o.end_seconds
                lines.append(
                    f"- **Ad {i}**: {_format_seconds(o.start_seconds)} → "
                    f"{_format_seconds(c.end_seconds)}"
                    f" &nbsp; gap {gap:.0f}s"
                    f" &nbsp; open {o.score:.3f}"
                    f" · close {c.score:.3f}\n"
                )

    if no_ads:
        lines.append("\n## Clean Episodes (no ad breaks detected)\n\n")
        for r in sorted(no_ads, key=lambda x: x.path.name):
            lines.append(f"- {r.path.name}\n")

    if errors:
        lines.append("\n## Errors\n\n")
        for r in sorted(errors, key=lambda x: x.path.name):
            lines.append(f"- **{r.path.name}**: {r.error}\n")

    out.write_text("".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch ad-break detection across a directory of MP3 files."
    )
    parser.add_argument("source_dir", type=Path, help="Directory of MP3 files to scan")
    parser.add_argument("open_sample", type=Path, help="Opening-tag reference sample")
    parser.add_argument("close_sample", type=Path, help="Closing-tag reference sample")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("downloads") / "ads_report.md",
        help="Output Markdown report path (default: downloads/ads_report.md)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Match score threshold (default: 0.8)"
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=_DEFAULT_MIN_GAP,
        help="Min gap seconds between open-end and close-start (default: 25)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=_DEFAULT_MAX_GAP,
        help="Max gap seconds between open-end and close-start (default: 300)",
    )
    parser.add_argument(
        "--step", type=float, default=0.5, help="Search step in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score cutoff: keep only matches >= mean + N*std (default: 3.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel workers (default: min(4, cpu_count))",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    for path in (args.open_sample, args.close_sample):
        if not path.exists():
            parser.exit(2, f"Sample file not found: {path}\n")

    if not args.source_dir.is_dir():
        parser.exit(2, f"Not a directory: {args.source_dir}\n")

    mp3s = sorted(args.source_dir.glob("*.mp3"))
    if not mp3s:
        parser.exit(2, f"No MP3 files found in {args.source_dir}\n")

    total = len(mp3s)
    sys.stderr.write(f"Scanning {total} files with {args.workers} workers…\n")

    results: list[_EpisodeResult] = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _scan_one,
                mp3,
                args.open_sample,
                args.close_sample,
                threshold=args.threshold,
                step=args.step,
                min_gap=args.min_gap,
                max_gap=args.max_gap,
                z_threshold=args.z_threshold,
            ): mp3
            for mp3 in mp3s
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1
            tag = f"{len(result.pairs)} ad(s)" if not result.error else f"ERROR: {result.error}"
            sys.stderr.write(f"[{done}/{total}] {result.path.name} — {tag}\n")
            sys.stderr.flush()
            results.append(result)

    _write_report(results, args.output)
    sys.stderr.write(f"\nReport written to {args.output}\n")


if __name__ == "__main__":
    main()
