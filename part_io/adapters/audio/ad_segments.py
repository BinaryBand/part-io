"""Ad-segment pairing and manifest loading for the ad-removal pipeline.

Given open and close match manifests (produced by audio-review-batch), this
module pairs each detected ad open with the nearest following close within a
plausible time window and produces a list of AdSegment objects that define
exactly which spans of the source file should be cut.
"""

from __future__ import annotations

import json
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch


@dataclass(frozen=True)
class AdSegment:
    """One detected ad break — the span [open_start, close_end] should be cut."""

    open_start: float
    open_end: float
    close_start: float
    close_end: float
    open_score: float
    close_score: float

    @property
    def cut_start(self) -> float:
        return self.open_start

    @property
    def cut_end(self) -> float:
        return self.close_end

    @property
    def gap_seconds(self) -> float:
        return round(self.close_start - self.open_end, 3)


def load_manifest_matches(
    manifest_path: Path,
    labels_path: Path | None = None,
) -> list[AudioMatch]:
    """Read matches from a manifest CSV, optionally filtered to labeled true positives.

    When *labels_path* is given and its ``true_positive_indices`` list is
    non-empty, only those rows are returned.  If the list is empty (labels not
    yet filled in) all manifest rows are returned with a warning logged to the
    caller via the second element of the returned tuple.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    true_positive_indices: frozenset[int] = frozenset()
    labels_filled = False

    if labels_path is not None and labels_path.exists():
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        raw = data.get("true_positive_indices", [])
        if raw:
            true_positive_indices = frozenset(int(i) for i in raw)
            labels_filled = True

    matches: list[AudioMatch] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        for row in DictReader(f):
            idx = int(row["index"])
            if labels_filled and idx not in true_positive_indices:
                continue
            matches.append(
                AudioMatch(
                    start_seconds=float(row["start_seconds"]),
                    end_seconds=float(row["end_seconds"]),
                    duration_seconds=float(row["duration_seconds"]),
                    score=float(row["score"]),
                )
            )

    return sorted(matches, key=lambda m: m.start_seconds)


def pair_ad_segments(
    opens: list[AudioMatch],
    closes: list[AudioMatch],
    *,
    min_gap: float = 10.0,
    max_gap: float = 600.0,
) -> tuple[list[AdSegment], list[AudioMatch], list[AudioMatch]]:
    """Greedily pair each open with the nearest following eligible close.

    Returns ``(segments, unpaired_opens, unpaired_closes)``.  An open is
    skipped (added to *unpaired_opens*) when no close falls within
    ``[min_gap, max_gap]`` seconds after the open's end.  Consumed closes are
    removed from further consideration so each close matches at most one open.
    """
    sorted_opens = sorted(opens, key=lambda m: m.start_seconds)
    available_closes = sorted(closes, key=lambda m: m.start_seconds)

    segments: list[AdSegment] = []
    unpaired_opens: list[AudioMatch] = []

    for open_match in sorted_opens:
        best: AudioMatch | None = None
        for close_match in available_closes:
            gap = close_match.start_seconds - open_match.end_seconds
            if gap < min_gap:
                continue
            if gap > max_gap:
                break
            best = close_match
            break

        if best is None:
            unpaired_opens.append(open_match)
            continue

        available_closes.remove(best)
        segments.append(
            AdSegment(
                open_start=open_match.start_seconds,
                open_end=open_match.end_seconds,
                close_start=best.start_seconds,
                close_end=best.end_seconds,
                open_score=open_match.score,
                close_score=best.score,
            )
        )

    return segments, unpaired_opens, list(available_closes)


__all__ = ["AdSegment", "load_manifest_matches", "pair_ad_segments"]
