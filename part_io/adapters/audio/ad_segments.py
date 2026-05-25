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
    *,
    approved_indices: frozenset[int] | None = None,
) -> list[AudioMatch]:
    """Read matches from a manifest CSV, optionally filtered to labeled true positives.

    Pass *approved_indices* (a frozenset of CSV index values) to filter without
    a JSON labels file — takes precedence over *labels_path* when provided.
    When neither source supplies non-empty indices, all manifest rows are returned.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    filter_indices: frozenset[int] = frozenset()

    if approved_indices is not None:
        filter_indices = approved_indices
    elif labels_path is not None and labels_path.exists():
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        raw = data.get("true_positive_indices", [])
        if raw:
            filter_indices = frozenset(int(i) for i in raw)

    matches: list[AudioMatch] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        for row in DictReader(f):
            idx = int(row["index"])
            if filter_indices and idx not in filter_indices:
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
    """Optimally pair opens and closes to maximize the total number of valid segments.

    Returns ``(segments, unpaired_opens, unpaired_closes)``. When multiple pairings
    yield the same maximum number of valid ad breaks, the configuration that minimizes
    the total gap durations is chosen. This prevents an orphaned open from "stealing"
    a close that better completes a subsequent ad break.
    """
    sorted_opens = sorted(opens, key=lambda m: m.start_seconds)
    sorted_closes = sorted(closes, key=lambda m: m.start_seconds)

    n_o = len(sorted_opens)
    n_c = len(sorted_closes)

    # dp[i][j] = (max_pairs, min_total_gap, list_of_index_pairs)
    dp: list[list[tuple[int, float, list[tuple[int, int]]]]] = [
        [(0, 0.0, []) for _ in range(n_c + 1)] for _ in range(n_o + 1)
    ]

    for i in range(n_o - 1, -1, -1):
        for j in range(n_c - 1, -1, -1):
            open_match = sorted_opens[i]
            close_match = sorted_closes[j]

            # Option 1: skip open_match
            best = dp[i + 1][j]

            # Option 2: skip close_match
            opt2 = dp[i][j + 1]
            if opt2[0] > best[0] or (opt2[0] == best[0] and opt2[1] < best[1]):
                best = opt2

            # Option 3: pair them if valid
            if close_match.start_seconds > open_match.start_seconds:
                gap = close_match.start_seconds - open_match.end_seconds
                if min_gap <= gap <= max_gap:
                    sub_pairs, sub_cost, sub_choices = dp[i + 1][j + 1]
                    opt3 = (sub_pairs + 1, sub_cost + gap, [(i, j)] + sub_choices)
                    if opt3[0] > best[0] or (opt3[0] == best[0] and opt3[1] < best[1]):
                        best = opt3

            dp[i][j] = best

    _, _, paired_indices = dp[0][0]
    paired_o = {idx_o for idx_o, _ in paired_indices}
    paired_c = {idx_c for _, idx_c in paired_indices}

    segments = []
    for idx_o, idx_c in paired_indices:
        o_match = sorted_opens[idx_o]
        c_match = sorted_closes[idx_c]
        segments.append(
            AdSegment(
                open_start=o_match.start_seconds,
                open_end=o_match.end_seconds,
                close_start=c_match.start_seconds,
                close_end=c_match.end_seconds,
                open_score=o_match.score,
                close_score=c_match.score,
            )
        )

    unpaired_opens = [m for i, m in enumerate(sorted_opens) if i not in paired_o]
    unpaired_closes = [m for i, m in enumerate(sorted_closes) if i not in paired_c]

    return segments, unpaired_opens, unpaired_closes


__all__ = ["AdSegment", "load_manifest_matches", "pair_ad_segments"]
