"""Review orchestration service for remote pipeline.

This module centralizes uncertain-candidate collection, decision application, and
reclassification logic so these operations stay consistent across CLI and other
orchestration flows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

Classification = Literal["positive", "negative", "uncertain", "undetected"]
REVIEW_KINDS = ("open", "close", "intro", "outro")


class EpisodeStateLike(Protocol):
    def candidates_for(self, kind: str) -> list[Any]: ...

    def class_for(self, kind: str) -> str: ...

    def set_class(self, kind: str, value: str) -> None: ...


@dataclass(frozen=True)
class ReviewItem:
    """One candidate to review during interactive session."""

    stem: str
    kind: str  # "open", "close", "intro", or "outro"
    candidate_idx: int  # index into episode's candidates list for this kind
    score: float  # candidate score (for sorting)


@dataclass(frozen=True)
class ReviewDecision:
    """Result of applying a review decision (approve/reject)."""

    action: Literal["approved", "rejected"]
    segment_source: str
    segment_start: float
    segment_end: float
    segment_score: float


@dataclass
class UndoEntry:
    """Undo state for one review decision."""

    stem: str
    kind: str
    action: str  # "a" or "r"
    segment_source: str
    segment_start: float
    segment_end: float
    segment_score: float
    target_list_was_positive: bool  # True if appended to positives, False if negatives
    prev_class: str  # previous classification before decision


def episode_to_review_dict(
    episode: EpisodeStateLike,
    *,
    include_bounds: bool,
) -> dict[str, Any]:
    """Convert an episode state object into the dict shape used by this service."""
    data: dict[str, Any] = {}
    for kind in REVIEW_KINDS:
        if include_bounds:
            data[f"{kind}_candidates"] = [
                {"score": float(match.score), "start": float(match.start), "end": float(match.end)}
                for match in episode.candidates_for(kind)
            ]
        else:
            data[f"{kind}_candidates"] = [
                {"score": float(match.score)} for match in episode.candidates_for(kind)
            ]
        data[f"{kind}_class"] = episode.class_for(kind)
    return data


def apply_review_dict_classes(episode: EpisodeStateLike, episode_dict: dict[str, Any]) -> None:
    """Copy service-produced class values back into an episode state object."""
    for kind in REVIEW_KINDS:
        episode.set_class(kind, str(episode_dict.get(f"{kind}_class", episode.class_for(kind))))


def apply_review_decision(
    *,
    episode: dict,
    kind: str,
    candidate_idx: int,
    action: Literal["a", "r"],
    source: str,
    open_target_positives: list[dict],
    open_target_negatives: list[dict],
    close_target_positives: list[dict],
    close_target_negatives: list[dict],
) -> tuple[ReviewDecision, UndoEntry]:
    """Apply one review action (approve/reject) to episode and targets.

    Mutates *episode* and relevant target lists in-place and returns the
    corresponding decision/undo records.
    """
    candidates = episode.get(f"{kind}_candidates", [])
    if candidate_idx < 0 or candidate_idx >= len(candidates):
        raise IndexError(f"candidate_idx out of range for {kind}: {candidate_idx}")

    cand = candidates[candidate_idx]
    segment = {
        "source": source,
        "start": float(cand.get("start", 0.0)),
        "end": float(cand.get("end", 0.0)),
        "score": float(cand.get("score", 0.0)),
    }
    prev_class = str(episode.get(f"{kind}_class", "uncertain"))

    if action == "a":
        if kind == "open":
            open_target_positives.append(segment)
        elif kind == "close":
            close_target_positives.append(segment)
        episode[f"{kind}_class"] = "positive"
        decision_action: Literal["approved", "rejected"] = "approved"
        target_list_was_positive = True
    else:
        if kind == "open":
            open_target_negatives.append(segment)
        elif kind == "close":
            close_target_negatives.append(segment)
        elif kind in ("intro", "outro"):
            episode[f"{kind}_class"] = "negative"
        decision_action = "rejected"
        target_list_was_positive = False

    decision = ReviewDecision(
        action=decision_action,
        segment_source=segment["source"],
        segment_start=float(segment["start"]),
        segment_end=float(segment["end"]),
        segment_score=float(segment["score"]),
    )
    undo = UndoEntry(
        stem="",
        kind=kind,
        action=action,
        segment_source=decision.segment_source,
        segment_start=decision.segment_start,
        segment_end=decision.segment_end,
        segment_score=decision.segment_score,
        target_list_was_positive=target_list_was_positive,
        prev_class=prev_class,
    )
    return decision, undo


def undo_review_decision(
    *,
    episode: dict,
    undo: UndoEntry,
    open_target_positives: list[dict],
    open_target_negatives: list[dict],
    close_target_positives: list[dict],
    close_target_negatives: list[dict],
) -> None:
    """Undo one previously applied review decision in-place."""

    def _remove_segment(target_list: list[dict]) -> None:
        for i, seg in enumerate(target_list):
            if (
                str(seg.get("source", "")) == undo.segment_source
                and float(seg.get("start", 0.0)) == float(undo.segment_start)
                and float(seg.get("end", 0.0)) == float(undo.segment_end)
                and float(seg.get("score", 0.0)) == float(undo.segment_score)
            ):
                target_list.pop(i)
                return

    if undo.kind == "open":
        _remove_segment(
            open_target_positives if undo.target_list_was_positive else open_target_negatives
        )
    elif undo.kind == "close":
        _remove_segment(
            close_target_positives if undo.target_list_was_positive else close_target_negatives
        )

    episode[f"{undo.kind}_class"] = undo.prev_class


# Thresholds for 98% t-critical values used in MOE computation.
_T_CRIT: dict[int, float] = {
    2: 31.821,
    3: 6.965,
    4: 4.541,
    5: 3.747,
    6: 3.365,
    7: 3.143,
    8: 2.998,
    9: 2.896,
    10: 2.821,
    15: 2.624,
    20: 2.539,
    25: 2.492,
    30: 2.462,
}
_T_CRIT_LARGE = 2.326


def _t_critical(n: int) -> float:
    """98% two-tailed t-critical value for n samples (df = n-1)."""
    if n < 2:
        return math.inf
    if n >= 31:
        return _T_CRIT_LARGE
    for threshold in sorted(_T_CRIT, reverse=True):
        if n >= threshold:
            return _T_CRIT[threshold]
    return math.inf


def _moe(scores: list[float]) -> float:
    """Margin of error using the t-distribution (98% CI on the sample mean).

    Returns math.inf for n < 2 so that a single confirmed example never
    triggers auto-classification — the uncertain zone collapses only as
    evidence accumulates across multiple confirmed samples.
    """
    n = len(scores)
    if n < 2:
        return math.inf
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)  # Bessel's correction
    return _t_critical(n) * math.sqrt(variance) / math.sqrt(n)


def compute_classification_thresholds(
    positives: list[float], negatives: list[float]
) -> tuple[float, float]:
    """Return (theta_plus, theta_minus) classification thresholds.

    θ⁺ = min(positives) + moe  — auto-positive requires exceeding the minimum confirmed
                                  positive BY the uncertainty buffer (worst-case threshold).
    θ⁻ = max(negatives) - moe  — auto-negative requires falling below the maximum confirmed
                                  negative BY the uncertainty buffer.
    The uncertain zone (θ⁻, θ⁺) widens with high variance and narrows as evidence accumulates.
    With no positives: θ⁺ = +inf. With no negatives: θ⁻ = -inf.
    With fewer than 2 confirmed samples of either kind, moe = inf so nothing auto-classifies.
    """
    theta_plus = (min(positives) + _moe(positives)) if positives else math.inf
    theta_minus = (max(negatives) - _moe(negatives)) if negatives else -math.inf
    return theta_plus, theta_minus


def classify_score_with_thresholds(
    score: float, theta_plus: float, theta_minus: float
) -> Classification:
    """Classify a score using decision thresholds."""
    if score >= theta_plus:
        return "positive"
    if score <= theta_minus:
        return "negative"
    return "uncertain"


def collect_uncertain_candidates(
    episodes: dict[str, dict],
    open_target_positives: list[dict],
    open_target_negatives: list[dict],
    close_target_positives: list[dict],
    close_target_negatives: list[dict],
) -> list[ReviewItem]:
    """Return all candidates in the uncertain zone (θ⁻, θ⁺).

    Accepts target scores as dicts with "score" keys since this is stateless.
    Sorted by (candidate_idx, -score): all top candidates across every uncertain
    (stem, kind) pair come first, then all second candidates, etc.
    """
    tp_o, tm_o = compute_classification_thresholds(
        [float(s["score"]) for s in open_target_positives],
        [float(s["score"]) for s in open_target_negatives],
    )
    tp_c, tm_c = compute_classification_thresholds(
        [float(s["score"]) for s in close_target_positives],
        [float(s["score"]) for s in close_target_negatives],
    )

    items: list[ReviewItem] = []
    for stem, ep in episodes.items():
        # Open candidates
        if ep.get("open_class") == "uncertain":
            for i, cand in enumerate(ep.get("open_candidates", [])):
                score = float(cand.get("score", 0.0))
                if tm_o < score < tp_o:
                    items.append(ReviewItem(stem=stem, kind="open", candidate_idx=i, score=score))

        # Close candidates
        if ep.get("close_class") == "uncertain":
            for i, cand in enumerate(ep.get("close_candidates", [])):
                score = float(cand.get("score", 0.0))
                if tm_c < score < tp_c:
                    items.append(ReviewItem(stem=stem, kind="close", candidate_idx=i, score=score))

        # Intro candidates (no global target; all uncertain are reviewable)
        if ep.get("intro_class") == "uncertain":
            for i, cand in enumerate(ep.get("intro_candidates", [])):
                score = float(cand.get("score", 0.0))
                items.append(ReviewItem(stem=stem, kind="intro", candidate_idx=i, score=score))

        # Outro candidates (no global target; all uncertain are reviewable)
        if ep.get("outro_class") == "uncertain":
            for i, cand in enumerate(ep.get("outro_candidates", [])):
                score = float(cand.get("score", 0.0))
                items.append(ReviewItem(stem=stem, kind="outro", candidate_idx=i, score=score))

    items.sort(key=lambda x: (x.candidate_idx, -x.score))
    return items


def next_uncertain_episode_kind(
    episodes: dict[str, dict],
    *,
    exclude: set[tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    """Return (stem, kind) for the highest-scoring uncertain target.

    This mirrors the interactive review queue behavior used by the CLI.
    Only kinds whose class is ``uncertain`` and whose top candidate has a
    strictly positive score are considered.
    """
    candidates: list[tuple[float, str, str]] = []
    for stem, ep in episodes.items():
        for kind in ("open", "close", "intro", "outro"):
            if exclude is not None and (kind, stem) in exclude:
                continue
            if ep.get(f"{kind}_class") != "uncertain":
                continue
            kind_candidates = ep.get(f"{kind}_candidates", [])
            if not kind_candidates:
                continue
            score = float(kind_candidates[0].get("score", 0.0))
            if score <= 0:
                continue
            candidates.append((score, stem, kind))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, stem, kind = candidates[0]
    return stem, kind


def reclassify_all_episodes(
    episodes: dict[str, dict],
    open_target_positives: list[dict],
    open_target_negatives: list[dict],
    close_target_positives: list[dict],
    close_target_negatives: list[dict],
) -> None:
    """Recompute MOE-derived thresholds and reclassify uncertain episodes in-place.

    Only `open` and `close` are globally thresholded. `intro` and `outro`
    remain human-reviewed classes and are intentionally left unchanged here.
    """
    tp_o, tm_o = compute_classification_thresholds(
        [float(s["score"]) for s in open_target_positives],
        [float(s["score"]) for s in open_target_negatives],
    )
    tp_c, tm_c = compute_classification_thresholds(
        [float(s["score"]) for s in close_target_positives],
        [float(s["score"]) for s in close_target_negatives],
    )

    for ep in episodes.values():
        # Open
        if ep.get("open_class") == "uncertain" and ep.get("open_candidates"):
            score = float(ep["open_candidates"][0]["score"])
            ep["open_class"] = classify_score_with_thresholds(score, tp_o, tm_o)

        # Close
        if ep.get("close_class") == "uncertain" and ep.get("close_candidates"):
            score = float(ep["close_candidates"][0]["score"])
            ep["close_class"] = classify_score_with_thresholds(score, tp_c, tm_c)
