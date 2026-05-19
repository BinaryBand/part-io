"""Review orchestration service for remote pipeline.

This module centralizes uncertain-candidate collection, decision application, and
reclassification logic so these operations stay consistent across CLI and other
orchestration flows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Classification = Literal["positive", "negative", "uncertain", "undetected"]


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
_SINGLE_SAMPLE_MOE = 0.05


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


def _get_kind_scores(episode: dict, kind: str) -> tuple[list[float], list[float]]:
    """Extract positive and negative scores for a kind from episode state dict."""
    positives = []
    negatives = []
    candidates_key = f"{kind}_candidates"
    class_key = f"{kind}_class"

    if candidates_key not in episode:
        return positives, negatives

    ep_class = episode.get(class_key, "undetected")
    if ep_class == "positive":
        for cand in episode[candidates_key]:
            positives.append(float(cand.get("score", 0.0)))
    elif ep_class == "negative":
        for cand in episode[candidates_key]:
            negatives.append(float(cand.get("score", 0.0)))

    return positives, negatives


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


def reclassify_all_episodes(
    episodes: dict[str, dict],
    open_target_positives: list[dict],
    open_target_negatives: list[dict],
    close_target_positives: list[dict],
    close_target_negatives: list[dict],
) -> None:
    """Recompute MOE-derived thresholds and reclassify uncertain episodes in-place."""
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
