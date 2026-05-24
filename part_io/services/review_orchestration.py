"""Review orchestration service for remote pipeline.

This module centralizes uncertain-candidate collection, decision application, and
reclassification logic so these operations stay consistent across CLI and other
orchestration flows.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from scipy.stats import t as _t_dist

Classification = Literal["positive", "negative", "uncertain", "undetected"]
REVIEW_KINDS = ("open", "close", "intro", "outro")


class EpisodeStateLike(Protocol):
    def candidates_for(self, kind: str) -> list[Any]: ...

    def class_for(self, kind: str) -> str: ...


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
    candidate_idx: int
    target_list_was_positive: bool  # True if appended to positives, False if negatives
    prev_class: str  # previous classification before decision
    prev_label: str | None  # previous candidate label before decision


def _candidate_label(cand: dict[str, Any]) -> str | None:
    label = cand.get("label")
    if label in ("positive", "negative"):
        return str(label)
    return None


def _episode_class_for_kind(ep: dict[str, Any], kind: str) -> Classification:
    candidates = ep.get(f"{kind}_candidates", [])
    if not candidates:
        return "undetected"
    labels = [_candidate_label(cand) for cand in candidates]
    if any(label == "positive" for label in labels):
        return "positive"
    if all(label == "negative" for label in labels):
        return "negative"
    return "uncertain"


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
                {
                    "score": float(match.score),
                    "start": float(match.start),
                    "end": float(match.end),
                    "label": getattr(match, "label", None),
                }
                for match in episode.candidates_for(kind)
            ]
        else:
            data[f"{kind}_candidates"] = [
                {"score": float(match.score), "label": getattr(match, "label", None)}
                for match in episode.candidates_for(kind)
            ]
        data[f"{kind}_class"] = episode.class_for(kind)
    return data


def apply_review_dict_classes(episode: EpisodeStateLike, episode_dict: dict[str, Any]) -> None:
    """Copy service-produced candidate labels back into an episode state object."""
    for kind in REVIEW_KINDS:
        candidates = episode.candidates_for(kind)
        raw_candidates = episode_dict.get(f"{kind}_candidates", [])
        for idx, candidate in enumerate(candidates):
            if idx >= len(raw_candidates):
                break
            candidate.label = _candidate_label(raw_candidates[idx])


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
    prev_class = _episode_class_for_kind(episode, kind)
    prev_label = _candidate_label(cand)

    if action == "a":
        if kind == "open":
            open_target_positives.append(segment)
        elif kind == "close":
            close_target_positives.append(segment)
        cand["label"] = "positive"
        decision_action: Literal["approved", "rejected"] = "approved"
        target_list_was_positive = True
    else:
        if kind == "open":
            open_target_negatives.append(segment)
        elif kind == "close":
            close_target_negatives.append(segment)
        cand["label"] = "negative"
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
        candidate_idx=candidate_idx,
        target_list_was_positive=target_list_was_positive,
        prev_class=prev_class,
        prev_label=prev_label,
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

    candidates = episode.get(f"{undo.kind}_candidates", [])
    if 0 <= undo.candidate_idx < len(candidates):
        candidates[undo.candidate_idx]["label"] = undo.prev_label


@lru_cache(maxsize=1)
def _confidence_level() -> float:
    """Read MOE confidence level from pyproject.toml, defaulting to 0.98."""
    for candidate in (Path("pyproject.toml"), Path(__file__).parents[3] / "pyproject.toml"):
        if candidate.exists():
            with candidate.open("rb") as f:
                data = tomllib.load(f)
            return float(
                data.get("tool", {}).get("part_io", {}).get("moe", {}).get("confidence_level", 0.98)
            )
    return 0.98


def _t_critical(n: int) -> float:
    """Two-tailed t-critical value for n samples (df = n-1) at the configured confidence level."""
    if n < 2:
        return math.inf
    alpha = (1.0 + _confidence_level()) / 2.0
    return float(_t_dist.ppf(alpha, df=n - 1))


def _moe(scores: list[float]) -> float:
    """Margin of error at the configured confidence level (t-distribution, sample mean CI).

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
        if _episode_class_for_kind(ep, "open") == "uncertain":
            for i, cand in enumerate(ep.get("open_candidates", [])):
                score = float(cand.get("score", 0.0))
                if _candidate_label(cand) is None and tm_o < score < tp_o:
                    items.append(ReviewItem(stem=stem, kind="open", candidate_idx=i, score=score))

        # Close candidates
        if _episode_class_for_kind(ep, "close") == "uncertain":
            for i, cand in enumerate(ep.get("close_candidates", [])):
                score = float(cand.get("score", 0.0))
                if _candidate_label(cand) is None and tm_c < score < tp_c:
                    items.append(ReviewItem(stem=stem, kind="close", candidate_idx=i, score=score))

        # Intro candidates (no global target; all uncertain are reviewable)
        if _episode_class_for_kind(ep, "intro") == "uncertain":
            for i, cand in enumerate(ep.get("intro_candidates", [])):
                score = float(cand.get("score", 0.0))
                if _candidate_label(cand) is None:
                    items.append(ReviewItem(stem=stem, kind="intro", candidate_idx=i, score=score))

        # Outro candidates (no global target; all uncertain are reviewable)
        if _episode_class_for_kind(ep, "outro") == "uncertain":
            for i, cand in enumerate(ep.get("outro_candidates", [])):
                score = float(cand.get("score", 0.0))
                if _candidate_label(cand) is None:
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
            kind_candidates = ep.get(f"{kind}_candidates", [])
            if not kind_candidates:
                continue
            if _episode_class_for_kind(ep, kind) != "uncertain":
                continue
            pending = [cand for cand in kind_candidates if _candidate_label(cand) is None]
            if not pending:
                continue
            score = float(pending[0].get("score", 0.0))
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
        for cand in ep.get("open_candidates", []):
            if _candidate_label(cand) is not None:
                continue
            score = float(cand.get("score", 0.0))
            classified = classify_score_with_thresholds(score, tp_o, tm_o)
            if classified in ("positive", "negative"):
                cand["label"] = classified

        # Close
        for cand in ep.get("close_candidates", []):
            if _candidate_label(cand) is not None:
                continue
            score = float(cand.get("score", 0.0))
            classified = classify_score_with_thresholds(score, tp_c, tm_c)
            if classified in ("positive", "negative"):
                cand["label"] = classified


def _count_newly_classified(
    uncertain_scores: list[float],
    theta_plus: float,
    theta_minus: float,
) -> int:
    """Count how many scores would auto-classify under the given thresholds."""
    return sum(1 for s in uncertain_scores if s >= theta_plus or s <= theta_minus)


def _expected_savings(
    candidate_score: float,
    kind_uncertain_scores: list[float],
    positives: list[float],
    negatives: list[float],
) -> float:
    """Expected number of other uncertain candidates auto-classified if we answer this one.

    Simulates both outcomes (approve / reject), weights by the score's position in
    the uncertain zone as a proxy for P(approve), and returns the weighted sum.
    """
    theta_plus, theta_minus = compute_classification_thresholds(positives, negatives)
    zone_width = theta_plus - theta_minus
    p_approve = (
        (candidate_score - theta_minus) / zone_width
        if math.isfinite(zone_width) and zone_width > 0
        else 0.5
    )

    new_pos = sorted(positives + [candidate_score])
    tp_if_approve, tm_if_approve = compute_classification_thresholds(new_pos, negatives)
    n_if_approve = _count_newly_classified(kind_uncertain_scores, tp_if_approve, tm_if_approve)

    new_neg = sorted(negatives + [candidate_score])
    tp_if_reject, tm_if_reject = compute_classification_thresholds(positives, new_neg)
    n_if_reject = _count_newly_classified(kind_uncertain_scores, tp_if_reject, tm_if_reject)

    return p_approve * n_if_approve + (1.0 - p_approve) * n_if_reject


def sort_by_expected_savings(
    items: list[ReviewItem],
    open_target_positives: list[dict],
    open_target_negatives: list[dict],
    close_target_positives: list[dict],
    close_target_negatives: list[dict],
) -> list[ReviewItem]:
    """Re-order *items* so highest expected cascade classification comes first.

    Only open/close candidates are reordered — they share global targets whose
    thresholds shift with each decision. intro/outro have no global target so
    expected savings is always 0; they are appended after open/close in their
    original relative order.
    """
    open_pos = [float(s["score"]) for s in open_target_positives]
    open_neg = [float(s["score"]) for s in open_target_negatives]
    close_pos = [float(s["score"]) for s in close_target_positives]
    close_neg = [float(s["score"]) for s in close_target_negatives]

    open_uncertain = [i.score for i in items if i.kind == "open"]
    close_uncertain = [i.score for i in items if i.kind == "close"]

    def _savings(item: ReviewItem) -> float:
        if item.kind == "open":
            return _expected_savings(item.score, open_uncertain, open_pos, open_neg)
        if item.kind == "close":
            return _expected_savings(item.score, close_uncertain, close_pos, close_neg)
        return 0.0

    global_items = [i for i in items if i.kind in ("open", "close")]
    local_items = [i for i in items if i.kind not in ("open", "close")]
    global_items.sort(key=_savings, reverse=True)
    return global_items + local_items
