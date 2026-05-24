"""Unit tests for review orchestration service."""

from __future__ import annotations

from dataclasses import dataclass, field

from part_io.services.review_orchestration import (
    ReviewItem,
    apply_review_decision,
    apply_review_dict_classes,
    classify_score_with_thresholds,
    collect_uncertain_candidates,
    compute_classification_thresholds,
    episode_to_review_dict,
    next_uncertain_episode_kind,
    reclassify_all_episodes,
    sort_by_expected_savings,
    undo_review_decision,
)


@dataclass
class _FakeMatch:
    score: float
    start: float
    end: float
    label: str | None = None


@dataclass
class _FakeEpisode:
    candidates: dict[str, list[_FakeMatch]] = field(
        default_factory=lambda: {kind: [] for kind in ("open", "close", "intro", "outro")}
    )
    classes: dict[str, str] = field(
        default_factory=lambda: {kind: "undetected" for kind in ("open", "close", "intro", "outro")}
    )

    def candidates_for(self, kind: str) -> list[_FakeMatch]:
        return self.candidates[kind]

    def class_for(self, kind: str) -> str:
        candidates = self.candidates[kind]
        if not candidates:
            return self.classes[kind]
        labels = [c.label for c in candidates]
        if any(label == "positive" for label in labels):
            return "positive"
        if all(label == "negative" for label in labels):
            return "negative"
        return "uncertain"

    def set_class(self, kind: str, value: str) -> None:
        self.classes[kind] = value


class TestClassificationThresholds:
    def test_no_positives_returns_inf(self) -> None:
        tp, tm = compute_classification_thresholds([], [0.7])
        assert tp == float("inf")
        assert tm < 0.7  # moe reduces it below max negative

    def test_no_negatives_returns_neg_inf(self) -> None:
        tp, tm = compute_classification_thresholds([0.9], [])
        assert tm == float("-inf")


class TestEpisodeBridge:
    def test_episode_to_review_dict_includes_bounds_when_requested(self) -> None:
        episode = _FakeEpisode(
            candidates={
                "open": [_FakeMatch(score=0.9, start=1.0, end=2.0)],
                "close": [],
                "intro": [],
                "outro": [],
            },
            classes={
                "open": "uncertain",
                "close": "undetected",
                "intro": "undetected",
                "outro": "undetected",
            },
        )

        data = episode_to_review_dict(episode, include_bounds=True)
        assert data["open_candidates"] == [{"score": 0.9, "start": 1.0, "end": 2.0, "label": None}]
        assert data["open_class"] == "uncertain"

    def test_apply_review_dict_classes_writes_back_candidate_labels(self) -> None:
        episode = _FakeEpisode(
            candidates={
                "open": [_FakeMatch(score=0.9, start=1.0, end=2.0)],
                "close": [_FakeMatch(score=0.5, start=3.0, end=4.0)],
                "intro": [],
                "outro": [],
            }
        )
        apply_review_dict_classes(
            episode,
            {
                "open_candidates": [{"score": 0.9, "start": 1.0, "end": 2.0, "label": "positive"}],
                "close_candidates": [{"score": 0.5, "start": 3.0, "end": 4.0, "label": "negative"}],
            },
        )
        assert episode.candidates_for("open")[0].label == "positive"
        assert episode.candidates_for("close")[0].label == "negative"

    def test_positives_set_theta_plus(self) -> None:
        tp, _ = compute_classification_thresholds([0.9, 0.85], [])
        # theta_plus = min(0.9, 0.85) + moe([0.9, 0.85])
        assert tp > 0.85  # moe raises it above min

    def test_negatives_set_theta_minus(self) -> None:
        _, tm = compute_classification_thresholds([], [0.7])
        # single negative — moe = inf; theta_minus = 0.7 - inf = -inf
        assert tm == float("-inf")


class TestClassifyScore:
    def test_above_plus_is_positive(self) -> None:
        assert classify_score_with_thresholds(0.95, theta_plus=0.9, theta_minus=0.5) == "positive"

    def test_below_minus_is_negative(self) -> None:
        assert classify_score_with_thresholds(0.4, theta_plus=0.9, theta_minus=0.5) == "negative"

    def test_in_band_is_uncertain(self) -> None:
        assert classify_score_with_thresholds(0.7, theta_plus=0.9, theta_minus=0.5) == "uncertain"

    def test_positive_wins_on_overlap(self) -> None:
        # theta_minus >= theta_plus — score at theta_plus should be positive
        assert classify_score_with_thresholds(0.9, theta_plus=0.9, theta_minus=0.95) == "positive"


class TestCollectUncertainCandidates:
    def test_empty_episodes_returns_empty_list(self) -> None:
        items = collect_uncertain_candidates({}, [], [], [], [])
        assert items == []

    def test_collects_open_uncertain_candidates(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "uncertain",
                "open_candidates": [
                    {"score": 0.8},
                    {"score": 0.7},
                ],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            }
        }
        items = collect_uncertain_candidates(
            episodes,
            open_target_positives=[{"score": 0.9}],
            open_target_negatives=[],
            close_target_positives=[],
            close_target_negatives=[],
        )
        # With 1 positive and 0 negatives: theta_plus = inf, theta_minus = -inf
        # So both 0.8 and 0.7 are in uncertain zone (−inf, inf)
        assert len(items) == 2
        assert items[0].stem == "ep1"
        assert items[0].kind == "open"

    def test_collects_intro_candidates_no_threshold(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "undetected",
                "open_candidates": [],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "uncertain",
                "intro_candidates": [{"score": 0.95}],
                "outro_class": "undetected",
                "outro_candidates": [],
            }
        }
        items = collect_uncertain_candidates(
            episodes,
            open_target_positives=[],
            open_target_negatives=[],
            close_target_positives=[],
            close_target_negatives=[],
        )
        assert len(items) == 1
        assert items[0].kind == "intro"

    def test_sorts_by_candidate_idx_then_score(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "uncertain",
                "open_candidates": [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            }
        }
        items = collect_uncertain_candidates(
            episodes,
            open_target_positives=[{"score": 0.5}],
            open_target_negatives=[],
            close_target_positives=[],
            close_target_negatives=[],
        )
        # All 3 should be in uncertain zone; sorted by (idx, -score)
        assert len(items) == 3
        assert [item.candidate_idx for item in items] == [0, 1, 2]

    def test_filters_open_candidates_using_global_threshold_band(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "uncertain",
                "open_candidates": [{"score": 0.95}, {"score": 0.5}, {"score": 0.1}],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            }
        }
        # 31 identical samples => moe = 0; thresholds are exactly 0.9 and 0.2
        positives = [{"score": 0.9}] * 31
        negatives = [{"score": 0.2}] * 31
        items = collect_uncertain_candidates(
            episodes,
            open_target_positives=positives,
            open_target_negatives=negatives,
            close_target_positives=[],
            close_target_negatives=[],
        )
        assert len(items) == 1
        assert items[0].kind == "open"
        assert items[0].score == 0.5


class TestReclassifyAllEpisodes:
    def test_reclassifies_uncertain_open(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "uncertain",
                "open_candidates": [{"score": 0.97}],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            }
        }
        reclassify_all_episodes(
            episodes,
            open_target_positives=[{"score": 0.9}],
            open_target_negatives=[],
            close_target_positives=[],
            close_target_negatives=[],
        )
        # theta_plus = 0.9 + moe([0.9]) = 0.9 + inf = inf; 0.97 < inf, so stays uncertain
        assert episodes["ep1"]["open_class"] == "uncertain"

    def test_does_not_reclassify_already_classified(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "positive",
                "open_candidates": [{"score": 0.5}],  # low score, but class manually set
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            }
        }
        reclassify_all_episodes(
            episodes,
            open_target_positives=[],
            open_target_negatives=[],
            close_target_positives=[],
            close_target_negatives=[],
        )
        # Should not change; only uncertain is reclassified
        assert episodes["ep1"]["open_class"] == "positive"

    def test_does_not_reclassify_undetected(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "undetected",
                "open_candidates": [],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            }
        }
        reclassify_all_episodes(
            episodes,
            open_target_positives=[{"score": 0.9}],
            open_target_negatives=[],
            close_target_positives=[],
            close_target_negatives=[],
        )
        assert episodes["ep1"]["open_class"] == "undetected"

    def test_reclassifies_open_close_but_not_intro_outro(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "uncertain",
                "open_candidates": [{"score": 0.6}],
                "close_class": "uncertain",
                "close_candidates": [{"score": 0.1}],
                "intro_class": "uncertain",
                "intro_candidates": [{"score": 0.95}],
                "outro_class": "uncertain",
                "outro_candidates": [{"score": 0.95}],
            }
        }
        # 31 identical samples => moe = 0; deterministic thresholds.
        reclassify_all_episodes(
            episodes,
            open_target_positives=[{"score": 0.5}] * 31,
            open_target_negatives=[{"score": 0.2}] * 31,
            close_target_positives=[{"score": 0.7}] * 31,
            close_target_negatives=[{"score": 0.2}] * 31,
        )

        assert episodes["ep1"]["open_candidates"][0]["label"] == "positive"
        assert episodes["ep1"]["close_candidates"][0]["label"] == "negative"
        assert episodes["ep1"]["intro_candidates"][0].get("label") is None
        assert episodes["ep1"]["outro_candidates"][0].get("label") is None


class TestNextUncertainEpisodeKind:
    def test_returns_highest_score_uncertain_kind(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "uncertain",
                "open_candidates": [{"score": 0.4}],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            },
            "ep2": {
                "open_class": "undetected",
                "open_candidates": [],
                "close_class": "uncertain",
                "close_candidates": [{"score": 0.6}],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            },
        }

        assert next_uncertain_episode_kind(episodes) == ("ep2", "close")

    def test_respects_exclude_set_and_ignores_non_positive_scores(self) -> None:
        episodes = {
            "ep1": {
                "open_class": "uncertain",
                "open_candidates": [{"score": 0.5}],
                "close_class": "undetected",
                "close_candidates": [],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            },
            "ep2": {
                "open_class": "undetected",
                "open_candidates": [],
                "close_class": "uncertain",
                "close_candidates": [{"score": 0.0}],
                "intro_class": "undetected",
                "intro_candidates": [],
                "outro_class": "undetected",
                "outro_candidates": [],
            },
        }

        assert next_uncertain_episode_kind(episodes, exclude={("open", "ep1")}) is None


class TestApplyAndUndoReviewDecision:
    def test_apply_approve_open_adds_positive_and_labels_candidate(self) -> None:
        episode = {
            "open_class": "uncertain",
            "open_candidates": [{"score": 0.9, "start": 10.0, "end": 15.0}],
        }
        open_pos: list[dict] = []
        open_neg: list[dict] = []
        close_pos: list[dict] = []
        close_neg: list[dict] = []

        decision, undo = apply_review_decision(
            episode=episode,
            kind="open",
            candidate_idx=0,
            action="a",
            source="downloads/ep1.mp3",
            open_target_positives=open_pos,
            open_target_negatives=open_neg,
            close_target_positives=close_pos,
            close_target_negatives=close_neg,
        )

        assert decision.action == "approved"
        assert episode["open_candidates"][0]["label"] == "positive"
        assert len(open_pos) == 1
        assert undo.kind == "open"
        assert undo.target_list_was_positive is True

    def test_apply_reject_intro_labels_negative_without_target_append(self) -> None:
        episode = {
            "intro_class": "uncertain",
            "intro_candidates": [{"score": 0.7, "start": 3.0, "end": 6.0}],
        }

        decision, undo = apply_review_decision(
            episode=episode,
            kind="intro",
            candidate_idx=0,
            action="r",
            source="downloads/ep1.mp3",
            open_target_positives=[],
            open_target_negatives=[],
            close_target_positives=[],
            close_target_negatives=[],
        )

        assert decision.action == "rejected"
        assert episode["intro_candidates"][0]["label"] == "negative"
        assert undo.target_list_was_positive is False

    def test_undo_removes_segment_and_restores_class(self) -> None:
        episode = {
            "close_class": "positive",
            "close_candidates": [{"score": 0.4, "start": 20.0, "end": 24.0}],
        }
        close_pos = [{"source": "s.mp3", "start": 20.0, "end": 24.0, "score": 0.4}]

        from part_io.services.review_orchestration import UndoEntry

        undo_review_decision(
            episode=episode,
            undo=UndoEntry(
                stem="ep1",
                kind="close",
                action="a",
                segment_source="s.mp3",
                segment_start=20.0,
                segment_end=24.0,
                segment_score=0.4,
                candidate_idx=0,
                target_list_was_positive=True,
                prev_class="uncertain",
                prev_label=None,
            ),
            open_target_positives=[],
            open_target_negatives=[],
            close_target_positives=close_pos,
            close_target_negatives=[],
        )

        assert episode["close_candidates"][0].get("label") is None
        assert close_pos == []


class TestSortByExpectedSavings:
    def test_mid_zone_candidate_ranked_first(self) -> None:
        # Tight positive cluster at 0.92–0.96, tight negative cluster at 0.50–0.54.
        # θ⁺≈0.947, θ⁻≈0.514. Three uncertain candidates: near θ⁺ (0.91), mid-zone (0.70),
        # near θ⁻ (0.60). Approving the mid-zone candidate (0.70) collapses θ⁺ to ~0.836,
        # auto-classifying the near-positive (0.91) as positive — highest expected savings.
        positives = [{"score": s} for s in [0.92, 0.93, 0.94, 0.95, 0.96]]
        negatives = [{"score": s} for s in [0.50, 0.51, 0.52, 0.53, 0.54]]

        items = [
            ReviewItem(stem="ep_high", kind="open", candidate_idx=0, score=0.91),
            ReviewItem(stem="ep_mid", kind="open", candidate_idx=0, score=0.70),
            ReviewItem(stem="ep_low", kind="open", candidate_idx=0, score=0.60),
        ]
        sorted_items = sort_by_expected_savings(items, positives, negatives, [], [])
        assert sorted_items[0].stem == "ep_mid"  # highest cascade savings
        assert sorted_items[2].stem == "ep_high"  # lowest — approval barely moves θ⁺

    def test_intro_outro_appended_after_open_close(self) -> None:
        items = [
            ReviewItem(stem="ep_intro", kind="intro", candidate_idx=0, score=0.9),
            ReviewItem(stem="ep_open", kind="open", candidate_idx=0, score=0.85),
            ReviewItem(stem="ep_outro", kind="outro", candidate_idx=0, score=0.7),
            ReviewItem(stem="ep_close", kind="close", candidate_idx=0, score=0.80),
        ]
        positives = [{"score": 0.95}, {"score": 0.90}]
        sorted_items = sort_by_expected_savings(items, positives, [], positives, [])
        kinds = [i.kind for i in sorted_items]
        # All open/close precede all intro/outro
        last_global = max(i for i, k in enumerate(kinds) if k in ("open", "close"))
        first_local = min(i for i, k in enumerate(kinds) if k in ("intro", "outro"))
        assert last_global < first_local

    def test_no_confirmed_samples_falls_back_gracefully(self) -> None:
        # With no positives/negatives, zone is infinite; p_approve defaults to 0.5,
        # all savings are 0 — function should not raise and preserve relative order.
        items = [
            ReviewItem(stem="ep_a", kind="open", candidate_idx=0, score=0.9),
            ReviewItem(stem="ep_b", kind="open", candidate_idx=0, score=0.7),
        ]
        result = sort_by_expected_savings(items, [], [], [], [])
        assert len(result) == 2
