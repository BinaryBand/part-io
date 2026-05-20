"""Unit tests for review orchestration service."""

from __future__ import annotations

from part_io.services.review_orchestration import (
    classify_score_with_thresholds,
    collect_uncertain_candidates,
    compute_classification_thresholds,
    reclassify_all_episodes,
)


class TestClassificationThresholds:
    def test_no_positives_returns_inf(self) -> None:
        tp, tm = compute_classification_thresholds([], [0.7])
        assert tp == float("inf")
        assert tm < 0.7  # moe reduces it below max negative

    def test_no_negatives_returns_neg_inf(self) -> None:
        tp, tm = compute_classification_thresholds([0.9], [])
        assert tm == float("-inf")
        assert tp > 0.9  # moe adds to min positive

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

        assert episodes["ep1"]["open_class"] == "positive"
        assert episodes["ep1"]["close_class"] == "negative"
        assert episodes["ep1"]["intro_class"] == "uncertain"
        assert episodes["ep1"]["outro_class"] == "uncertain"
