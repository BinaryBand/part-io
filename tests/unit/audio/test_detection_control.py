"""Control tests against audited ground-truth episodes in downloads/media/.

All confirmed positive positions were hand-verified by the operator and recorded
in downloads/media/__state__.toml.  These tests fail if the detector regresses
on known-positive episodes, providing a fast sanity-check before manual review.

Each episode contains the open/close jingle THREE times.  All three positions
are asserted so the test catches both missed detections and score degradation.

Skipped automatically when the media files are absent (CI / fresh checkout).

----

KEY FINDINGS (see diagnosis notes in-line):

  open.mp3 baseline score: 0.93–0.95 across the entire episode.
    True positives score only 0.001–0.015 above the local background.
    The 3rd occurrence in dece9384 scores 0.9487, with the nearest false
    positive at 0.9484 — a gap of 0.0003 that no fixed threshold can reliably
    use.  However, the true positives ARE the top-N global scorers after NMS,
    so a top-N approach (matching the pipeline's max_matches behaviour) works.

  close.mp3 baseline: well below 0.8 (unlike open), so the default
    score_threshold=0.8 works cleanly.

  intro.mp3 positional offset: the detected start is ~snippet_duration
    seconds BEFORE the actual intro onset.  The clip played during review ends
    right as the intro begins, not in the middle of it.  Root cause: intro.mp3
    was recorded starting before the jingle onset, so the distinctive audio is
    at the TAIL of the 24-second file.  The cross-correlation correctly aligns
    the snippet, but the reported start is 24 seconds early.
    Fix: trim intro.mp3 so it starts at the jingle onset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from part_io.adapters.audio.matcher import find_audio_sample_matches

ROOT = Path(__file__).resolve().parents[3]
MEDIA = ROOT / "downloads" / "media"
SNIPPETS = ROOT / "downloads" / "snippets"

OPEN_SNIPPET = SNIPPETS / "open.mp3"
CLOSE_SNIPPET = SNIPPETS / "close.mp3"

# Ground-truth positions (seconds) audited in downloads/media/__state__.toml.
# Tolerance ±2 s accommodates minor step/hop rounding.
_TOL = 2.0

_DECE = MEDIA / "dece9384-9892-4b4d-9c13-5298e44d67ab.mp3"
_DECE_OPEN_STARTS = [2780.67, 1355.97, 4152.51]
_DECE_CLOSE_STARTS = [1366.02, 2790.72, 4115.90]

_EP45 = MEDIA / "ep_45e2978e.mp3"
_EP45_OPEN_STARTS = [647.10, 2189.82, 3512.06]
_EP45_CLOSE_STARTS = [2198.78, 3520.90, 705.66]


def _has_match_near(matches, target: float, tol: float = _TOL) -> bool:
    return any(abs(m.start_seconds - target) <= tol for m in matches)


def _top_n(matches, n: int):
    """Return the *n* highest-scoring matches (pipeline's max_matches behaviour)."""
    return sorted(matches, key=lambda m: m.score, reverse=True)[:n]


# ---------------------------------------------------------------------------
# dece9384 — open snippet
#
# True positives score 0.9487–0.9688.  The 3rd occurrence at 4152.51s scores
# 0.9487 while the nearest false positive scores 0.9484 — a gap of 0.0003.
# A fixed score threshold cannot reliably separate these; instead we assert
# that all confirmed positions appear in the global top-5 after NMS, which
# mirrors the pipeline's max_matches=3 pruning.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DECE.exists() or not OPEN_SNIPPET.exists(),
    reason="media/snippets not present",
)
def test_open_top5_contains_all_confirmed_in_dece9384() -> None:
    """All three confirmed open positions must appear in the global top-5 by score."""
    matches = find_audio_sample_matches(
        source_path=_DECE,
        sample_path=OPEN_SNIPPET,
        score_threshold=0.9,
        step_seconds=0.1,
    )
    assert matches, "no open matches found above 0.9 — detector returned nothing"
    top = _top_n(matches, 5)
    missing = [s for s in _DECE_OPEN_STARTS if not _has_match_near(top, s)]
    assert not missing, (
        f"confirmed open position(s) {missing}s not in top-5 for dece9384.\n"
        f"Top-5: {[(round(m.start_seconds, 1), m.score) for m in top]}"
    )


# ---------------------------------------------------------------------------
# dece9384 — close snippet
#
# True positives score 0.826–0.853.  The close snippet's baseline is well
# below 0.8 so the default threshold=0.8 works cleanly.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DECE.exists() or not CLOSE_SNIPPET.exists(),
    reason="media/snippets not present",
)
def test_close_detected_in_dece9384() -> None:
    """Close jingle must be found at all three audited positions in dece9384."""
    matches = find_audio_sample_matches(
        source_path=_DECE,
        sample_path=CLOSE_SNIPPET,
        step_seconds=0.1,
    )
    assert matches, "no close matches found — detector returned nothing"
    missing = [s for s in _DECE_CLOSE_STARTS if not _has_match_near(matches, s)]
    assert not missing, (
        f"close jingle not found near {missing}s in dece9384; "
        f"found starts: {sorted(m.start_seconds for m in matches)}"
    )


# ---------------------------------------------------------------------------
# ep_45e2978e — open snippet
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _EP45.exists() or not OPEN_SNIPPET.exists(),
    reason="media/snippets not present",
)
def test_open_top5_contains_all_confirmed_in_ep_45e2978e() -> None:
    """All three confirmed open positions must appear in the global top-5 by score."""
    matches = find_audio_sample_matches(
        source_path=_EP45,
        sample_path=OPEN_SNIPPET,
        score_threshold=0.9,
        step_seconds=0.1,
    )
    assert matches, "no open matches found above 0.9 — detector returned nothing"
    top = _top_n(matches, 5)
    missing = [s for s in _EP45_OPEN_STARTS if not _has_match_near(top, s)]
    assert not missing, (
        f"confirmed open position(s) {missing}s not in top-5 for ep_45e2978e.\n"
        f"Top-5: {[(round(m.start_seconds, 1), m.score) for m in top]}"
    )


# ---------------------------------------------------------------------------
# ep_45e2978e — close snippet
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _EP45.exists() or not CLOSE_SNIPPET.exists(),
    reason="media/snippets not present",
)
def test_close_detected_in_ep_45e2978e() -> None:
    """Close jingle must be found at all three audited positions in ep_45e2978e."""
    matches = find_audio_sample_matches(
        source_path=_EP45,
        sample_path=CLOSE_SNIPPET,
        step_seconds=0.1,
    )
    assert matches, "no close matches found — detector returned nothing"
    missing = [s for s in _EP45_CLOSE_STARTS if not _has_match_near(matches, s)]
    assert not missing, (
        f"close jingle not found near {missing}s in ep_45e2978e; "
        f"found starts: {sorted(m.start_seconds for m in matches)}"
    )
