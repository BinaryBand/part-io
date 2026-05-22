"""Pydantic models for pipeline ``__config__.toml`` files.

Configuration is stable user intent: what snippets to detect and how to cut.
It is separate from ``__state__.toml`` (runtime detection results) so that
config can be committed to version control while state is gitignored.

Example ``__config__.toml``::

    [[snippet]]
    name       = "ad_open"
    seed_path  = "downloads/snippets/open.mp3"
    seed_hash  = "a3f1..."          # SHA-256 of first 64 KB

    [[snippet]]
    name       = "ad_close"
    seed_path  = "downloads/snippets/close.mp3"
    seed_hash  = "b7c2..."

    [[snippet]]
    name       = "intro"
    seed_path  = "downloads/snippets/intro.mp3"
    seed_hash  = "d9e4..."

    [[cut_rule]]
    type         = "pair"
    open_snippet  = "ad_open"
    close_snippet = "ad_close"
    inclusive    = true        # cut open.start → close.end (keeps jingles in removed segment)
    min_gap      = -15.0       # seconds; negative allows the close to overlap the open slightly
    max_gap      = 300.0

    [[cut_rule]]
    type      = "trim_before"
    snippet   = "intro"
    exclusive = true           # trim to intro.start (keeps jingle); false → trim to intro.end
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from part_io.models.pipeline.state import TargetStateModel


class AudioSnippetModel(BaseModel):
    """One detectable audio snippet and its accumulated training examples.

    ``targets`` stores the confirmed positive/negative hits collected during
    interactive review.  They are used to compute per-snippet classification
    thresholds (MoE/t-distribution) and to build consensus detection profiles.
    """

    model_config = ConfigDict(extra="forbid")

    name: str  # logical key, e.g. "ad_open", "intro"
    seed_path: str  # path to the seed audio file
    seed_hash: str | None = None  # SHA-256 of first 64 KB; None = unverified
    targets: TargetStateModel = Field(default_factory=TargetStateModel)


class PairCutRuleModel(BaseModel):
    """Cut the region bounded by a matched open/close snippet pair.

    ``inclusive = True``  → cut from ``open.start`` to ``close.end``
                             (the jingle audio is removed along with the ad).
    ``inclusive = False`` → cut from ``open.end``   to ``close.start``
                             (jingles are kept; only the ad content is removed).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["pair"]
    open_snippet: str  # name of the opening-boundary snippet
    close_snippet: str  # name of the closing-boundary snippet
    inclusive: bool = True
    min_gap: float = -15.0  # minimum seconds between open.end and close.start
    max_gap: float = 300.0  # maximum seconds between open.end and close.start


class TrimBeforeRuleModel(BaseModel):
    """Remove everything before a matched snippet (e.g. trim to the intro).

    ``exclusive = True``  → trim to ``snippet.start`` (keep the jingle).
    ``exclusive = False`` → trim to ``snippet.end``   (discard the jingle too).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["trim_before"]
    snippet: str
    exclusive: bool = True


class TrimAfterRuleModel(BaseModel):
    """Remove everything after a matched snippet (e.g. trim at the outro).

    ``exclusive = True``  → trim at ``snippet.end``   (keep the jingle).
    ``exclusive = False`` → trim at ``snippet.start`` (discard the jingle too).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["trim_after"]
    snippet: str
    exclusive: bool = True


CutRuleModel = Annotated[
    Union[PairCutRuleModel, TrimBeforeRuleModel, TrimAfterRuleModel],
    Field(discriminator="type"),
]


class PipelineConfigModel(BaseModel):
    """Root model for ``__config__.toml``.

    ``snippets`` defines what to detect; ``cut_rules`` defines what to do with
    the detections.  Any snippet name referenced in a cut rule must exist in
    ``snippets``.
    """

    model_config = ConfigDict(extra="forbid")

    snippets: list[AudioSnippetModel] = Field(default_factory=list)
    cut_rules: list[CutRuleModel] = Field(default_factory=list)

    def snippet(self, name: str) -> AudioSnippetModel | None:
        """Return the snippet with the given name, or None."""
        return next((s for s in self.snippets if s.name == name), None)
