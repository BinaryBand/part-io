"""Pydantic models for pipeline ``__state__.toml`` files.

This module covers runtime state: what the pipeline has detected so far and
the operational settings for the current run. Audio snippet definitions and
cut rules live in ``config.py`` (``__config__.toml``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Classification = Literal["positive", "negative", "uncertain", "undetected"]


class SegmentModel(BaseModel):
    """A confirmed positive or negative snippet hit stored in the target list."""

    model_config = ConfigDict(extra="forbid")

    source: str
    start: float
    end: float
    score: float


class MatchModel(BaseModel):
    """A single detection candidate within one episode."""

    model_config = ConfigDict(extra="forbid")

    score: float
    start: float
    end: float


class TargetStateModel(BaseModel):
    """Accumulated positive/negative examples for one snippet type.

    Used both here (legacy ``[targets.*]`` in state) and in ``AudioSnippetModel``
    (new design, embedded per snippet in config).
    """

    model_config = ConfigDict(extra="forbid")

    positives: list[SegmentModel] = Field(default_factory=list)
    negatives: list[SegmentModel] = Field(default_factory=list)


class RunSettingsModel(BaseModel):
    """Operational settings persisted in ``[settings]``.

    Fields marked *-> config* will migrate to ``__config__.toml`` once the new
    declarative config is wired in; they remain here for backward compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    step_seconds: float = 0.1  # sliding-window step during detection
    workers: int = 2  # parallel detection workers per batch
    max_matches: int = 3  # top-N candidates retained per episode per snippet
    fade: float = 0.5  # fade-in/out duration (seconds) at each cut point
    quiz_size: int = 10  # uncertain candidates per interactive review batch
    no_interactive: bool = False  # skip review; cut automatically
    overwrite: bool = False  # re-detect episodes that already have candidates
    output_dir: str = "downloads/remove"  # destination for cut MP3s
    debug: bool = False  # export planned cut clips for inspection

    # -> config: these move to AudioSnippetModel / CutRuleModel once wired
    snippets_dir: str = "downloads/snippets"
    open_sample: str = "open.mp3"
    close_sample: str = "close.mp3"
    intro_sample: str = "intro.mp3"
    outro_sample: str | None = None
    min_gap: float = -15.0  # -> PairCutRuleModel.min_gap
    max_gap: float = 300.0  # -> PairCutRuleModel.max_gap
    ad_inclusive: bool = True  # -> PairCutRuleModel.inclusive
    intro_exclusive: bool = True  # -> TrimBeforeRuleModel.exclusive


class EpisodeStateModel(BaseModel):
    """Detection state for one episode.

    Supports both layouts:
    - legacy per-kind keys written by the current runtime
      (``open_candidates``, ``open_class``, etc.)
    - dict-backed shape used by the refactored in-memory model
      (``candidates`` and ``classes``)
    """

    model_config = ConfigDict(extra="forbid")

    source: str = ""
    source_hash: str | None = None  # SHA-256 of first 64 KB; None = unverified
    open_candidates: list[MatchModel] = Field(default_factory=list)
    close_candidates: list[MatchModel] = Field(default_factory=list)
    intro_candidates: list[MatchModel] = Field(default_factory=list)
    outro_candidates: list[MatchModel] = Field(default_factory=list)
    open_class: Classification = "undetected"
    close_class: Classification = "undetected"
    intro_class: Classification = "undetected"
    outro_class: Classification = "undetected"
    candidates: dict[str, list[MatchModel]] = Field(default_factory=dict)
    classes: dict[str, Classification] = Field(default_factory=dict)
    cut: bool = False


class TargetsByKindModel(BaseModel):
    """Legacy top-level target banks persisted as ``[targets.open]`` / ``close``."""

    model_config = ConfigDict(extra="forbid")

    open: TargetStateModel = Field(default_factory=TargetStateModel)
    close: TargetStateModel = Field(default_factory=TargetStateModel)


class PipelineStateModel(BaseModel):
    """Root model for ``__state__.toml``."""

    model_config = ConfigDict(extra="forbid")

    targets: TargetsByKindModel = Field(default_factory=TargetsByKindModel)
    settings: RunSettingsModel = Field(default_factory=RunSettingsModel)
    episodes: dict[str, EpisodeStateModel] = Field(default_factory=dict)


class GenericStateModel(BaseModel):
    """Lenient base for future state files without a strict schema yet."""

    model_config = ConfigDict(extra="allow")

    schema_version: int | str | None = None
