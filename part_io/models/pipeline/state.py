"""Pydantic models for pipeline ``__state__.toml`` files.

This module covers runtime state: what the pipeline has detected so far and
the operational settings for the current run. Snippet profiles and seed paths
are embedded directly in ``__state__.toml`` via the ``[[snippets]]`` table.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SegmentModel(BaseModel):
    """A confirmed positive or negative snippet hit stored in the target list."""

    model_config = ConfigDict(extra="forbid")

    stem: str
    start: float
    end: float
    score: float


class MatchModel(BaseModel):
    """A single detection candidate within one episode."""

    model_config = ConfigDict(extra="forbid")

    score: float
    start: float
    end: float
    label: Literal["positive", "negative"] | None = None


class TargetStateModel(BaseModel):
    """Accumulated positive/negative examples for one snippet type."""

    model_config = ConfigDict(extra="forbid")

    positives: list[SegmentModel] = Field(default_factory=list)
    negatives: list[SegmentModel] = Field(default_factory=list)


class RunSettingsModel(BaseModel):
    """Operational settings persisted in ``[settings]``."""

    model_config = ConfigDict(extra="forbid")

    step_seconds: float = 0.1
    workers: int = 2
    max_matches: int = 3
    min_gap: float = -15.0
    max_gap: float = 300.0
    ad_inclusive: bool = True
    intro_exclusive: bool = True
    fade: float = 0.5
    quiz_size: int = 10
    overwrite: bool = False
    output_dir: str = "downloads/remove"
    debug: bool = False


class EpisodeStateModel(BaseModel):
    """Detection state for one episode."""

    model_config = ConfigDict(extra="forbid")

    source_hash: str | None = None
    open_candidates: list[MatchModel] = Field(default_factory=list)
    close_candidates: list[MatchModel] = Field(default_factory=list)
    intro_candidates: list[MatchModel] = Field(default_factory=list)
    outro_candidates: list[MatchModel] = Field(default_factory=list)
    cut: bool = False


class TargetsByKindModel(BaseModel):
    """Top-level target banks persisted as ``[targets.open]`` / ``[targets.close]``."""

    model_config = ConfigDict(extra="forbid")

    open: TargetStateModel = Field(default_factory=TargetStateModel)
    close: TargetStateModel = Field(default_factory=TargetStateModel)


class SnippetProfileStateModel(BaseModel):
    """Embedded detection profile stored under ``[snippets.profile]``."""

    model_config = ConfigDict(extra="forbid")

    n_frames: int
    analysis_rate: int
    hop_size: int
    band_count: int
    data: str  # base85-encoded byte-shuffled zlib-compressed float32 matrix


class SnippetStateModel(BaseModel):
    """One snippet entry in the ``[[snippets]]`` array-of-tables."""

    model_config = ConfigDict(extra="forbid")

    name: str
    profile: SnippetProfileStateModel


class PipelineStateModel(BaseModel):
    """Root model for ``__state__.toml``."""

    model_config = ConfigDict(extra="forbid")

    targets: TargetsByKindModel = Field(default_factory=TargetsByKindModel)
    settings: RunSettingsModel = Field(default_factory=RunSettingsModel)
    snippets: list[SnippetStateModel] = Field(default_factory=list)
    episodes: dict[str, EpisodeStateModel] = Field(default_factory=dict)


class GenericStateModel(BaseModel):
    """Lenient base for future state files without a strict schema yet."""

    model_config = ConfigDict(extra="allow")

    schema_version: int | str | None = None
