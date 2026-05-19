"""Pydantic schema models for remote pipeline `__state__.toml` files."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Classification = Literal["positive", "negative", "uncertain", "undetected"]


class SegmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    start: float
    end: float
    score: float


class MatchModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    start: float
    end: float


class TargetStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positives: list[SegmentModel] = Field(default_factory=list)
    negatives: list[SegmentModel] = Field(default_factory=list)


class RunSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    z_threshold: float | None = None
    step_seconds: float = 0.1
    workers: int = 2
    max_matches: int = 3
    min_gap: float = -15.0
    max_gap: float = 300.0
    yes: bool = False
    dry_run: bool = False
    inclusive: bool = False
    fade: float = 0.5
    quiz_size: int = 10
    no_interactive: bool = False
    overwrite: bool = False
    snippets_dir: str = "downloads/snippets"
    open_sample: str = "open.mp3"
    close_sample: str = "close.mp3"
    output_dir: str = "downloads/remove"


class EpisodeStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = ""
    open_candidates: list[MatchModel] = Field(default_factory=list)
    open_class: Classification = "undetected"
    close_candidates: list[MatchModel] = Field(default_factory=list)
    close_class: Classification = "undetected"
    cut: bool = False


class RemotePipelineTargetsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open: TargetStateModel = Field(default_factory=TargetStateModel)
    close: TargetStateModel = Field(default_factory=TargetStateModel)


class RemotePipelineStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: RunSettingsModel = Field(default_factory=RunSettingsModel)
    targets: RemotePipelineTargetsModel = Field(default_factory=RemotePipelineTargetsModel)
    episodes: dict[str, EpisodeStateModel] = Field(default_factory=dict)


class GenericStateModel(BaseModel):
    """Lenient base model for future state files without a strict schema yet."""

    model_config = ConfigDict(extra="allow")

    schema_version: int | str | None = None
