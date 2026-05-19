"""Pydantic schema models for remote pipeline `__state__.toml` files."""

# poetry run part-io-tasks remote-loop downloads/remote \
#   --output-dir downloads/remote \
#   --inclusive \
#   --yes \
#   --debug

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

    z_threshold: float | None = None  # relative score cutoff (std devs above mean); None = disabled
    step_seconds: float = 0.1  # sliding-window step size during snippet detection
    workers: int = 2  # parallel detection workers per batch
    max_matches: int = 3  # top-N candidates retained per episode per snippet type
    min_gap: float = -15.0  # minimum seconds between open end and close start (allows overlap)
    max_gap: float = 300.0  # maximum seconds between open end and close start
    yes: bool = False  # skip cut confirmation prompts
    dry_run: bool = False  # plan cuts without running ffmpeg
    inclusive: bool = False  # include short files (< 10 MB) that are normally skipped
    fade: float = 0.5  # fade-in/out duration in seconds applied at each cut point
    quiz_size: int = 10  # uncertain candidates to collect before presenting a review quiz
    no_interactive: bool = False  # skip interactive review; generate clips only
    overwrite: bool = False  # re-detect episodes that already have candidates
    snippets_dir: str = "downloads/snippets"  # directory containing open/close/intro snippet files
    open_sample: str = "open.mp3"  # filename of the ad-open snippet inside snippets_dir
    close_sample: str = "close.mp3"  # filename of the ad-close snippet inside snippets_dir
    intro_sample: str = "intro.mp3"  # intro snippet filename in snippets_dir
    outro_sample: str | None = None  # optional outro snippet filename in snippets_dir
    # if intro is detected, trim everything before it
    output_dir: str = "downloads/remove"  # destination directory for cut MP3s
    debug: bool = False  # export planned ad-cut clips into output_dir/debug_ads for inspection


class EpisodeStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = ""
    open_candidates: list[MatchModel] = Field(default_factory=list)
    open_class: Classification = "undetected"
    close_candidates: list[MatchModel] = Field(default_factory=list)
    close_class: Classification = "undetected"
    intro_candidates: list[MatchModel] = Field(default_factory=list)
    intro_class: Classification = "undetected"
    outro_candidates: list[MatchModel] = Field(default_factory=list)
    outro_class: Classification = "undetected"
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
