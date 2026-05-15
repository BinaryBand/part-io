"""Pydantic models for task registry and task metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskSpec(BaseModel):
    """One declared task entry from the registry."""

    model_config = ConfigDict(frozen=True)

    target: str = Field(..., description="Task identifier (e.g. 'lint.vulture')")
    module: str = Field(..., description="Python module path (e.g. 'part_io.cli.lint.vulture')")
    description: str = Field(..., description="Single-line human description")
    required: bool = Field(default=True, description="Whether task is required to pass")

    @field_validator("target", "module", "description", mode="before")
    @classmethod
    def validate_non_empty_string(cls, value: object) -> str:
        """Ensure string fields are non-empty."""
        _ = cls
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected non-empty string")
        return value.strip()


class TaskRegistry(BaseModel):
    """Validated registry containing tasks and profiles."""

    model_config = ConfigDict(frozen=True)

    tasks: dict[str, TaskSpec] = Field(..., description="Tasks by target ID")
    profiles: dict[str, tuple[str, ...]] = Field(..., description="Profile name to task IDs")
    default_profile: str = Field(..., description="Default profile name")

    @field_validator("default_profile", mode="after")
    @classmethod
    def validate_default_profile_exists(cls, value: str, info) -> str:
        """Ensure default_profile is a known profile."""
        _ = cls
        if hasattr(info, "data") and "profiles" in info.data:
            profiles = info.data["profiles"]
            if value not in profiles:
                raise ValueError(f"Unknown default_profile '{value}'")
        return value


