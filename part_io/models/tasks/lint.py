"""Pydantic models for lint tool registration metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolSpec(BaseModel):
    """Typed metadata for one executable lint tool registration."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(..., description="Internal tool key")
    executable: str = Field(..., description="Executable name passed to adapter")
    config_section: str | None = Field(None, description="Optional config/lint.toml section")

    @field_validator("key", "executable", mode="before")
    @classmethod
    def validate_non_empty_string(cls, value: object) -> str:
        """Ensure required string fields are non-empty."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected non-empty string")
        return value.strip()
