"""Pydantic models for normalized task execution results and reporting."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TaskRunResult(BaseModel):
    """Execution outcome for one task."""

    model_config = ConfigDict(frozen=True)

    target: str = Field(..., description="Task target ID (e.g. 'lint.vulture')")
    module: str = Field(..., description="Python module path")
    required: bool = Field(..., description="Whether task is required to pass")
    exit_code: int = Field(..., description="Process exit code")
    duration_ms: float = Field(..., description="Execution time in milliseconds")

    @computed_field
    @property
    def status(self) -> str:
        """Return normalized pass/fail status."""
        return "passed" if self.exit_code == 0 else "failed"


class LintRunReport(BaseModel):
    """Aggregate lint execution report."""

    model_config = ConfigDict(frozen=True)

    selected_profile: str | None = Field(None, description="Profile (null for targets)")
    results: list[TaskRunResult] = Field(..., description="Task results in order")
    exit_code: int = Field(..., description="Exit code (first failure or 0)")
    generated_at: str | None = Field(None, description="ISO8601 timestamp if set")

    @computed_field
    @property
    def selected_targets(self) -> list[str]:
        """Return task target IDs in execution order."""
        return [item.target for item in self.results]

    @computed_field
    @property
    def task_count(self) -> int:
        """Return total number of tasks executed."""
        return len(self.results)

    @computed_field
    @property
    def failed_count(self) -> int:
        """Return number of failed tasks."""
        return sum(1 for item in self.results if item.status == "failed")
