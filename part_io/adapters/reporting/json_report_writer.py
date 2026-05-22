"""JSON report writer adapter for lint execution results.

Callable ports implemented here are defined in part_io.models.ports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from part_io.models.tasks.results import LintRunReport


def write_lint_report(report_path: Path, report: LintRunReport) -> None:
    """Write *report* to *report_path* as formatted JSON."""
    if report.generated_at is None:
        report_data = report.model_copy(
            update={"generated_at": datetime.now(timezone.utc).isoformat()}
        )
    else:
        report_data = report

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": report_data.generated_at,
        "selected_profile": report_data.selected_profile,
        "selected_targets": report_data.selected_targets,
        "task_count": report_data.task_count,
        "failed_count": report_data.failed_count,
        "exit_code": report_data.exit_code,
        "results": [item.model_dump() for item in report_data.results],
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["write_lint_report"]
