"""Compile generated artifacts for part-io models."""

from __future__ import annotations

import json
from pathlib import Path

from part_io.models.remote_state import GenericStateModel, RemotePipelineStateModel


def _write_schema(path: Path, schema: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    schemas_dir = Path(__file__).resolve().parents[2] / "models" / "schemas"
    _write_schema(
        schemas_dir / "remote_pipeline_state.schema.json",
        RemotePipelineStateModel.model_json_schema(),
    )
    _write_schema(
        schemas_dir / "generic_state.schema.json",
        GenericStateModel.model_json_schema(),
    )


if __name__ == "__main__":
    main()
