# partio GUI server

Local FastAPI backend for the partio GUI. Imports `partio.core`/`partio.adapters`/ `partio.cli.library` directly -- no subprocess, no CLI layer in between.

## Run

`partio/cli/library`'s storage paths (`static/feeds.json`, `static/downloads/`, ...) are resolved relative to the process's current working directory, not to this package. Always launch from the partio repo root, or set `PARTIO_ROOT`:

```sh
cd gui/server
uv sync
PARTIO_ROOT=/home/nator/Dev/partio uv run uvicorn app.main:app --reload
```

## Gate

```sh
uv run pytest
```

Runs `ruff check`, `ruff format --check`, `ty check`, and pytest together (see `tests/test_lint.py`), scoped to this project only.
