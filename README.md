# partio

An audio jingle-matching toolkit: locate and extract recurring jingles/stingers inside long episode recordings, with strong architecture guardrails.

## Features

- Cold-start jingle discovery via human-in-the-loop tiled scanning and bisection.
- Spectral-feature reference matching to find a known jingle across episodes.
- Architecture and boundary enforcement via ruff, ty, import-linter, vulture, and ast-grep.

## Requirements

- Python 3.11+
- uv
- ffmpeg (for audio decoding/extraction)

## Installation

```bash
uv sync --all-groups
```

## Common Commands

Run tests (includes lint/type/architecture checks via `tests/test_lint.py`):

```bash
uv run pytest
```

Run the CLI:

```bash
uv run partio
```

## Project Layout

- `partio/` application package, in three layers: `cli/` (Typer entry points) ->
  `adapters/` (I/O implementations) -> `core/` (pure business logic and ports).
- `tests/` architecture, integration, and unit tests (`tests/unit/` mirrors
  `partio/` 1:1).

## License

MIT. See `LICENSE`.
