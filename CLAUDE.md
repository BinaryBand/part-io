# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
poetry install --with dev                # install (dev tools included)
poetry run pytest                        # all tests (e2e-marked tests excluded by default)
poetry run pytest tests/unit/audio/test_matcher.py            # one file
poetry run pytest tests/unit/audio/test_matcher.py::test_name # one test
poetry run pytest -m e2e                 # only e2e tests (need external systems)
poetry run part-io-tasks lint --profile strict                # full lint suite
poetry run python -m part_io.cli.lint.<tool>                  # one lint task (ruff, ty, semgrep, vulture, lizard, cpd, coverage)
poetry run semgrep scan --config config/semgrep part_io tests --error  # architecture policies
poetry run part-io-audio-review          # audio review bundle CLI
```

CPD checks require Node.js (they shell out to `npx jscpd`). Audio tooling requires `ffmpeg` on PATH.

WARNING: the root `conftest.py` runs `ruff format` and `ruff check --fix` on the whole repo at the start of every pytest session, so running tests mutates the working tree.

## Architecture

The package is layered, and the layering is machine-enforced (see below), so putting code in the wrong layer fails CI rather than just review:

- `part_io/models/` -- pure data (pydantic models, TaskRegistry, results) plus `models/ports/`, which defines callable/Protocol port types. May import only `part_io.models.*`. No print, subprocess, or file I/O.
- `part_io/services/` -- orchestration logic (e.g. `lint_orchestrator`). Receives port implementations as function arguments; may import only `part_io.services.*` and `part_io.models.*`. No print, subprocess, or file I/O.
- `part_io/adapters/` -- implementations of the ports: config loaders (`config/`), process runner (`process/`), lint runner (`lint/`), JSON report writer (`reporting/`), and the audio matcher/evaluation (`audio/`). May import `part_io.adapters.*`, `part_io.models.*`, `part_io.utils.*`.
- `part_io/utils/` -- leaf helpers; may import only `part_io.utils.*`. `utils/exec.py` is the ONLY module allowed to import `subprocess`; everything else must go through it or `adapters/process/runner.py`.
- `part_io/cli/` -- entrypoints only. This is the only layer allowed to call `print()` and `sys.exit()`. `cli/lint/` holds one module per lint tool, each runnable via `python -m`.

Enforcement lives in three places:

1. `config/semgrep/` -- `dependencies.yml` (import direction between layers), `golden-rules.yml` (no print/subprocess/file-I/O in models/services), `boundaries.yml` (no raw subprocess, eval/exec, unsafe yaml, dynamic importlib), `scaffold.yml` (which files/dirs may exist in each layer; `tests/unit/` must mirror `part_io/`'s top-level shape).
1. `tests/architecture/test_architecture_guardrails.py` -- AST-based checks that core modules do not import CLI and that print/sys.exit stay in entrypoints.
1. `static/rules/ast-grep/` (via `sgconfig.yml`) -- ast-grep rules, e.g. no guarded imports.

When adding a new lint tool: create `part_io/cli/lint/<tool>.py`, register it in `config/tasks.toml` (target `lint.<tool>`, module path, profile membership), and put its settings in `config/lint.toml`. The task registry is loaded from `config/tasks.toml` by `adapters/config/task_registry_loader.py`; per-tool thresholds (complexity, coverage floor, etc.) come from `config/lint.toml`.

The second domain is audio sample matching: `adapters/audio/matcher.py` decodes audio to mono 16 kHz PCM via ffmpeg and matches a reference sample against a longer recording using 32-band spectral-energy features plus deltas. `cli/audio_search.py` prints match timestamps; `cli/audio_review.py` extracts MP3 clips and writes a review manifest. `adapters/audio/evaluation.py` scores labeled review results.

## Conventions

- Tests are split into `tests/architecture/`, `tests/integration/`, and `tests/unit/`; unit test directories mirror the `part_io/` package layout (semgrep-enforced).
- Ruff runs with pydocstyle (Google convention); docstrings are required except in `cli/`, `models/`, and conftest files.
- Type checking is `ty` (not mypy) with `error-on-warning = true`.
