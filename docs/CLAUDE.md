# Architecture & Lint Rules for AI Assistants

This document captures the architectural invariants, lint rules, and quality gates
enforced by the automated test suite. Violating any of these will cause CI to fail.

---

## Package Structure

Only these top-level packages are allowed under `part_io/`:

```text
part_io/
  adapters/   – I/O, external-tool wrappers, file readers/writers
  cli/        – entrypoints (print/sys.exit allowed here only)
  models/     – pure data; no I/O, no subprocess, no print
  services/   – orchestration; no I/O, no subprocess, no print
  utils/      – shared helpers; subprocess allowed ONLY in utils/exec.py
```

Stable sub-shapes enforced by semgrep (`config/semgrep/scaffold.yml`):

| Package | Allowed modules |
| --- | --- |
| `adapters/` | `audio/`, `config/`, `lint/`, `process/`, `reporting/`, `errors.py` |
| `models/` | `lint.py`, `registry.py`, `results.py`, `ports/` |
| `cli/lint/` | `cpd`, `coverage`, `entrypoints`, `execution`, `lizard`, `registry`, `ruff`, `semgrep`, `ty`, `vulture` |
| `services/` | `lint_orchestrator.py` |

---

## Import Boundaries

Enforced by `config/semgrep/dependencies.yml` (severity: ERROR).

```text
models      → models only
services    → services, models
adapters    → adapters, models, utils
utils       → utils only
cli         → anything (entrypoints sit at the top)
```

Tests outside `tests/integration/` and `tests/unit/cli/` must not import `part_io.cli.*`.

---

## Print / sys.exit

Enforced by `tests/architecture/test_architecture_guardrails.py` (AST scan).

- **Only** files listed in `ENTRYPOINT_PATHS` or inside `ENTRYPOINT_DIRS` (`cli/lint/`) may call `print()` or `sys.exit()`.
- `models/`, `services/`, and `adapters/` must never call `print()` (use `logging`).

Current registered entrypoints:

- `cli/tasks.py`
- `cli/audio_search.py`, `cli/audio_review.py`, `cli/audio_review_batch.py`
- `cli/audio_detect_ads.py`, `cli/audio_detect_ads_batch.py`
- `cli/audio_ad_detect.py`, `cli/audio_ad_remove.py`
- `cli/download_unmatched.py`, `cli/generate_rss.py`, `cli/remote_pipeline.py`

**Adding a new CLI file that calls `print()` requires adding it to `ENTRYPOINT_PATHS`.**

---

## Subprocess

Enforced by `config/semgrep/boundaries.yml` (severity: ERROR).

- `import subprocess` / `from subprocess import …` is only allowed in `part_io/utils/exec.py`.
- All other code must use `part_io.utils.exec.run_resolved()` or `launch_resolved()`.
- Exception pattern for type-checking stubs in CLI files:

  ```python
  if TYPE_CHECKING:
      import subprocess  # nosemgrep: no-direct-subprocess-import-except-in-utils
  ```

---

## Security Rules (`config/semgrep/boundaries.yml`)

These are all severity ERROR and block the test run:

| Rule | What is banned |
| --- | --- |
| `no-eval-exec-compile` | `eval()`, `exec()`, `compile()`, `__import__()` |
| `no-dynamic-importlib` | `importlib.import_module($NAME)` with a runtime string |
| `no-yaml-load-unsafe` | `yaml.load()` without `Loader=yaml.SafeLoader` |
| `no-pickle-unsafe` | `import pickle/marshal/shelve`, `pickle.loads()`, etc. |

---

## Complexity Limits (Lizard)

Configured in `config/lint.toml`. The runner exits 1 on **any** warning.

| Metric | Limit |
| --- | --- |
| Cyclomatic complexity (CCN) | 15 |
| Function length (physical lines) | 60 |

Scope: `part_io/` only. When a function approaches these limits, extract helpers.

---

## Ruff

Line length: **100**. Configured in `pyproject.toml`.

Selected rule sets: `E, F, W, I, N, A, B, S, D` (Google docstring convention).

Per-file ignores:

- `**/cli/**/*.py` → `D` (no docstrings required), `S603`, `S607`
- `**/models/**/*.py` → `D`
- `tests/**/*.py` → `D`, `S101` (assert is fine in tests)
- `**/conftest.py` → `D`
- `adapters/audio/ad_segments.py` → `D102`

---

## Type Checking (ty)

- `error-on-warning = true` — warnings are treated as errors.
- Avoid `try/except ImportError` for stdlib modules that exist in Python 3.11+
  (e.g. `tomllib`). Import directly; `ty` raises `conflicting-declarations` otherwise.
- For `subprocess` types needed only at type-check time, use `TYPE_CHECKING` guard
  with `nosemgrep` comment (see Subprocess section above).

---

## Code Duplication (jscpd)

Threshold: **0%** (zero tolerance). Config in `config/jscpd.json`.

Duplicate blocks of ≥5 lines across production code will fail the CPD test.
Extract shared logic to a common helper rather than copying.

---

## Test Coverage

Floor: **80%** (`config/lint.toml` → `[coverage] floor = 80`).

---

## Test File Layout

`tests/unit/` must mirror the `part_io/` top-level shape:

```text
tests/unit/adapters/   tests/unit/audio/   tests/unit/cli/
tests/unit/models/     tests/unit/services/ tests/unit/utils/
```

Adding a new top-level package under `part_io/` requires a matching `tests/unit/<pkg>/`
directory (enforced by `config/semgrep/scaffold.yml`).
