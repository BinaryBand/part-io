# part-io

A small Python toolkit for task orchestration and lint automation, with strong architecture guardrails.

## Features

- Typed task registry and profile-driven task selection.
- Lint orchestration via module entrypoints.
- Architecture and boundary enforcement with Semgrep.
- Focused adapters for config loading, process execution, and reporting.

## Requirements

- Python 3.11+
- Poetry
- Node.js (for `npx`-based CPD checks)

## Installation

```bash
poetry install --with dev
```

## Common Commands

Run lint profile:

```bash
poetry run part-io-tasks lint --profile strict
```

Run tests:

```bash
poetry run pytest
```

Run architecture/security checks:

```bash
poetry run semgrep scan --config config/semgrep part_io tests --error
```

Run CPD check directly:

```bash
poetry run python -m part_io.cli.lint.cpd
```

## Project Layout

- `part_io/` application package.
- `config/` lint and Semgrep policy configuration.
- `tests/` architecture, integration, and unit tests.

## License

MIT. See `LICENSE`.
