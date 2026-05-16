# part-io

## License

MIT. See `LICENSE`.

## Project Layout

- `part_io/`: application package (CLI, adapters, services, models, utils)
- `config/`: lint policies and generated task config inputs
- `tests/`: architecture, integration, and unit test suites
- `downloads/`: local media/snippet assets and generated review bundles

## Command Reference

Task runner entrypoint:

```bash
poetry run part-io-tasks help
```

Run lint by profile:

```bash
poetry run part-io-tasks lint --profile strict
```

Run tests:

```bash
poetry run part-io-tasks test
```

Generate/check make-style task targets:

```bash
poetry run part-io-tasks generate-tasks
poetry run part-io-tasks check-tasks
```

Run duplicate-code detection:

```bash
poetry run python -m part_io.cli.lint.cpd
```

Run Semgrep policies:

```bash
poetry run semgrep scan --config config/semgrep part_io tests --error
```

## Audio Review Workflows

Single file + sample review bundle:

```bash
poetry run part-io-audio-review \
 downloads/media/ep_45e2978e.mp3 \
 downloads/snippets/close.mp3 \
 --threshold 0.8 \
 --step-seconds 0.1 \
 --max-clips 25 \
 --bundle-name ep_45e2978e/close_high_points \
 --overwrite
```

Batch review all media files (close/open samples):

```bash
poetry run python -m part_io.cli.audio_review_batch \
 --threshold 0.8 \
 --step-seconds 0.1 \
 --max-clips 25 \
 --overwrite
```

Batch review through task runner:

```bash
poetry run part-io-tasks audio-review-batch \
 --threshold 0.8 \
 --step-seconds 0.1 \
 --max-clips 25 \
 --overwrite
```

## Setup

Requirements:

- Python 3.11+
- Poetry
- Node.js (for `npx`-based CPD checks)

Install dependencies:

```bash
poetry install --with dev
```

## Overview

part-io is a Python toolkit for task orchestration and lint automation with strict architecture guardrails.

Core characteristics:

- Typed task registry and profile-driven task selection
- Lint orchestration via module entrypoints
- Architecture and boundary enforcement via Semgrep
- Focused adapters for config loading, process execution, and reporting
