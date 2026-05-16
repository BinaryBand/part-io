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

## Ad Removal Workflow

`downloads/snippets/open.mp3` marks the start of an ad break; `close.mp3` marks the end. The pipeline has three steps: detect candidates, review them manually, then cut.

### Step 1 — Generate review bundles

```bash
poetry run part-io-tasks audio-review-batch \
  --threshold 0.8 \
  --step-seconds 0.1 \
  --max-clips 25 \
  --overwrite --refine \
  --workers 3
```

Outputs clip files and manifests under `downloads/review/{episode}/open_high_points/` and `close_high_points/`.

### Step 2 — Label true positives

Listen to the extracted clips. For each bundle, open `match_labels.json` and fill in the `true_positive_indices` array with the clip indices that are genuine ad opens/closes.

### Step 3 — Pair opens with closes

```bash
poetry run part-io-tasks audio-ad-detect \
  --episode ep_ce79a6d1 \
  --use-labels
```

Writes `downloads/review/ep_ce79a6d1/ad_segments.json`. Without `--use-labels` all manifest rows are used (useful for a first-pass inspection, less reliable).

### Step 4 — Dry run to verify cuts

```bash
poetry run part-io-tasks audio-ad-remove \
  --source downloads/media/ep_ce79a6d1.mp3 \
  --segments downloads/review/ep_ce79a6d1/ad_segments.json \
  --dry-run
```

Prints the exact time ranges that will be removed without touching any audio.

### Step 5 — Cut

```bash
poetry run part-io-tasks audio-ad-remove \
  --source downloads/media/ep_ce79a6d1.mp3 \
  --segments downloads/review/ep_ce79a6d1/ad_segments.json
```

Writes the cleaned episode to `downloads/cleaned/ep_ce79a6d1.mp3`. The command errors on overlapping segments and refuses to cut if `ad_segments.json` is empty.

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
