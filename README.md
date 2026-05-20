# PartIO

PartIO is a command-line toolkit for detecting and safely removing short ad jingles
and other markers in podcast audio. It combines spectral-template detection, a
small conservative classifier, and an interactive review workflow so you can
verify candidates before performing destructive cuts.

## Features

- Detect four target kinds: `open`, `close`, `intro`, and `outro`.
- Build consensus spectral profiles from confirmed positives to improve detection.
- Interactive review queue with undo and conservative margin-of-error (MoE).
- Safe ffmpeg-based cutting with staging and atomic promotion to avoid data loss.
- Background profile pre-cache and verbose decode logging for large collections.

## Quickstart

Prerequisites: Python (Poetry-managed project) and `ffmpeg` on your PATH.

Install dependencies:

```bash
poetry install
```

Detect and review (interactive):

```bash
poetry run part-io-tasks remote-loop <REMOTE_DIR> \
	--snippets-dir snippets --open-sample open.mp3 --close-sample close.mp3
```

Detect-only (non-interactive):

```bash
poetry run part-io-tasks remote-loop <REMOTE_DIR> --no-interactive
```

Run a cut pass (write to a staging directory):

```bash
poetry run part-io-tasks remote-loop <REMOTE_DIR> --output-dir staging --yes
```

Preview and promote staged replacements:

```bash
poetry run part-io-tasks remote-promote staging <REMOTE_DIR> --dry-run
poetry run part-io-tasks remote-promote staging <REMOTE_DIR>
```

Start a background pre-cache worker (builds spectral profiles):

```bash
poetry run part-io-tasks remote-precache-start <REMOTE_DIR> --sleep 10
poetry run part-io-tasks remote-precache-status <REMOTE_DIR>
poetry run part-io-tasks remote-precache-stop <REMOTE_DIR>
```

Notes:

- A `__state__.toml` file is created under the target directory to persist
	detection and review state across runs. Deleting it resets state.
- Never set `--output-dir` to the same directory as the input; ffmpeg cannot
	atomically read-and-write the same file.

## Development

Run the full test suite:

```bash
poetry run pytest -q
```

Lint and checks:

```bash
poetry run ruff format .
poetry run ruff check .
poetry run python -m part_io.cli.lint.semgrep
```

Run a single command directly via the module (alternate to `part-io-tasks`):

```bash
poetry run python -m part_io.cli.remote_pipeline review <REMOTE_DIR> \
	--snippets-dir snippets --open-sample open.mp3 --close-sample close.mp3
```

## Contributing

Contributions are welcome. Open an issue or PR with a clear description and
tests for behaviour changes.

## License

See the `LICENSE` file for license terms.

