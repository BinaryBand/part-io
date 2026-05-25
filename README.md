# PartIO

PartIO is a command-line toolkit for detecting and safely removing short ad jingles and other markers in podcast audio. It combines spectral-template detection, a small conservative classifier, and an interactive review workflow so you can verify candidates before performing destructive cuts.

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

Stage 1: pre-cache profiles (background by default):

```bash
poetry run part-io-tasks remote-precache <REMOTE_DIR>
```

Run in the foreground when desired:

```bash
poetry run part-io-tasks remote-precache <REMOTE_DIR> --no-background
```

Stage 2: prepare quiz candidates (`precache` side effect included):

```bash
poetry run part-io-tasks remote-prep-quiz <REMOTE_DIR> \
  --intro-seed downloads/snippets/intro.mp3 \
  --open-seed downloads/snippets/open.mp3 \
  --close-seed downloads/snippets/close.mp3
```

Stage 3: run interactive quiz and save labels:

```bash
poetry run part-io-tasks remote-prep-cut <REMOTE_DIR>
```

Stage 4: execute cut pass (background by default):

```bash
poetry run part-io-tasks remote-execute-cut <REMOTE_DIR> --output-dir staging --yes
```

Preview and promote staged replacements:

```bash
poetry run part-io-tasks remote-promote staging <REMOTE_DIR> --dry-run
poetry run part-io-tasks remote-promote staging <REMOTE_DIR>
```

Disable background mode for prep/cut stages with `--no-background`.

Notes:

- A `__state__.toml` file is created under the target directory to persist detection and quiz state across runs. Deleting it resets state.
- Never set `--output-dir` to the same directory as the input; ffmpeg cannot atomically read-and-write the same file.

Configuration

You can control per-extension encoding defaults used when writing cut output by adding
`[tool.part_io.defaults.codecs]` to your `pyproject.toml`. Keys are file extensions
without the leading dot; each entry should provide a `codec` and optionally a `bitrate`.
For example:

```toml
[tool.part_io.defaults.codecs]
mp3 = { codec = "libmp3lame", bitrate = "128k" }
opus = { codec = "libopus", bitrate = "64k" }
aac = { codec = "aac", bitrate = "128k" }
wav = { codec = "pcm_s16le" }
```

When present, these settings determine the ffmpeg codec and bitrate used for the final
encoded output. The pipeline also preserves the source file extension for cut outputs
(`stem + source.suffix`), and debug clips use the same extension. If no configuration
is provided for an extension, part-io falls back to sensible built-in defaults.

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
poetry run python -m part_io.cli.remote_pipeline prep-quiz <REMOTE_DIR> \
 --open-seed downloads/snippets/open.mp3 \
 --close-seed downloads/snippets/close.mp3
```

## Contributing

Contributions are welcome. Open an issue or PR with a clear description and tests for behaviour changes.

## License

See the `LICENSE` file for license terms.

> kill PID
