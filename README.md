# part-io

An audio jingle-matching toolkit: locate and extract recurring jingles/stingers inside long episode
recordings, with strong architecture guardrails.

## Features

- Cold-start jingle discovery via human-in-the-loop tiled scanning and bisection.
- Spectral-feature reference matching to find a known jingle across episodes.
- Architecture and boundary enforcement with Semgrep.

## Requirements

- Python 3.11+
- Poetry
- Node.js (for `npx`-based CPD duplicate-code checks in the test suite)
- ffmpeg (for audio decoding/extraction)

## Installation

```bash
poetry install --with dev
```

## Common Commands

Run tests (includes lint/type/architecture checks via `tests/integration/test_lint.py`):

```bash
poetry run pytest
```

Run architecture/security checks:

```bash
poetry run semgrep scan --config config/semgrep part_io tests --error
```

Run the audio review bundle CLI:

```bash
poetry run part-io-audio-review
```

## Project Layout

- `part_io/` application package.
- `config/` lint and Semgrep policy configuration.
- `tests/` architecture, integration, and unit tests.

## License

MIT. See `LICENSE`.
