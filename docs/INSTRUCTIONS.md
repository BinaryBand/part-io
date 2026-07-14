# Getting Started with part-io

part-io finds and extracts recurring jingles/stingers inside long episode recordings. This guide walks through installation and the unified CLI.

## 1. Install

Requirements: Python 3.11+, Poetry, ffmpeg on `PATH`, and Node.js (only needed for the `npx`-based duplicate-code check in the test suite).

```bash
poetry install --with dev
```

Verify everything is wired up:

```bash
poetry run pytest
```

Note: running the test suite mutates your working tree -- `conftest.py` runs `ruff format` and `ruff check --fix` across the repo at the start of every pytest session.

## 2. The workflow, in order

You always start with a long episode recording and no reference clip. The commands chain together like this:

1. **`part-io audio bootstrap`** -- cold start. Interactively narrows down a region of the episode to a clean jingle clip ("seed") that you can reuse.
1. **`part-io audio search`** / **`part-io audio locate`** -- once you have a seed clip, find every place (or the single best place) it recurs across the same or other episodes.
1. **`part-io audio review`** -- generate an extracted-clip bundle plus a manifest so you can sanity-check matches by ear before trusting them.

### Step 1: Bootstrap a seed clip

No reference sample needed yet. Point it at the region of the episode where you expect the jingle to appear; it plays candidate tiles through `ffplay` and asks yes/no questions until it has pinned down the exact onset and offset.

```bash
poetry run part-io audio bootstrap --source episode.mp3 \
  --region-start 0 --region-end 120
```

This writes `static/jingles/episode_seed.mp3` by default (override with `--output`).

Useful flags:

- `--region-start` / `--region-end` -- seconds to search within (default `0` to `120`). Point this at wherever the jingle you care about actually plays.
- `--max-occurrences` -- if the *same* jingle plays more than once in the region, set this above `1` to walk through and seed each occurrence as `episode_seed_01.mp3`, `episode_seed_02.mp3`, etc.
- `--tile-seconds`, `--probe-seconds`, `--resolution` -- tuning knobs for the discovery/bisection granularity; the defaults are fine to start.

### Step 2: Find where a seed clip recurs

Once you have a seed clip, search for it across the full episode (or a different episode entirely):

```bash
poetry run part-io audio search --source episode.mp3 \
  --sample static/jingles/episode_seed.mp3 --threshold 0.8
```

This prints every match above the score threshold with its timestamp. If you only care about the single strongest match, use `audio locate` instead -- it scores matches by a prominence z-score rather than a fixed threshold:

```bash
poetry run part-io audio locate --source episode.mp3 \
  --sample static/jingles/episode_seed.mp3 --search-seconds 120 --min-prominence 2.0
```

### Step 3: Review matches before trusting them

`part-io audio search` gives you timestamps, but you should listen before relying on them. `part-io audio review` extracts an MP3 clip per match plus a manifest and a labels file:

```bash
poetry run part-io audio review --source episode.mp3 \
  --sample static/jingles/episode_seed.mp3 \
  --threshold 0.8 --max-clips 25 --interactive
```

- Without `--interactive`, it writes an empty `match_labels.json` template for you to fill in after listening to the clips manually.
- With `--interactive`, it plays each candidate clip through `ffplay` and asks you to confirm it live.
- Output lands under `downloads/review/<episode>/<seed>/` by default (override with `--output-root` / `--bundle-name`).

## 3. Quick reference

| Command | Purpose |
|---|---|
| `part-io audio bootstrap` | Find a jingle with no reference clip yet |
| `part-io audio search` | List every match of a seed clip above a threshold |
| `part-io audio locate` | Find the single best/strongest match |
| `part-io audio review` | Extract clips + manifest for manual verification |

All commands are accessible through the unified `part-io` entry point. Running `part-io` with no arguments opens an interactive picker.

Global options:

- `--json` -- output results as JSON instead of human-readable text.
- `--version` / `-v` -- show the installed version.

## 4. Where things are enforced

If you plan to contribute code rather than just run the tools, see the root [CLAUDE.md](../CLAUDE.md) for the layered architecture and how lint/type/architecture checks enforce it.
