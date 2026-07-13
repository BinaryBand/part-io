# Getting Started with part-io

part-io finds and extracts recurring jingles/stingers inside long episode recordings. This guide walks through installation and the three CLI tools you'll use day to day.

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

You always start with a long episode recording and no reference clip. The tools chain together like this:

1. **`audio_bootstrap`** -- cold start. Interactively narrows down a region of the episode to a clean jingle clip ("seed") that you can reuse.
1. **`audio_search`** / **`audio_locate`** -- once you have a seed clip, find every place (or the single best place) it recurs across the same or other episodes.
1. **`audio_review`** -- generate an extracted-clip bundle plus a manifest so you can sanity-check matches by ear before trusting them.

### Step 1: Bootstrap a seed clip

No reference sample needed yet. Point it at the region of the episode where you expect the jingle to appear; it plays candidate tiles through `ffplay` and asks yes/no questions until it has pinned down the exact onset and offset.

```bash
poetry run part-io-audio-bootstrap episode.mp3 \
  --region-start 0 --region-end 120
```

This writes `static/jingles/episode_seed.mp3` by default (override with `--output`).

Useful flags:

- `--region-start` / `--region-end` -- seconds to search within (default `0` to `120`). Point this at wherever the jingle you care about actually plays -- e.g. the first couple of minutes for an episode opener, or a later window if you're after something that only occurs further into the episode.
- `--max-occurrences` -- if the *same* jingle plays more than once in the region (e.g. it stings both going into and coming out of a segment), set this above `1` to walk through and seed each occurrence as `episode_seed_01.mp3`, `episode_seed_02.mp3`, etc. under `--output` (now treated as a directory).
- `--tile-seconds`, `--probe-seconds`, `--resolution` -- tuning knobs for the discovery/bisection granularity; the defaults are fine to start.

There's no built-in notion of jingle "types" (opener vs. mid-break, etc.) -- if you're after two audibly different jingles, just run `audio_bootstrap` twice against different regions with different `--output` paths, and treat the results as independent seed clips from here on.

### Step 2: Find where a seed clip recurs

Once you have a seed clip, search for it across the full episode (or a different episode entirely):

```bash
poetry run python -m part_io.cli.audio_search episode.mp3 \
  static/jingles/episode_seed.mp3 --threshold 0.8
```

This prints every match above the score threshold with its timestamp. If you only care about the single strongest match (e.g. confirming there's exactly one clean intro), use `audio_locate` instead -- it scores matches by a prominence z-score rather than a fixed threshold, which holds up better in speech-heavy audio:

```bash
poetry run python -m part_io.cli.audio_locate episode.mp3 \
  static/jingles/episode_seed.mp3 --search-seconds 120 --min-prominence 2.0
```

### Step 3: Review matches before trusting them

`audio_search` gives you timestamps, but you should listen before relying on them. `part-io-audio-review` extracts an MP3 clip per match plus a manifest and a labels file you fill in by hand (or interactively):

```bash
poetry run part-io-audio-review episode.mp3 \
  static/jingles/episode_seed.mp3 \
  --threshold 0.8 --max-clips 25 --interactive
```

- Without `--interactive`, it writes an empty `match_labels.json` template for you to fill in (`true_positive_indices` / `false_positive_indices`) after listening to the clips manually.
- With `--interactive`, it plays each candidate clip through `ffplay` and asks you to confirm it live, writing the labels automatically.
- Output lands under `downloads/review/<episode>/<seed>/` by default (override with `--output-root` / `--bundle-name`).

## 3. Quick reference

| Tool | Installed command | Purpose | | -------- | ------------------------------------- | ---------------------------------------------------- | | Bootstrap | `part-io-audio-bootstrap` | Find a jingle with no reference clip yet | | Search | `python -m part_io.cli.audio_search` | List every match of a seed clip above a threshold | | Locate | `python -m part_io.cli.audio_locate` | Find the single best/strongest match | | Review | `part-io-audio-review` | Extract clips + manifest for manual verification |

Only `part-io-audio-bootstrap` and `part-io-audio-review` are registered as installed commands; `audio_search` and `audio_locate` are run as modules with `python -m`.

## 4. Where things are enforced

If you plan to contribute code rather than just run the tools, see the root [CLAUDE.md](../CLAUDE.md) for the layered architecture (`models` -> `services` -> `adapters` -> `cli`) and how Semgrep/AST checks enforce it.
