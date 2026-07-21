# Getting Started with partio

partio finds and extracts recurring jingles/stingers inside long episode recordings. This guide walks through installation and the unified CLI.

## 1. Install

Requirements: Python 3.11+, uv, and ffmpeg on `PATH`.

```bash
uv sync --all-groups
```

Verify everything is wired up:

```bash
uv run pytest
```

Note: running the test suite mutates your working tree -- `conftest.py` runs `ruff format` and `ruff check --fix` across the repo at the start of every pytest session.

## 2. The workflow, in order

You always start with a long episode recording and no reference clip. The commands chain together like this:

1. **`partio audio bootstrap`** -- cold start. Interactively narrows down a region of the episode to a clean jingle clip ("seed") that you can reuse.
1. **`partio audio search`** / **`partio audio locate`** -- once you have a seed clip, find every place (or the single best place) it recurs across the same or other episodes.
1. **`partio audio review`** -- generate an extracted-clip bundle plus a manifest so you can sanity-check matches by ear before trusting them.

### Step 0 (optional): Remember a podcast feed

If you don't already have an episode on disk, remember a feed and every one of its episodes becomes selectable:

```bash
uv run partio feed add --url https://example.com/podcast/rss
```

Nothing is downloaded here. From now on, any command that asks for a `--source` opens a picker listing that feed's episodes -- `●` for ones already on disk, `○` for ones that aren't. Picking a `○` downloads it right there into `static/downloads/` and hands the local path to the command. Picking it again later costs nothing.

The picker reads only each feed's newest episodes (roughly the last 40), because a long-running podcast feed is tens of megabytes and takes seconds to parse in full. If you need something older, the `load every episode` row at the bottom reads the whole back catalogue, and `partio feed list` always does.

That is the whole of feed management:

- `partio feed add` -- remember a feed.
- `partio feed list` -- see the feeds and their episodes, newest first.
- `partio feed remove` -- forget a feed (pick it by name; downloads stay on disk).

There is no separate library to curate. Bootstrapped seed clips join the same picker automatically, under `on disk`.

### Step 1: Bootstrap a seed clip

No reference sample needed yet. Point it at the region of the episode where you expect the jingle to appear; it plays candidate tiles through `ffplay` and asks yes/no questions until it has pinned down the exact onset and offset.

```bash
uv run partio audio bootstrap --source episode.mp3 \
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
uv run partio audio search --source episode.mp3 \
  --sample static/jingles/episode_seed.mp3 --threshold 0.8
```

This prints every match above the score threshold with its timestamp. If you only care about the single strongest match, use `audio locate` instead -- it scores matches by a prominence z-score rather than a fixed threshold:

```bash
uv run partio audio locate --source episode.mp3 \
  --sample static/jingles/episode_seed.mp3 --search-seconds 120 --min-prominence 2.0
```

### Step 3: Review matches before trusting them

`partio audio search` gives you timestamps, but you should listen before relying on them. `partio audio review` extracts an MP3 clip per match plus a manifest and a labels file:

```bash
uv run partio audio review --source episode.mp3 \
  --sample static/jingles/episode_seed.mp3 \
  --threshold 0.8 --max-clips 25 --interactive
```

- Without `--interactive`, it writes an empty `match_labels.json` template for you to fill in after listening to the clips manually.
- With `--interactive`, it plays each candidate clip through `ffplay` and asks you to confirm it live.
- Output lands under `downloads/review/<episode>/<seed>/` by default (override with `--output-root` / `--bundle-name`).

## 3. Quick reference

| Command | Purpose |
| --- | --- |
| `partio audio bootstrap` | Find a jingle with no reference clip yet |
| `partio audio search` | List every match of a seed clip above a threshold |
| `partio audio locate` | Find the single best/strongest match |
| `partio audio review` | Extract clips + manifest for manual verification |
| `partio feed add` | Remember a feed so its episodes become selectable |
| `partio feed list` | See every feed and the episodes it offers |
| `partio feed remove` | Forget a feed |

All commands are accessible through the unified `partio` entry point. Running `partio` with no arguments opens an interactive picker.

Global options:

- `--json` -- output results as JSON instead of human-readable text.
- `--version` / `-v` -- show the installed version.

## 4. Where things are enforced

If you plan to contribute code rather than just run the tools, see the root [CLAUDE.md](../CLAUDE.md) for the layered architecture and how lint/type/architecture checks enforce it.
