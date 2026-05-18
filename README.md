# part-io

A toolkit for removing ads from podcast episodes. It detects ad-break jingles (open/close snippets) inside source audio, lets you confirm matches interactively, and cuts the ad segments out with ffmpeg.

## Requirements

- Python 3.11+
- Poetry
- ffmpeg / ffplay

```bash
poetry install --with dev
```

## Remote pipeline (recommended)

The remote pipeline works directly against a mounted remote directory (e.g. an rclone pCloud mount at `downloads/remote/`). All state — match candidates, labels, adaptive thresholds, cut status — lives in a single human-editable file: `downloads/review/state.toml`. No clip files are written to disk.

### Streamlined: one episode at a time

Detect matches, review them interactively, cut the episode, then move on:

```bash
poetry run part-io-tasks remote-loop
```

During review, for each match you hear a few seconds of the source audio at the detected position:

| Key | Action |
| --- | ------ |
| `a` | Approve (true positive — will be used for cutting) |
| `r` | Reject (false positive) |
| `p` | Replay the current match |
| `c` | Play the reference snippet for comparison |
| `s` | Skip (leave unlabeled) |
| `u` | Undo the previous decision |
| `q` | Quit and save progress |

Progress is saved after every episode. Restart at any time — already-cut episodes are skipped automatically.

### Batch: detect many, then review

Generate match candidates for a batch of episodes before reviewing:

```bash
# Detect matches for 10 episodes at a time (no review yet)
poetry run part-io-tasks remote-review --batch-size 10 --no-interactive

# Review previously detected matches
poetry run part-io-tasks remote-review

# Cut all labeled episodes
poetry run part-io-tasks remote-cut
```

### Key options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--threshold` | `0.8` | Minimum match score. Adapts upward automatically as you approve clips. |
| `--z-threshold` | `3.0` | Z-score cutoff — keeps only matches that are statistical outliers against the full episode distribution. |
| `--max-matches` | `10` | Maximum candidates to store per snippet type per episode. |
| `--min-gap` | `-15.0` | Minimum seconds between open-end and close-start (negative allows back-to-back jingles). |
| `--max-gap` | `600.0` | Maximum seconds between open-end and close-start. |
| `--yes` | flag | Skip cut confirmation prompts. |
| `--dry-run` | flag | Show planned cuts without running ffmpeg. |
| `--overwrite` | flag | Re-detect, re-review, and re-cut already-processed episodes. |

### State file

`downloads/review/state.toml` is the single source of truth. It is safe to edit by hand:

```toml
# Remote episode pipeline state.
# Edit freely — delete this file to start fresh.

[thresholds]
open  = 0.9503
close = 0.8378

[episodes.6b173fd3-0c42-45d1-806a-88ab69b861da]
source = "downloads/remote/6b173fd3-0c42-45d1-806a-88ab69b861da.mp3"
open_matches   = [{index = 1, score = 0.9704, start = 1190.900, end = 1200.900}]
close_matches  = [{index = 1, score = 0.8369, start = 4396.608, end = 4406.608}]
open_approved  = [1]
open_rejected  = []
close_approved = [1]
close_rejected = []
cut = false
```

Delete the file to treat the next run as a first-time run.

---

## Local pipeline

For working with files already on disk (e.g. `downloads/media/`).

### Detect + pair + cut a single episode

```bash
# Pair open/close matches and write ad_segments.json
poetry run part-io-tasks audio-ad-detect \
  --episode ep_ce79a6d1 --use-labels

# Dry run
poetry run part-io-tasks audio-ad-remove \
  --source downloads/media/ep_ce79a6d1.mp3 \
  --segments downloads/review/ep_ce79a6d1/ad_segments.json \
  --dry-run

# Cut
poetry run part-io-tasks audio-ad-remove \
  --source downloads/media/ep_ce79a6d1.mp3 \
  --segments downloads/review/ep_ce79a6d1/ad_segments.json
```

### Batch review bundle generation

Generates clip files and manifests under `downloads/review/` for manual inspection:

```bash
poetry run part-io-tasks audio-review-batch \
  --threshold 0.8 --z-threshold 3.0 \
  --max-clips 10 --refine --workers 2
```

---

## How it works

`downloads/snippets/open.mp3` is the jingle that plays at the start of an ad break; `close.mp3` plays at the end. The matcher builds a 32-band log-energy spectral fingerprint for both the snippet and the source, then slides the snippet fingerprint across the source and scores each window by mean cosine similarity. A z-score filter keeps only windows that score as statistical outliers against the full distribution — separating genuine matches (~99% similarity) from background noise (~95%).

Detected opens and closes are paired greedily: each open is matched with the nearest following close within a configurable time window. The paired spans are cut from the source with a single `ffmpeg` `atrim`+`concat` filter graph — no intermediate files.

---

## Development

```bash
poetry run part-io-tasks test          # run tests
poetry run part-io-tasks lint          # run declared lint tasks
poetry run part-io-tasks generate-tasks  # regenerate config/generated.mk
poetry run part-io-tasks clean         # remove caches and build artifacts
```

## License

MIT. See `LICENSE`.
