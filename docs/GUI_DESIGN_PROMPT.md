# GUI Design Prompt for Claude Design

Copy everything from the horizontal rule below into Claude Design as the brief. It is written to be self-contained: a designer who has never seen this repo should be able to produce a coherent, buildable UI from it.

______________________________________________________________________

## The brief: design a desktop GUI for **partio**

### 1. What partio is

partio is an audio jingle-matching toolkit. It finds and extracts short recurring audio clips -- jingles, stingers, theme music, ad-break "bumpers" -- inside long recordings, typically podcast episodes that run 30--120 minutes.

The concrete jobs users bring to it:

1. **"Find this sound for me."** The user heard a jingle in an episode but has no clip of it. They need to isolate it once, by ear, and save it as a reusable reference ("seed").
1. **"Where else does it appear?"** With a seed clip in hand, scan the same or other episodes for every occurrence, or find the single strongest one.
1. **"Is this match real?"** Audio matching is statistical. The user must be able to listen to each candidate match and confirm or reject it before trusting it.
1. **"Cut the ad break out."** An ad break is usually bracketed by an opening jingle and a closing jingle. Remove the whole span between them -- both stingers included -- and write a new episode file.
1. **"Get me the episode first."** Episodes live in podcast RSS feeds. The user should be able to remember a feed and pull an episode without leaving the app.

Everything runs **locally and offline** on the user's own machine. Audio decode, clip extraction, playback, and transcoding are all done by `ffmpeg`/`ffplay` subprocesses. There is no server, no account, no cloud upload, and no network access except fetching RSS feeds and downloading episode audio from them.

### 2. Who uses it

- **Podcast editors / producers** trimming ad breaks out of back-catalogue episodes. They are comfortable with audio timelines but not with terminals.
- **Audio archivists / researchers** cataloguing recurring sonic elements (station IDs, sponsor stings) across a large corpus.
- **The project's own power users**, who currently live in the terminal and will judge the GUI on whether it is *faster* than the CLI, not merely prettier.

They are patient with slow operations (a full-episode spectral scan can take minutes) but will not tolerate being lied to about confidence. A false "match found" costs them trust and a manual re-listen.

### 3. Design principles (please honour these)

1. **Ears first.** The primary verification loop is *listen $\rightarrow$ judge*. Every place a match is shown, it must be one gesture away from being heard in context (a few seconds before and after), not just as a row in a table.
1. **Never hide the uncertainty.** The detector produces a correlation-like score (roughly 0.0 = unrelated, approaching 1.0 = a true occurrence) plus, for single-best results, a **prominence z-score** measuring how much the peak stands out from that recording's own baseline. Show both, always, with plain language. Scores are *source-dependent*: the same seed scored against two episodes is calibrated to each episode separately -- so never imply a score is an absolute global truth.
1. **Reversible and inspectable.** The user must always be able to see *why* a match was proposed (its time span, duration, score, prominence) and undo or exclude it before anything is written to disk.
1. **Show progress honestly.** Long scans must show real, cancellable progress with a time estimate, not an indefinite spinner.
1. **Local-first is a feature, not an accident.** Communicate that audio never leaves the machine. No cloud sync, no account, no telemetry in the UI.
1. **Do not invent capability.** The list in section 5 is the *entire* feature set. Do not add collaboration, sharing, batch cloud processing, or AI transcription features that do not exist.

### 4. Domain vocabulary (use these words consistently in the UI)

| Term | Meaning |
| --- | --- |
| **Episode** | A long source recording being searched (MP3 or similar). |
| **Seed** | A short extracted reference clip, produced by bootstrap, reused for searching. |
| **Sample / reference** | A seed clip used as the pattern to find. |
| **Region** | A time window inside an episode the user restricts the search to. |
| **Tile** | A fixed-width slice (default 10 s) played during discovery. |
| **Probe** | A short clip (default 1.5 s) used to bisect a boundary. |
| **Onset / offset** | The exact start and end times of a jingle. |
| **Discovery / bootstrap** | Finding a jingle with no reference yet, by yes/no listening. |
| **Match** | A time span in an episode the detector believes contains the sample. |
| **Score** | Match similarity, ~0.0--1.0. |
| **Prominence** | Z-score of how far a peak stands above the recording's own baseline. |
| **Threshold** | Minimum score to accept a match (default 0.8). |
| **Break** | A span bracketed by an opening jingle and a closing jingle. |
| **Bundle** | A review folder of extracted clips + a manifest + a labels file. |
| **Feed / track** | A remembered podcast RSS feed, and one selectable episode in it. |

### 5. Complete feature set to design for

This is the whole product. Each line is an existing, working capability.

#### A. Library (episodes & samples)

- Remember a podcast feed by URL (title is auto-detected; the user may override a friendly name). Nothing is downloaded at this point.
- List remembered feeds and the episodes each offers, newest first.
- Remove a remembered feed. Already-downloaded audio stays on disk.
- A unified picker of everything selectable:
  - Episodes from every remembered feed, grouped under the feed's name.
  - A group called **"on disk"** for local files and for bootstrapped seeds.
  - A per-row marker: `*` = already on disk, `o` = will download when picked.
  - A **"load every episode"** row: feeds are read partially by default (roughly the newest 40 episodes, ~256 KB) because full back-catalogues can be tens of MB and take many seconds to parse. This row deliberately reads the whole feed.
  - Picking an `o` episode downloads it into the app's downloads folder and then hands the local path to whatever action needed it. Picking it again is free.

#### B. Bootstrap -- discover a jingle with no reference

- User supplies: an episode, a search region (start/end seconds; default is the whole file), a maximum number of occurrences (walks through each one and saves numbered seeds), and advanced tuning: tile width (10 s), probe length (1.5 s), bisection resolution (0.5 s).
- The app plays candidate tiles aloud and asks yes/no questions until the onset and offset are pinned down, then extracts and saves a seed clip (default `static/jingles/<episode>_seed.mp3`, or `_seed_01.mp3`, `_02.mp3`, ... when multiple occurrences are requested).
- Seeds are automatically registered in the sample library so they appear in later "choose a sample" pickers.
- **This is the signature interaction of the product. Please treat the visual design of it as the centrepiece (spec in section 6).**
- If no jingle is found in the region, say so clearly -- it is an expected outcome, not an error.

#### C. Search -- list every occurrence of a seed

- Inputs: an episode and a reference sample, plus a score threshold (default 0.8).
- Output: every match above the threshold, as start time, end time, and score.
- Expected outcome "no matches found" must read as a normal empty state.

#### D. Locate -- find the single best occurrence

- Inputs: episode, reference sample, a sliding-window step (default 0.1 s), an optional "search only the first N seconds" limit (useful for intros/outros), and a minimum prominence z-score below which the result is rejected.
- Output: one best match with start, end, score, and prominence. A weak peak is reported as "no confident match found".

#### E. Review -- extract clips and label them by ear

- Inputs: episode, reference sample, threshold, step, an overlap-dedupe ratio (default 0.5), a cap on the number of top-scored matches to export (default 25; 0 = all), an output root folder (default `downloads/review`), an optional bundle name, an overwrite toggle, and an interactive toggle.
- Output: a bundle directory containing
  - one MP3 clip per selected match, named with its index, score, and start time;
  - `matches_manifest.csv` with columns index, score, start_seconds, end_seconds, duration_seconds, clip_path;
  - `match_labels.json` with the source path, sample path, the threshold, and `true_positive_indices` / `false_positive_indices` arrays plus a notes field.
- Non-interactive mode writes an empty labels template for the user to fill in after listening offline.
- Interactive mode plays each candidate clip and collects a confirm/reject verdict live, and writes the verdicts into the labels file.
- The summary reports how many clips were exported out of how many total matches.
- Must refuse to write into an existing bundle unless overwriting is enabled.

#### F. Cut -- remove ad breaks bounded by two jingles

- Inputs: episode, an opening jingle, a closing jingle, an optional output path (default `<episode>_cut.mp3` beside the source), an overwrite toggle, a step, and two modes:
  - **single** (default): locate the best occurrence of each jingle using the prominence method, then remove everything from the start of the opening jingle to the end of the closing one -- both stingers included.
  - **all**: find every occurrence of each jingle above a score threshold, pair them up, and remove every bracketed span (for episodes with several mid-rolls sharing the same stingers).
- Output: a new MP3, plus a summary listing each removed span and the total time kept. If a jingle cannot be confidently found, report that instead of cutting.
- This is a **destructive-feeling** operation even though it writes a new file -- the design must make the exact spans being removed reviewable *before* commit.

#### G. Comfort features

- Every operation that reads a path should offer the unified picker rather than a raw file dialog where possible (with a normal file picker as an escape hatch).
- Results should be exportable/inspectable as machine-readable JSON as well as human-readable text (the CLI has a global `--json` flag; the GUI can surface this as "copy as JSON" / "export").
- A clipboard-friendly way to copy a timestamp or a range (e.g. `32.44s -> 41.90s`) for use in another editor.
