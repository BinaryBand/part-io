# PartIO - Formal Specifications

## 1\. Target Types

Four target types are defined for podcast audio:

| Type | Description |
| --- | --- |
| `open` | Opening jingle of an ad break |
| `close` | Closing jingle of an ad break |
| `intro` | Opening of the podcast episode itself |
| `outro` | Closing of the podcast episode itself |

* * *

## 2\. Setup

Point the tool at a directory $\Lambda$ whose root contains episode audio files:

```bash
partio remote-loop <path-to-Λ>
```

A `__state__.toml` file is written under $\Lambda$ to persist all classification state across runs. Deleting it starts fresh.

* * *

## 3\. Types

```python
class State:
    targets: list[AudioTarget]
    output:  Path
    max_k: int = 5   # max condidates per AudioTarget
    max_gap: float = 60 * 5  # max seconds between open and close
    moe_factor: float = 1.5  # 
    threshold: float = 0.8  #
    is_inclusive: bool = False # cut transitions too?
    fade_dur: float = 0
    
class AudioTarget:
    type: Literal['intro', 'outro', 'open', 'close']
    positives: list[Segment]  # confirmed examples
    negatives: list[Segment]  # confirmed non-examples

class Segment:
    source: str # path relative to Λ
    start: float # seconds
    end: float  # seconds
```

> `Segment.source` is always relative to $\Lambda$ so the state file is portable across machines.

* * *

## 4\. Similarity Score and Windowed Search

Let $B$ be the set of audio files at the root of $\Lambda$.

The implemented matcher builds a per-frame fingerprint composed of a 32-band log-energy spectrum concatenated with first-order temporal deltas (64 dimensions per frame). Analysis is performed at 16 kHz with a frame size of 2048 samples and a hop of 1024 samples (frame hop $=1024/16000\approx0.064\,$s).

For a target $t$ with a reference fingerprint $R\in\mathbb{R}^{m\times d}$ and a source profile $S\in\mathbb{R}^{N\times d}$ the matcher scores every length-$m$ window of $S$ by the mean frame-wise dot product (equivalent to mean frame-wise cosine similarity when frames are L2-normalised):

$$
\sigma(b,t) = \max_{0\le i\le N-m} \frac{1}{m}\sum_{j=0}^{m-1} \langle R_j,\; S_{i+j} \rangle
$$

Implementation notes:

- The canonical detector uses an FFT-based cross-correlation over the feature axes to compute the full lag-series in $O(N\log N)$ and then samples it at a stride derived from `step_seconds` (see below).
- A direct sliding-window dot-product implementation (`_windowed_search`) also exists but is not the fast path.
- The `step_seconds` CLI/setting maps to a frame hop as `hop = max(1, int(step_seconds / frame_hop_seconds))` with `frame_hop_seconds = 1024/16000 \approx 0.064s`. The matcher therefore evaluates scores at every `hop` frames.
- Per-file pre-filtering: when `z_threshold` is supplied, the raw per-window scores for a file are used to compute `mean` and `std`; an `effective_threshold` is set to

$$
	ext{effective\_threshold} = \max\bigl(\text{score\_threshold},\; \mu_{scores} + z\_threshold \cdot \sigma_{scores}\bigr)
$$

Only peaks above `effective_threshold` are kept. This removes peaks that are not significantly above the local noise floor for that file.
- Overlapping candidate suppression keeps the highest-scoring candidate among matches whose overlap ratio exceeds the configured `dedupe_overlap` (default 0.5).
- Matches are optionally refined by `anchor_to_onset` (shift to the first clear energy onset inside the matched window) and `cross_correlate_align` (waveform cross-correlation on a padded window for sample-accurate alignment).
- Source spectral profiles can be cached to disk (`.npz`) and are validated by file mtime and size to avoid stale reuse.
When confirmed positive segments $P_t$ are available, the reference fingerprint for $t$ is the **centroid** of their fingerprints (mean across confirmed examples, re-normalised). Before any positives are confirmed, a seed snippet supplied at startup is used as the initial reference.

* * *

## 5\. Thresholds and Margin of Error (MoE)

There are two orthogonal thresholding stages in the implemented pipeline:

1. Per-file window filtering during candidate extraction (see Section 4). This is controlled by the optional `z_threshold` setting: when present, windows must exceed `mean + z_threshold * std` (and also the configured `score_threshold` floor) to be considered candidates for that file. This removes locally insignificant peaks on noisy files.

2. Global classification thresholds used to automatically classify episodes as `positive`, `negative`, or `uncertain` after human approvals accumulate. These are computed from confirmed positive and negative example scores using a statistically principled Margin-of-Error on the sample mean (98% two-tailed confidence).

The implemented MoE is the t-distribution based margin on the sample mean:

$$
	ext{moe}(S) = t_{\alpha/2,\,n-1} \cdot \frac{\sqrt{\operatorname{Var}(S)}}{\sqrt{n}}
$$

where $t_{\alpha/2,\,n-1}$ is a 98% two-tailed critical value chosen by sample size ($n$). For $n\ge 31$ the normal approximation is used ($\approx 2.326$). For $n<2$ the implementation treats `moe = +\infty` so that a single confirmed example never triggers auto-classification (the uncertain zone collapses only as evidence accumulates).

Global classification thresholds are:

$$
	heta^+(t) = \begin{cases}
\min P_t + \text{moe}(P_t) & P_t \neq \emptyset \\
 +\infty & P_t = \emptyset
\end{cases}
$$

$$
	heta^-(t) = \begin{cases}
\max N_t - \text{moe}(N_t) & N_t \neq \emptyset \\
 -\infty & N_t = \emptyset
\end{cases}
$$

An episode whose top candidate score is $\ge\theta^+$ is auto-labelled `positive`; one with top score $\le\theta^-$ is auto-labelled `negative`; otherwise it remains `uncertain` and is presented in the human review queue. After each human decision the thresholds are recomputed and the uncertain set is re-evaluated.

Compatibility note: a legacy, variance-based *k·std* mode is supported as a compatibility option in internal helpers, but the default and canonical method is the t-distribution MoE described above.

* * *

## 6\. Classification

Using the global thresholds from Section 5, episodes are partitioned into three regions:

$$
B'(t) = \{\, b \in B \mid \sigma(b, t) \geq \theta^+(t) \,\}
$$

$$
\lnot B'(t) = \{\, b \in B \mid \sigma(b, t) \leq \theta^-(t) \,\}
$$

$$
\Diamond B'(t) = B \setminus \bigl(B'(t) \cup \lnot B'(t)\bigr)
$$

| Set | Meaning |
| --- | --- |
| $B'(t)$ | Auto-classified positives — episode contains the target |
| $\lnot B'(t)$ | Auto-classified negatives — episode does not contain the target |
| $\Diamond B'(t)$ | Uncertain — surfaced for human review |

If the computed bands cross ($\theta^- \ge \theta^+$) the implementation resolves the overlap conservatively in favour of positive classification. Classified episodes are not re-presented during review unless the user explicitly requests overwrite or performs manual changes.

* * *

## 7\. Human-in-the-Loop Resolution

Files in $\Diamond B'(t)$ are presented to the user in descending score order. For each file the tool plays the best-matching window (via `ffplay -ss -t`, no disk write) and prompts:

| Key | Action |
| --- | --- |
| `a` | Approve — adds segment to $P_t$; folds file into $B'(t)$ |
| `r` | Reject — adds segment to $N_t$; folds file into $\lnot B'(t)$ |
| `p` | Replay current segment |
| `c` | Play the reference snippet for comparison |
| `s` | Skip — leaves file in $\Diamond B'(t)$ for the next run |
| `u` | Undo the previous decision |
| `q` | Quit and save progress |

After each decision the thresholds $\theta^\pm(t)$ are recomputed and $\Diamond B'(t)$ is re-evaluated. Files that fall outside the uncertain region after recomputation are removed from the review queue automatically.

The loop repeats until $\Diamond B'(t) = \emptyset$ or the user exits.

* * *

## 8\. Output

For episodes in $B'(\text{open}) \cap B'(\text{close})$, **all** valid open→close pairs are identified greedily and the cleaned episode is written to `output/`.

Episodes in $B'(\text{open})$ but not $B'(\text{close})$, or vice versa, are logged as unpaired and skipped.

### 8.1 Pairing

Let $O_b$ and $C_b$ denote the candidate positions (top-$k$ matches) for the open and close targets in episode $b$. The greedy pairing algorithm processes opens in ascending time order: for each open candidate $o \in O_b$, it selects the earliest unused close candidate $c \in C_b$ satisfying:

$$
\text{min\_gap} \leq c.\text{start} - o.\text{end} \leq \text{max\_gap}
$$

This yields an ordered sequence of pairs $\{(o_i, c_i)\}_{i=1}^{n}$ representing all $n$ ad breaks found in the episode.

### 8.2 Cut Spans

Depending on `is_inclusive`, the cut region for each ad break $(o_i, c_i)$ is:

$$
\text{cut}_i = \begin{cases}
[o_i.\text{end},\; c_i.\text{start}] & \text{if } \lnot\,\text{is\_inclusive} \quad \text{(keep jingles)} \\
[o_i.\text{start},\; c_i.\text{end}] & \text{if } \text{is\_inclusive} \quad \text{(remove jingles)}
\end{cases}
$$

The keep-spans are the complement of all cut regions over the episode duration $\tau_b$:

$$
\text{keep}(b) = [0, \tau_b] \setminus \bigcup_{i=1}^{n} \text{cut}_i
$$

### 8.3 Fade

When `fade_dur` $= \delta > 0$, each keep-span except the first gains a fade-in of duration $\delta$ at its start, and each keep-span except the last gains a fade-out of duration $\delta$ at its end, smoothing each seam.

### 8.4 Reconstruction

All keep-spans are concatenated via a single `ffmpeg filter_complex` call (no temporary files):

$$
\texttt{output}(b) = \bigoplus_{[s_i,\, e_i] \,\in\, \text{keep}(b)} \texttt{atrim}(b,\, s_i,\, e_i)
$$

where $\oplus$ denotes time-domain concatenation.
