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

## 4\. Similarity Score

Let $B$ be the set of audio files at the root of $\Lambda$:

$$
B = \{\, \lambda \in \Lambda \mid \texttt{is\_audio}(\lambda) \,\}
$$

For a given `AudioTarget` $t$, the **per-file similarity score** $\sigma(b, t)$ is the maximum window score found by sliding the target's reference fingerprint across the episode:

$$
\sigma(b, t) = \max_{w \,\in\, \texttt{windows}(b)} \texttt{cos\_sim}(w,\, t)
$$

The fingerprint is a 32-band log-energy spectrum with first-order temporal deltas (64-dimensional, L2-normalised). Each window is scored by mean frame-wise cosine similarity.

When confirmed positive segments $P_t$ are available, the reference fingerprint for $t$ is the **centroid** of their fingerprints (mean across confirmed examples, re-normalised). Before any positives are confirmed, a seed snippet supplied at startup is used as the initial reference.

* * *

## 5\. Threshold Derivation

Let $\text{scores}(S)$ denote the set of $\sigma$ values for all segments in $S$, and let:

$$
\text{moe}(S) = k \cdot \text{std}\bigl(\text{scores}(S)\bigr)
$$

where $k = 1.5$ by default (tunable via `--moe-factor`). $\text{moe}(S) = 0$ when $|S| < 2$.

The positive and negative thresholds are:

$$
\theta^+(t) = \min_{s \,\in\, P_t} \sigma(s, t) - \text{moe}(P_t)
$$

$$
\theta^-(t) = \max_{s \,\in\, N_t} \sigma(s, t) + \text{moe}(N_t)
$$

Before $P_t$ is non-empty, $\theta^+(t)$ defaults to the configured `--threshold` (default `0.8`). Before $N_t$ is non-empty, $\theta^-(t) = -\infty$ (no lower bound).

* * *

## 6\. Classification

The three classification regions for target $t$:

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
| $B'(t)$ | Classified positives — episode contains target |
| $\lnot B'(t)$ | Classified negatives — episode does not contain target |
| $\Diamond B'(t)$ | Uncertain — surfaced for human review |

> **Overlap resolution:** if $\theta^-(t) \geq \theta^+(t)$ (the bands cross), positive wins: $\lnot B'(t) \mathrel{{-}{=}} B'(t)$ after classification.

Classified episodes are **not re-presented** to the user on subsequent runs unless `--overwrite` is passed.

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
