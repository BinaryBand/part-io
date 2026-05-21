"""Benchmark harness to collect timing traces using the project's timing utilities.

Usage:
  python scripts/benchmark_timing.py --synthetic
  python scripts/benchmark_timing.py --summarize ./downloads/timings.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, median

import numpy as np


def synthetic_run(out_path: str, runs: int = 3) -> None:
    os.environ.setdefault("PART_IO_TIMING", "1")
    os.environ.setdefault("PART_IO_TIMING_OUT", out_path)

    # Import after env vars so timing picks up the enabled flag
    from part_io.adapters.audio import matcher

    rng = np.random.default_rng(123)

    cases = [
        (5000, 200, 64),
        (56250, 156, 64),
    ]

    for n, m, d in cases:
        print(f"Running synthetic case n={n:,}, m={m}, d={d}")
        src = rng.random((n, d), dtype=np.float32)
        ref = rng.random((m, d), dtype=np.float32)

        # warm-up
        _ = matcher._windowed_search(ref, src, 1)
        _ = matcher._cross_correlation_search(ref, src, 1)

        for _ in range(runs):
            _ = matcher._windowed_search(ref, src, 1)
            _ = matcher._cross_correlation_search(ref, src, 1)

    print(f"Wrote timings to {out_path}")


def summarize(path: str) -> None:
    if not os.path.exists(path):
        print("No timing file found:", path)
        return
    groups: dict[str, list[float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:  # noqa: S112
                continue
            groups.setdefault(obj.get("label"), []).append(float(obj.get("elapsed_seconds", 0.0)))

    rows = []
    for label, vals in groups.items():
        vals = sorted(vals)
        rows.append(
            (
                label,
                len(vals),
                mean(vals),
                median(vals),
                vals[int(len(vals) * 0.95) - 1] if len(vals) > 1 else vals[0],
            )
        )

    print("Summary (label, count, mean, median, p95)")
    for r in sorted(rows, key=lambda x: x[2], reverse=True):
        print(
            f"{r[0]:40s} {r[1]:6d} {r[2] * 1000:8.2f}ms {r[3] * 1000:8.2f}ms {r[4] * 1000:8.2f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", default="./downloads/timings.jsonl")
    parser.add_argument("--summarize", nargs="?", const="./downloads/timings.jsonl")
    args = parser.parse_args()

    if args.synthetic:
        synthetic_run(args.out)
        return

    if args.summarize:
        summarize(args.summarize)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
