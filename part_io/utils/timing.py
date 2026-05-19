"""Lightweight, opt-in timing helpers for performance investigation.

Usage:
 - Enable with `PART_IO_TIMING=1` and set `PART_IO_TIMING_OUT` for output path.
 - Use `with Timer('label'):` or decorate functions with `@timed('label')`.

Logs are appended as JSON lines to the output path and are safe for quick
post-processing with jq/pandas.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable

_ENABLED = bool(os.environ.get("PART_IO_TIMING"))
_OUT_PATH = os.environ.get("PART_IO_TIMING_OUT", "./downloads/timings.jsonl")
_LOCK = threading.Lock()


def _write_record(record: dict[str, Any]) -> None:
    if not _ENABLED:
        return
    # ensure directory
    try:
        dirname = os.path.dirname(_OUT_PATH)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        with _LOCK:
            with open(_OUT_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Timing must never crash the host process.
        return


class Timer(ContextDecorator):
    """Context manager that records elapsed wall-clock time.

    Example:
        with Timer("matcher.xcorr"):
            do_work()
    """

    def __init__(self, label: str, extra: dict[str, Any] | None = None) -> None:
        """Create a timer for the given label.

        The timer does nothing unless `PART_IO_TIMING` is set in the
        environment.
        """
        self.label = label
        self.extra = dict(extra or {})
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        """Start timing on context enter.

        Returns self for optional use by callers.
        """
        if not _ENABLED:
            return self
        self._start = time.perf_counter()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        """Stop timing and emit a JSONL record on context exit.

        Swallow exceptions raised while writing to the timing output so the
        host process is not impacted.
        """
        if not _ENABLED or self._start is None:
            return False
        end = time.perf_counter()
        record = {
            "label": self.label,
            "elapsed_seconds": (end - self._start),
            "timestamp": time.time(),
            "thread": threading.current_thread().name,
        }
        record.update(self.extra)
        _write_record(record)
        return False


def timed(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that times function calls and logs a JSON line.

    When timings are disabled this becomes a no-op decorator.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not _ENABLED:
            return func

        @wraps(func)
        def _wrapped(*args: Any, **kwargs: Any):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                _write_record(
                    {
                        "label": label,
                        "elapsed_seconds": end - start,
                        "timestamp": time.time(),
                        "thread": threading.current_thread().name,
                    }
                )

        return _wrapped

    return _decorator


def summarize(path: str) -> dict[str, dict[str, float]]:
    """Summarise a JSONL timing file by label (count, mean, median, p95).

    Returns a dict: label -> stats dict.
    """
    import statistics

    if not os.path.exists(path):
        return {}
    groups: dict[str, list[float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = obj.get("label")
            elapsed = obj.get("elapsed_seconds")
            if label is None or elapsed is None:
                continue
            groups.setdefault(label, []).append(float(elapsed))

    out: dict[str, dict[str, float]] = {}
    for label, vals in groups.items():
        vals_sorted = sorted(vals)
        out[label] = {
            "count": float(len(vals)),
            "mean": float(statistics.mean(vals)),
            "median": float(statistics.median(vals)),
            "p95": float(vals_sorted[int(len(vals_sorted) * 0.95) - 1])
            if len(vals_sorted) > 1
            else float(vals_sorted[0]),
        }
    return out
