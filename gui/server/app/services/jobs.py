"""Generic one-way progress bridge for SSE endpoints.

Shared by every "start an operation, stream progress, stream one final
result" flow (download, search, locate, cut-plan): runs a blocking function
on a worker thread and turns whatever it reports into an async generator of
events, without blocking the event loop. Unlike the bootstrap WebSocket
bridge, this is one-way -- the worker never waits on the client -- so a plain
queue with no answer channel is enough.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

T = TypeVar("T")

Report = Callable[[dict[str, Any]], None]


async def stream_job(
    work: Callable[[Report], T],
    *,
    on_success: Callable[[T], dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Run *work* on a worker thread, yielding the events it reports.

    *work* receives a ``report(event)`` callback it may call any number of
    times from the worker thread; its return value is passed through
    *on_success* to build the final ``{"type": "result", ...}`` event. An
    exception from *work* becomes a ``{"type": "error", "message": ...}``
    event instead of propagating, so one failed job can't kill the SSE
    connection. ``None`` is the queue's own end-of-stream sentinel (real
    events are always a populated dict, never ``None``).
    """
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def report(event: dict[str, Any]) -> None:
        events.put(event)

    def run() -> None:
        try:
            result = work(report)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the client below
            events.put({"type": "error", "message": str(exc)})
        else:
            events.put(on_success(result))
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    while True:
        event = await asyncio.to_thread(events.get)
        if event is None:
            return
        yield event


def sse_encode(event: dict[str, Any]) -> str:
    """Encode one event as a single Server-Sent Events ``data:`` line."""
    return f"data: {json.dumps(event, default=str)}\n\n"
