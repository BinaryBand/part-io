"""Stream a remote file to disk over HTTP."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import httpx

_DOWNLOAD_TIMEOUT_SECONDS = 300.0


def download_file(*, url: str, destination_path: Path) -> None:
    """Stream the resource at *url* into *destination_path*.

    Creates the parent directory if needed and raises :class:`httpx.HTTPStatusError`
    on a non-2xx response.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS
    ) as response:
        response.raise_for_status()
        with destination_path.open("wb") as sink:
            for chunk in response.iter_bytes():
                sink.write(chunk)


__all__ = ["download_file"]
