"""Audio clip playback, probing, and extraction helpers built around ffmpeg."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from partio.adapters.process.runner import run_resolved


def play_audio_segment(*, source_path: Path, start_seconds: float, duration_seconds: float) -> None:
    """Play a segment of *source_path* for auditioning, blocking until it ends."""
    command = [
        "ffplay",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nodisp",
        "-autoexit",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        str(source_path),
    ]
    result = run_resolved(command, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"ffplay failed to play segment: {source_path}")


def audio_duration_seconds(source_path: Path) -> float:
    """Return the playable length of *source_path* in seconds, via ffprobe.

    Raises ``ValueError`` when ffprobe fails or the container reports no
    duration (some streamed MP3s), so callers can fall back to an explicit
    region rather than searching a file of unknown length.
    """
    command = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]
    result = run_resolved(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed to read duration: {source_path}")
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise ValueError(f"ffprobe reported no duration for {source_path}") from None
    if duration <= 0:
        raise ValueError(f"ffprobe reported no duration for {source_path}")
    return duration


def extract_audio_clip(
    *, source_path: Path, destination_path: Path, start_seconds: float, duration_seconds: float
) -> None:
    """Extract a segment of *source_path* into an MP3 clip at *destination_path*."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source_path),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(destination_path),
    ]
    result = run_resolved(command, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg failed to write clip: {destination_path}")


__all__ = ["audio_duration_seconds", "extract_audio_clip", "play_audio_segment"]
