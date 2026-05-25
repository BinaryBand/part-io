"""Set server output volume for the default PulseAudio/PipeWire sink.

Usage:
  python scripts/volume.py
  python scripts/volume.py 35

If no argument is provided, ``DEFAULT_VOLUME_PERCENT`` is used.
"""

from __future__ import annotations

import argparse
import sys

from part_io.utils.exec import CalledProcessError, run_resolved

# Dial this value once and run the script with no args.
DEFAULT_VOLUME_PERCENT = 100


def _volume_arg(raw: str) -> int:
    """Parse and validate an integer volume percentage in [0, 100]."""
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f"invalid volume '{raw}': must be an integer between 0 and 100"
        raise argparse.ArgumentTypeError(msg) from exc
    if value < 0 or value > 100:
        msg = f"invalid volume '{raw}': must be between 0 and 100"
        raise argparse.ArgumentTypeError(msg)
    return value


def set_default_sink_volume(percent: int) -> None:
    """Set volume on the current default sink."""
    run_resolved(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
        check=True,
    )
    # Ensure we can hear output if the sink was muted.
    run_resolved(
        ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set default output volume on this server (0-100).",
    )
    parser.add_argument(
        "volume",
        nargs="?",
        default=DEFAULT_VOLUME_PERCENT,
        type=_volume_arg,
        help=(
            f"Target volume percent (0-100). If omitted, uses default: {DEFAULT_VOLUME_PERCENT}."
        ),
    )

    args = parser.parse_args()
    try:
        set_default_sink_volume(args.volume)
    except FileNotFoundError:
        sys.stderr.write("Error: 'pactl' not found. Install pulseaudio-utils or pipewire-pulse.\n")
        return 1
    except CalledProcessError as exc:
        sys.stderr.write(f"Error: failed to set volume via pactl (exit {exc.returncode}).\n")
        return exc.returncode or 1

    print(f"Default sink volume set to {args.volume}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
