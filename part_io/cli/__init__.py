"""CLI entry-points for part-io."""

from __future__ import annotations

import sys
from typing import NoReturn


def handle_cli_error(exc: BaseException) -> NoReturn:
    """Print *exc* to stderr and exit with code 2."""
    print(str(exc), file=sys.stderr)
    sys.exit(2)
