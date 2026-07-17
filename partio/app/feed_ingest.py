"""Pure planning of which feed episodes to download and where they land."""

from __future__ import annotations

from typing import TYPE_CHECKING

from partio.core.models import DownloadPlan
from partio.utils.text import slugify

if TYPE_CHECKING:
    from pathlib import Path

    from partio.core.models import FeedEpisode


def plan_downloads(
    episodes: list[FeedEpisode],
    *,
    count: int,
    dest_dir: Path,
    existing_paths: set[Path],
) -> list[DownloadPlan]:
    """Choose up to *count* episodes to download into *dest_dir*.

    Episodes keep feed order (newest first for podcasts). Each maps to
    ``dest_dir/<slug-of-title>.mp3``; any whose destination is already in
    *existing_paths* is skipped so re-runs do not re-download or duplicate
    library entries. Performs no I/O -- only path arithmetic.
    """
    plans: list[DownloadPlan] = []
    for episode in episodes:
        if len(plans) >= count:
            break
        destination_path = dest_dir / f"{slugify(episode.title)}.mp3"
        if destination_path in existing_paths:
            continue
        plans.append(
            DownloadPlan(
                episode=episode,
                destination_path=destination_path,
                label=episode.title or destination_path.stem,
            )
        )
    return plans


__all__ = ["plan_downloads"]
