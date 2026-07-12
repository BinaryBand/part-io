"""Human-in-the-loop jingle discovery via tiled scanning and monotone bisection.

The auditor port answers yes/no questions about audio segments (a human
listening via the CLI, or a synthetic oracle in tests). Discovery walks
contiguous tiles across a hinted region; tuning bisects two monotone
predicates -- "already playing when this clip starts?" for the onset and
"still playing when this clip ends?" for the offset.
"""

from __future__ import annotations

from part_io.models.ports.audio import AuditorFn

_TILE_QUESTION = "Is the jingle anywhere in this clip?"
_ONSET_QUESTION = "Is the jingle already playing when this clip STARTS?"
_OFFSET_QUESTION = "Is the jingle STILL playing when this clip ENDS?"


def _discover_yes_span(
    *, auditor: AuditorFn, region_start: float, region_end: float, tile_seconds: float
) -> tuple[float, float, float, float] | None:
    """Walk contiguous tiles until the jingle is heard.

    Returns ``(onset_low, yes_start, yes_end, offset_high)`` where
    ``[yes_start, yes_end]`` covers the consecutive tiles that answered yes,
    ``onset_low`` is the start of the preceding "no" tile (or the region start
    when the jingle overlaps the first tile), and ``offset_high`` is the end of
    the trailing "no" tile (or the region end when the jingle may run past it).
    Returns ``None`` when no tile contains the jingle.
    """
    previous_start: float | None = None
    tile_start = region_start

    while tile_start < region_end:
        tile_end = min(tile_start + tile_seconds, region_end)
        if auditor(tile_start, tile_end - tile_start, _TILE_QUESTION):
            onset_low = previous_start if previous_start is not None else region_start
            yes_start = tile_start
            yes_end, offset_high = _extend_yes_span(
                auditor=auditor,
                yes_end=tile_end,
                region_end=region_end,
                tile_seconds=tile_seconds,
            )
            return onset_low, yes_start, yes_end, offset_high
        previous_start = tile_start
        tile_start = tile_end

    return None


def _extend_yes_span(
    *, auditor: AuditorFn, yes_end: float, region_end: float, tile_seconds: float
) -> tuple[float, float]:
    """Scan forward from the first "yes" tile to the next "no" tile.

    Returns ``(yes_end, offset_high)``: the end of the last consecutive "yes"
    tile and the end of the trailing "no" tile (``region_end`` when the span
    reaches the region boundary without one).
    """
    while yes_end < region_end:
        tile_end = min(yes_end + tile_seconds, region_end)
        if not auditor(yes_end, tile_end - yes_end, _TILE_QUESTION):
            return yes_end, tile_end
        yes_end = tile_end
    return yes_end, region_end


def _find_true_point(
    *, auditor: AuditorFn, yes_start: float, yes_end: float, probe_seconds: float
) -> float | None:
    """Find a point where the jingle is known to be already playing.

    Probes the midpoint of the yes-span first (sufficient for jingles longer
    than a tile), then scans the span at ``probe_seconds`` granularity for
    short jingles. Returns ``None`` when no probe answers yes.
    """
    midpoint = (yes_start + yes_end) / 2
    if auditor(midpoint, probe_seconds, _ONSET_QUESTION):
        return midpoint

    probe = yes_start
    while probe < yes_end:
        if auditor(probe, probe_seconds, _ONSET_QUESTION):
            return probe
        probe += probe_seconds
    return None


def _bisect_onset(
    *, auditor: AuditorFn, low: float, high: float, probe_seconds: float, resolution: float
) -> float:
    """Bisect the onset: lowest point where the jingle is already playing."""
    while high - low > resolution:
        midpoint = (low + high) / 2
        if auditor(midpoint, probe_seconds, _ONSET_QUESTION):
            high = midpoint
        else:
            low = midpoint
    return high


def _bisect_offset(
    *, auditor: AuditorFn, low: float, high: float, probe_seconds: float, resolution: float
) -> float:
    """Bisect the offset: highest point where the jingle is still playing."""
    while high - low > resolution:
        midpoint = (low + high) / 2
        clip_start = max(midpoint - probe_seconds, 0.0)
        if auditor(clip_start, midpoint - clip_start, _OFFSET_QUESTION):
            low = midpoint
        else:
            high = midpoint
    return low


def locate_jingle_span(
    *,
    auditor: AuditorFn,
    region_start: float,
    region_end: float,
    tile_seconds: float = 10.0,
    probe_seconds: float = 1.5,
    resolution: float = 0.5,
) -> tuple[float, float] | None:
    """Locate a jingle inside ``[region_start, region_end)`` via the auditor.

    Args:
        auditor: Port answering yes/no for ``(start, duration, question)``.
        region_start: Start of the search region in seconds.
        region_end: End of the search region in seconds.
        tile_seconds: Width of the contiguous discovery tiles.
        probe_seconds: Length of the short clips auditioned during tuning.
        resolution: Stop bisecting once the bracket is narrower than this.

    Returns:
        The ``(onset, offset)`` span in seconds, the unrefined yes-tile bounds
        when no probe inside the span answers yes (noisy edges or a
        sub-probe-length jingle), or ``None`` when no tile contains the jingle.

    Raises:
        ValueError: If the region is empty or a tuning parameter is not
            positive.
    """
    if region_end <= region_start:
        raise ValueError("region_end must be greater than region_start")
    if min(tile_seconds, probe_seconds, resolution) <= 0:
        raise ValueError("tile_seconds, probe_seconds, and resolution must be positive")

    span = _discover_yes_span(
        auditor=auditor,
        region_start=region_start,
        region_end=region_end,
        tile_seconds=tile_seconds,
    )
    if span is None:
        return None
    onset_low, yes_start, yes_end, offset_high = span

    true_point = _find_true_point(
        auditor=auditor, yes_start=yes_start, yes_end=yes_end, probe_seconds=probe_seconds
    )
    if true_point is None:
        return yes_start, yes_end

    onset = _bisect_onset(
        auditor=auditor,
        low=onset_low,
        high=true_point,
        probe_seconds=probe_seconds,
        resolution=resolution,
    )
    offset = _bisect_offset(
        auditor=auditor,
        low=true_point,
        high=offset_high,
        probe_seconds=probe_seconds,
        resolution=resolution,
    )
    return onset, offset


__all__ = ["locate_jingle_span"]
