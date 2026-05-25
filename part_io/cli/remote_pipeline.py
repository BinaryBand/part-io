"""Remote episode pipeline: detect ad-break positions, review them, cut them out.

Subcommands:
  review  — detect open/close matches per episode, review each interactively
  cut     — use state.toml to pair and cut ad segments
    loop    - detect -> review -> cut one episode at a time until done

State is stored entirely in {review_root}/state.toml.
No clip files or manifest CSVs are written.
Delete state.toml to start fresh.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TypeVar

from part_io.adapters.audio.matcher import warm_source_profile
from part_io.adapters.audio.snippet_profile import decode_matrix, snapshot_snippet_profile
from part_io.cli.remote._cut import CutSettings, _cut_cuttable
from part_io.cli.remote._detect import _detect_batch
from part_io.cli.remote._review import (
    ReviewItem,
    _collect_uncertain_candidates,
    _count_uncertain,
    _reclassify_all,
    _run_quiz,
    _run_review_loop,
)
from part_io.cli.remote._state import PipelineState, SnippetEntry
from part_io.utils.config import get_profile_cache_dir

_MIN_EPISODE_BYTES = 10 * 1024 * 1024  # skip promos — < 10 MB ≈ < 5 min at 128 kbps
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_AUDIO_EXTENSIONS = frozenset({".mp3", ".opus"})


def _build_config_init_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "config-init",
        help="Snapshot seed audio files into __state__.toml as embedded snippet profiles.",
    )
    _add_remote_dir_arg(p)
    p.add_argument("--open-seed", type=Path, required=True, help="Seed file for open snippet")
    p.add_argument("--close-seed", type=Path, required=True, help="Seed file for close snippet")
    p.add_argument("--intro-seed", type=Path, default=None, help="Optional seed file for intro")
    p.add_argument("--outro-seed", type=Path, default=None, help="Optional seed file for outro")
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing snippet profiles",
    )


def _snapshot_snippets(
    state: PipelineState,
    state_path: Path,
    *,
    open_seed: Path,
    close_seed: Path,
    intro_seed: Path | None = None,
    outro_seed: Path | None = None,
) -> None:
    """Snapshot seed audio files into state.snippets and save."""
    seed_map: dict[str, Path] = {"open": open_seed, "close": close_seed}
    if intro_seed is not None:
        seed_map["intro"] = intro_seed
    if outro_seed is not None:
        seed_map["outro"] = outro_seed

    for name, seed_path in seed_map.items():
        if not seed_path.exists() or not seed_path.is_file():
            sys.exit(f"Seed file for '{name}' not found: {seed_path}")

    new_snippets: list[SnippetEntry] = []
    for name, seed_path in seed_map.items():
        _emit(f"  [snapshot] {name}: {seed_path}")
        model = snapshot_snippet_profile(seed_path)
        profile = decode_matrix(model.data, model.n_frames, model.band_count)
        new_snippets.append(SnippetEntry(name=name, profile=profile))

    state.snippets = new_snippets
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state.save(state_path)
    _emit(f"Snippet profiles written to {state_path}.")


def _cmd_config_init(args: argparse.Namespace) -> None:
    state_path = args.remote_dir / "__state__.toml"
    state = PipelineState.load(state_path)

    if state.snippets and not args.force:
        sys.exit(
            f"Snippet profiles already exist in {state_path} (use --force to overwrite).\n"
            f"  Current snippets: {[s.name for s in state.snippets]}"
        )

    _snapshot_snippets(
        state,
        state_path,
        open_seed=args.open_seed,
        close_seed=args.close_seed,
        intro_seed=args.intro_seed,
        outro_seed=args.outro_seed,
    )


def _full_episodes(remote_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in remote_dir.iterdir()
        if p.suffix.lower() in _AUDIO_EXTENSIONS
        and p.is_file()
        and p.stat().st_size >= _MIN_EPISODE_BYTES
    )


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


_T = TypeVar("_T")


def _resolve_opt(cli_value: _T | None, state_value: _T) -> _T:
    return state_value if cli_value is None else cli_value


def _apply_sticky_review_args(args: argparse.Namespace, state: PipelineState) -> None:
    d = state.settings.detect
    args.step_seconds = float(_resolve_opt(args.step_seconds, d.step_seconds))
    args.workers = int(_resolve_opt(args.workers, d.workers))
    args.max_matches = int(_resolve_opt(args.max_matches, d.max_matches))
    # no_interactive is a session flag — never sticky; always use CLI value (default False)
    args.no_interactive = bool(args.no_interactive)
    args.overwrite = bool(_resolve_opt(args.overwrite, d.overwrite))

    d.step_seconds = args.step_seconds
    d.workers = args.workers
    d.max_matches = args.max_matches
    d.overwrite = args.overwrite


def _apply_sticky_cut_args(args: argparse.Namespace, state: PipelineState) -> None:
    c = state.settings.cut
    args.output_dir = Path(_resolve_opt(args.output_dir, c.output_dir))
    args.min_gap = float(_resolve_opt(args.min_gap, c.min_gap))
    args.max_gap = float(_resolve_opt(args.max_gap, c.max_gap))
    # yes/dry_run are session flags — never sticky; always use CLI value (default False)
    args.yes = bool(args.yes)
    args.dry_run = bool(args.dry_run)
    args.inclusive = bool(_resolve_opt(args.inclusive, c.ad_inclusive))
    args.fade = float(_resolve_opt(args.fade, c.fade))

    c.output_dir = str(args.output_dir)
    c.min_gap = args.min_gap
    c.max_gap = args.max_gap
    c.ad_inclusive = args.inclusive
    c.fade = args.fade


def _apply_sticky_loop_args(args: argparse.Namespace, state: PipelineState) -> None:
    _apply_sticky_review_args(args, state)
    _apply_sticky_cut_args(args, state)
    s = state.settings
    c = s.cut
    args.quiz_size = int(_resolve_opt(args.quiz_size, s.quiz_size))
    args.debug = bool(_resolve_opt(args.debug, c.debug))
    s.quiz_size = args.quiz_size
    c.debug = args.debug


# Cut helpers moved to part_io.cli.remote._cut


def _loop_work_counts(state: PipelineState, all_full: list[Path]) -> tuple[int, int, int]:
    """Return remaining undetected, uncertain, and cuttable counts."""
    n_undetected = sum(1 for ep in all_full if not state.episode(ep.stem).is_detected())
    n_uncertain = _count_uncertain(state)
    n_cuttable = sum(1 for ep in state.episodes.values() if ep.is_cuttable() and not ep.cut)
    return n_undetected, n_uncertain, n_cuttable


def _collect_loop_candidates(
    state: PipelineState,
    all_full: list[Path],
    *,
    overwrite: bool,
    quiz_size: int,
    step_seconds: float,
    workers: int,
    max_matches: int,
    state_path: Path,
    remote_dir: Path,
    profile_cache_dir: Path | None = None,
    exclude_decided: set[tuple[str, str, int]] | None = None,
) -> tuple[list[ReviewItem], int]:
    """Detect until enough uncertain candidates are available or no work remains.

    *exclude_decided* filters out candidates already reviewed this session so they
    are not re-presented before new detections arrive.
    """
    undetected = [ep for ep in all_full if overwrite or not state.episode(ep.stem).is_detected()]
    n_already = len(all_full) - len(undetected)
    _emit(f"Episodes: {len(all_full)} total, {n_already} detected, {len(undetected)} to detect")

    _excl = exclude_decided or set()
    quiz_items = [
        i
        for i in _collect_uncertain_candidates(state)
        if (i.stem, i.kind, i.candidate_idx) not in _excl
    ]
    while len(quiz_items) < quiz_size and undetected:
        ep_path = undetected.pop(0)
        _emit(f"\nDetecting {ep_path.stem}...")
        _detect_batch(
            [ep_path],
            state,
            remote_dir=remote_dir,
            step_seconds=step_seconds,
            workers=workers,
            max_matches=max_matches,
            profile_cache_dir=profile_cache_dir,
        )
        _reclassify_all(state)
        state.save(state_path)
        quiz_items = [
            i
            for i in _collect_uncertain_candidates(state)
            if (i.stem, i.kind, i.candidate_idx) not in _excl
        ]

    return quiz_items[:quiz_size], _count_uncertain(state)


def _run_loop_cut_pass(
    state: PipelineState,
    *,
    all_full: list[Path],
    remote_dir: Path,
    output_dir: Path,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    state_path: Path,
    ad_inclusive: bool,
    intro_exclusive: bool,
    fade_dur: float,
    debug: bool,
) -> tuple[int, int, int, int, int, int]:
    """Run a cut pass and return counts plus remaining work summary."""
    settings = CutSettings(
        min_gap=min_gap,
        max_gap=max_gap,
        yes=yes,
        dry_run=dry_run,
        ad_inclusive=ad_inclusive,
        intro_exclusive=intro_exclusive,
        fade_dur=fade_dur,
        debug=debug,
    )
    n_cut, n_skipped, n_failed = _cut_cuttable(
        state,
        remote_dir=remote_dir,
        output_dir=output_dir,
        settings=settings,
        state_path=state_path,
    )
    n_remain_undet, n_remain_unc, n_remain_cut = _loop_work_counts(state, all_full)
    return n_cut, n_skipped, n_failed, n_remain_undet, n_remain_unc, n_remain_cut


def _run_loop_once(
    state: PipelineState,
    *,
    all_full: list[Path],
    remote_dir: Path,
    output_dir: Path,
    snippets: dict[str, Path],
    quiz_size: int,
    overwrite: bool,
    step_seconds: float,
    workers: int,
    max_matches: int,
    state_path: Path,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    ad_inclusive: bool,
    intro_exclusive: bool,
    fade_dur: float,
    debug: bool,
) -> None:
    """Run one detect/review/cut pass in non-interactive mode."""
    quiz_items, n_unc = _collect_loop_candidates(
        state,
        all_full,
        overwrite=overwrite,
        quiz_size=quiz_size,
        step_seconds=step_seconds,
        workers=workers,
        max_matches=max_matches,
        state_path=state_path,
        remote_dir=remote_dir,
        profile_cache_dir=get_profile_cache_dir(remote_dir),
    )
    if quiz_items:
        _emit(f"\n{len(quiz_items)} candidate(s) to review ({n_unc} uncertain total).")
    else:
        msg = (
            f"{n_unc} uncertain remaining — nothing new to review."
            if n_unc
            else "Nothing to review."
        )
        _emit(f"\n{msg}")

    n_cut, n_skipped, n_failed, n_remain_undet, n_remain_unc, n_remain_cut = _run_loop_cut_pass(
        state,
        all_full=all_full,
        remote_dir=remote_dir,
        output_dir=output_dir,
        min_gap=min_gap,
        max_gap=max_gap,
        yes=yes,
        dry_run=dry_run,
        state_path=state_path,
        ad_inclusive=ad_inclusive,
        intro_exclusive=intro_exclusive,
        fade_dur=fade_dur,
        debug=debug,
    )
    _emit(
        f"\nProgress: {n_remain_undet} undetected, {n_remain_unc} uncertain,"
        f" {n_remain_cut} cuttable remaining."
    )
    if n_cut or n_skipped or n_failed:
        print(f"Cut: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        sys.exit(1)


def _run_loop_until_clean(
    state: PipelineState,
    *,
    all_full: list[Path],
    remote_dir: Path,
    output_dir: Path,
    snippets: dict[str, Path],
    quiz_size: int,
    overwrite: bool,
    step_seconds: float,
    workers: int,
    max_matches: int,
    state_path: Path,
    min_gap: float,
    max_gap: float,
    yes: bool,
    dry_run: bool,
    ad_inclusive: bool,
    intro_exclusive: bool,
    fade_dur: float,
    debug: bool,
) -> None:
    """Run repeat detect/review/cut passes until no work remains or the user quits."""
    # Accumulate rejected-uncertain candidates across outer passes so the same
    # candidate is never re-presented within the same session.
    session_skipped: set[tuple[str, str, int]] = set()
    while True:
        quiz_items, n_unc = _collect_loop_candidates(
            state,
            all_full,
            overwrite=overwrite,
            quiz_size=quiz_size,
            step_seconds=step_seconds,
            workers=workers,
            max_matches=max_matches,
            state_path=state_path,
            remote_dir=remote_dir,
            profile_cache_dir=get_profile_cache_dir(remote_dir),
            exclude_decided=session_skipped,
        )
        n_remain_undet, n_remain_unc, n_remain_cut = _loop_work_counts(state, all_full)
        if not n_remain_undet and not n_remain_unc and not n_remain_cut:
            _emit("\nDirectory is clean.")
            return

        if quiz_items:
            _emit(f"\n{len(quiz_items)} candidate(s) to review ({n_unc} uncertain total).")
            _, interrupted, new_skipped = _run_quiz(
                state,
                quiz_items,
                snippets=snippets,
                state_path=state_path,
                remote_dir=remote_dir,
                pre_skipped=session_skipped,
            )
            session_skipped |= new_skipped
            if interrupted:
                return
        else:
            msg = (
                f"{n_unc} uncertain remaining — nothing new to review."
                if n_unc
                else "Nothing to review."
            )
            _emit(f"\n{msg}")

        n_cut, n_skipped, n_failed, n_remain_undet, n_remain_unc, n_remain_cut = _run_loop_cut_pass(
            state,
            all_full=all_full,
            remote_dir=remote_dir,
            output_dir=output_dir,
            min_gap=min_gap,
            max_gap=max_gap,
            yes=yes,
            dry_run=dry_run,
            state_path=state_path,
            ad_inclusive=ad_inclusive,
            intro_exclusive=intro_exclusive,
            fade_dur=fade_dur,
            debug=debug,
        )
        _emit(
            f"\nProgress: {n_remain_undet} undetected, {n_remain_unc} uncertain,"
            f" {n_remain_cut} cuttable remaining."
        )
        if n_cut or n_skipped or n_failed:
            print(f"Cut: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
        if n_failed:
            sys.exit(1)
        if dry_run and not n_remain_undet and not n_remain_unc:
            _emit("Dry-run pass complete.")
            return
        if not n_remain_undet and not n_remain_unc and not n_remain_cut:
            _emit("Directory is clean.")
            return


# ---------------------------------------------------------------------------
# review subcommand
# ---------------------------------------------------------------------------


def _cmd_review(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"

    state = PipelineState.load(state_path)
    _apply_sticky_review_args(args, state)
    state.save(state_path)

    if not state.snippets:
        if not args.open_seed or not args.close_seed:
            sys.exit(
                f"No snippet profiles in {state_path}.\n"
                f"Re-run with --open-seed <file> --close-seed <file> to initialize."
            )
        _snapshot_snippets(
            state,
            state_path,
            open_seed=args.open_seed,
            close_seed=args.close_seed,
            intro_seed=args.intro_seed,
            outro_seed=args.outro_seed,
        )
    if not remote_dir.exists():
        sys.exit(f"Remote dir not found: {remote_dir}")

    all_full = _full_episodes(remote_dir)
    if not all_full:
        sys.exit(f"No full-length MP3s (>= 10 MB) found in {remote_dir}")

    to_detect = [
        ep for ep in all_full if args.overwrite or not state.episode(ep.stem).is_detected()
    ]
    n_already = len(all_full) - len(to_detect)
    _emit(f"Episodes: {len(all_full)} total, {n_already} detected, {len(to_detect)} to detect")

    if to_detect:
        _emit(f"\nDetecting {len(to_detect)} episode(s) with {args.workers} worker(s)...")
        _detect_batch(
            to_detect,
            state,
            remote_dir=remote_dir,
            step_seconds=args.step_seconds,
            workers=args.workers,
            max_matches=args.max_matches,
            profile_cache_dir=get_profile_cache_dir(remote_dir),
        )
        _reclassify_all(state)
        state.save(state_path)

    if args.no_interactive:
        _emit("\nDetection complete. Run remote-review without --no-interactive to label.")
        return

    n_unc = _count_uncertain(state)
    _emit(f"\n{n_unc} uncertain target(s) to review.")
    if not n_unc:
        _emit("Nothing to review — all episodes classified.")
        return

    _run_review_loop(
        state,
        snippets={},
        state_path=state_path,
        remote_dir=remote_dir,
    )


# ---------------------------------------------------------------------------
# cut subcommand
# ---------------------------------------------------------------------------


def _cmd_cut(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"

    state = PipelineState.load(state_path)
    _apply_sticky_cut_args(args, state)
    state.save(state_path)
    output_dir: Path = args.output_dir
    n_cuttable = sum(1 for ep in state.episodes.values() if ep.is_cuttable() and not ep.cut)
    if not n_cuttable:
        print("No cuttable episodes in state.toml — need open and close both positive.")
        return

    print(f"Found {n_cuttable} cuttable episode(s).")
    settings = CutSettings(
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        yes=args.yes,
        dry_run=args.dry_run,
        ad_inclusive=args.inclusive,
        intro_exclusive=state.settings.cut.intro_exclusive,
        fade_dur=args.fade,
    )
    n_cut, n_skipped, n_failed = _cut_cuttable(
        state,
        remote_dir=remote_dir,
        output_dir=output_dir,
        settings=settings,
        state_path=state_path,
    )
    print(f"\nDone: {n_cut} cut, {n_skipped} skipped, {n_failed} failed.")
    if n_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# loop subcommand - detect -> review -> cut in one pass
# ---------------------------------------------------------------------------


def _cmd_loop(args: argparse.Namespace) -> None:
    """Detect one episode at a time, accumulate uncertain candidates, quiz, then cut."""
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"

    state = PipelineState.load(state_path)
    _apply_sticky_loop_args(args, state)
    state.save(state_path)

    if not state.snippets:
        if not args.open_seed or not args.close_seed:
            sys.exit(
                f"No snippet profiles in {state_path}.\n"
                f"Re-run with --open-seed <file> --close-seed <file> to initialize."
            )
        _snapshot_snippets(
            state,
            state_path,
            open_seed=args.open_seed,
            close_seed=args.close_seed,
            intro_seed=args.intro_seed,
            outro_seed=args.outro_seed,
        )
    if not remote_dir.exists():
        sys.exit(f"Remote dir not found: {remote_dir}")

    output_dir: Path = args.output_dir

    all_full = _full_episodes(remote_dir)
    if not all_full:
        sys.exit(f"No full-length MP3s (>= 10 MB) found in {remote_dir}")
    if args.no_interactive:
        _run_loop_once(
            state,
            all_full=all_full,
            remote_dir=remote_dir,
            output_dir=output_dir,
            snippets={},
            quiz_size=args.quiz_size,
            overwrite=args.overwrite,
            step_seconds=args.step_seconds,
            workers=args.workers,
            max_matches=args.max_matches,
            state_path=state_path,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            yes=args.yes,
            dry_run=args.dry_run,
            ad_inclusive=args.inclusive,
            intro_exclusive=state.settings.cut.intro_exclusive,
            fade_dur=args.fade,
            debug=args.debug,
        )
        return

    _run_loop_until_clean(
        state,
        all_full=all_full,
        remote_dir=remote_dir,
        output_dir=output_dir,
        snippets={},
        quiz_size=args.quiz_size,
        overwrite=args.overwrite,
        step_seconds=args.step_seconds,
        workers=args.workers,
        max_matches=args.max_matches,
        state_path=state_path,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        yes=args.yes,
        dry_run=args.dry_run,
        ad_inclusive=args.inclusive,
        intro_exclusive=state.settings.cut.intro_exclusive,
        fade_dur=args.fade,
        debug=args.debug,
    )


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--step-seconds", type=float, default=None)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for detection",
    )
    p.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Top-N candidate positions to store",
    )


def _add_cut_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--min-gap",
        type=float,
        default=None,
        help="Min gap: open end -> close start",
    )
    p.add_argument(
        "--max-gap",
        type=float,
        default=None,
        help="Max gap in seconds (default 5 min)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        default=None,
        help="Cut without confirmation prompt",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Show cut plan without running ffmpeg",
    )
    p.add_argument(
        "--inclusive",
        action="store_true",
        default=None,
        help="Cut jingle transitions too",
    )
    p.add_argument(
        "--fade",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Fade duration at cut points (default 0.5s, 0 to disable)",
    )


def _add_remote_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "remote_dir",
        type=Path,
        nargs="?",
        default=Path("downloads/remote"),
        metavar="REMOTE_DIR",
        help="Directory of episode MP3s (default: downloads/remote)",
    )


def _add_verbose_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable INFO-level logging (shows ffmpeg decode progress, cache hits, etc.)",
    )


def _add_seed_args(p: argparse.ArgumentParser) -> None:
    """Optional seed files — only needed on first run if __state__.toml has no profiles."""
    p.add_argument("--open-seed", type=Path, default=None, metavar="FILE")
    p.add_argument("--close-seed", type=Path, default=None, metavar="FILE")
    p.add_argument("--intro-seed", type=Path, default=None, metavar="FILE")
    p.add_argument("--outro-seed", type=Path, default=None, metavar="FILE")


def _build_review_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("review", help="Detect matches and review them interactively")
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    _add_detect_args(p)
    _add_seed_args(p)
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument(
        "--no-interactive", action="store_true", default=None, help="Detect only, skip review"
    )


def _build_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("cut", help="Cut ad segments using labels from __state__.toml")
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument("--output-dir", type=Path, default=None)
    _add_cut_args(p)


def _build_loop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "loop",
        help="Detect one episode at a time, quiz uncertain candidates, then cut. Run repeatedly.",
    )
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument("--output-dir", type=Path, default=None)
    _add_detect_args(p)
    _add_cut_args(p)
    _add_seed_args(p)
    p.add_argument(
        "--quiz-size",
        type=int,
        default=None,
        help="Target number of uncertain candidates to collect before quizzing (default: 10)",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        default=None,
        help="Skip interactive review",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Re-detect and re-cut episodes",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Write each planned cut ad segment to output_dir/debug_ads for review.",
    )


def _build_precache_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "precache",
        help="Pre-warm spectral profile cache for all episodes overnight.",
    )
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Pause between episodes to reduce sustained CPU load (default: 5)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-build cache entries that already exist",
    )


def _cmd_precache(args: argparse.Namespace) -> None:
    import time

    remote_dir: Path = args.remote_dir
    sleep_s: float = args.sleep
    overwrite: bool = args.overwrite
    profile_cache_dir = get_profile_cache_dir(remote_dir)
    profile_cache_dir.mkdir(parents=True, exist_ok=True)

    episodes = _full_episodes(remote_dir)
    if not episodes:
        print("No episodes found.", file=sys.stderr)
        return

    cached = 0
    built = 0
    for i, ep in enumerate(episodes, 1):
        cache_path = profile_cache_dir / f"{ep.stem}.npz"
        if not overwrite and cache_path.exists():
            _LOG.info("[precache] %d/%d  %s  (cached)", i, len(episodes), ep.stem)
            cached += 1
            continue
        _LOG.info("[precache] %d/%d  %s  building…", i, len(episodes), ep.stem)
        print(f"[{i}/{len(episodes)}] {ep.stem}", flush=True)
        try:
            warm_source_profile(ep, profile_cache_dir)
            built += 1
        except Exception as exc:
            print(f"  WARNING: failed — {exc}", file=sys.stderr)
        if i < len(episodes) and sleep_s > 0:
            time.sleep(sleep_s)

    print(f"\nDone. {built} built, {cached} already cached.", flush=True)


def _precache_paths(remote_dir: Path) -> tuple[Path, Path]:
    """Return (pidfile, logfile) for background precache process."""
    base = remote_dir.parent
    return base / ".precache.pid", base / ".precache.log"


def _precache_running(pid_path: Path) -> int | None:
    """Return the running PID if the precache process is alive, else None."""
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _build_precache_start_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("precache-start", help="Start precache as a background process.")
    _add_remote_dir_arg(p)
    p.add_argument(
        "--sleep",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Pause between episodes (default: 10)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-build cache entries that already exist",
    )


def _build_precache_stop_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("precache-stop", help="Stop a running background precache process.")
    _add_remote_dir_arg(p)


def _build_precache_status_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("precache-status", help="Show background precache status and recent log.")
    _add_remote_dir_arg(p)
    p.add_argument(
        "--lines",
        type=int,
        default=20,
        metavar="N",
        help="Number of recent log lines to show (default: 20)",
    )


def _cmd_precache_start(args: argparse.Namespace) -> None:
    from part_io.utils.exec import launch_resolved

    remote_dir: Path = args.remote_dir
    pid_path, log_path = _precache_paths(remote_dir)

    existing = _precache_running(pid_path)
    if existing is not None:
        print(f"Already running (pid {existing}). Use precache-stop first.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "part_io.cli.remote_pipeline",
        "precache",
        "--verbose",
        "--sleep",
        str(args.sleep),
        str(remote_dir),
    ]
    if args.overwrite:
        cmd.append("--overwrite")

    log_f = log_path.open("a")
    devnull = open(os.devnull, "rb")
    proc = launch_resolved(
        cmd,
        stdout=log_f,
        stderr=log_f,
        stdin=devnull,
        start_new_session=True,  # detach from the terminal's process group
    )
    pid_path.write_text(str(proc.pid))
    print(f"Started precache (pid {proc.pid}).")
    print(f"Log: {log_path}")
    print("Stop with: poetry run part-io-tasks remote-precache-stop")


def _cmd_precache_stop(args: argparse.Namespace) -> None:
    import signal

    remote_dir: Path = args.remote_dir
    pid_path, _ = _precache_paths(remote_dir)

    pid = _precache_running(pid_path)
    if pid is None:
        print("No precache process is running.")
        pid_path.unlink(missing_ok=True)
        return

    os.kill(pid, signal.SIGTERM)
    pid_path.unlink(missing_ok=True)
    print(f"Stopped precache (pid {pid}).")


def _cmd_precache_status(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    pid_path, log_path = _precache_paths(remote_dir)

    pid = _precache_running(pid_path)
    if pid is None:
        print("Precache: not running.")
        pid_path.unlink(missing_ok=True)
    else:
        print(f"Precache: running (pid {pid})")

    if log_path.exists():
        lines = log_path.read_text().splitlines()
        tail = lines[-args.lines :]
        print(f"\n--- last {len(tail)} lines of {log_path} ---")
        print("\n".join(tail))
    else:
        print("No log file found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote episode review and ad-cut pipeline.")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    _build_config_init_parser(sub)
    _build_review_parser(sub)
    _build_cut_parser(sub)
    _build_loop_parser(sub)
    _build_precache_parser(sub)
    _build_precache_start_parser(sub)
    _build_precache_stop_parser(sub)
    _build_precache_status_parser(sub)
    args = parser.parse_args()

    if args.verbose if hasattr(args, "verbose") else False:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.subcommand == "review":
        _cmd_review(args)
    elif args.subcommand == "config-init":
        _cmd_config_init(args)
    elif args.subcommand == "cut":
        _cmd_cut(args)
    elif args.subcommand == "precache":
        _cmd_precache(args)
    elif args.subcommand == "precache-start":
        _cmd_precache_start(args)
    elif args.subcommand == "precache-stop":
        _cmd_precache_stop(args)
    elif args.subcommand == "precache-status":
        _cmd_precache_status(args)
    else:
        _cmd_loop(args)


if __name__ == "__main__":
    main()
