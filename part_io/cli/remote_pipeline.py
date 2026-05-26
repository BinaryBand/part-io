"""Remote episode pipeline with explicit, staged commands.

Subcommands:
    precache    — cache episode profiles (background by default)
    prep-quiz   — cache + detect candidates into __state__.toml (background by default)
    prep-cut    — run interactive review quiz and persist answers
    execute-cut — cut confident episodes from __state__.toml (background by default)

State is stored in {remote_dir}/__state__.toml.
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
from part_io.cli.remote._review import _count_uncertain, _reclassify_all, _run_review_loop
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


def _emit(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _background_paths(remote_dir: Path, job_name: str) -> tuple[Path, Path]:
    """Return (pidfile, logfile) for a detached background job."""
    base = remote_dir.parent
    safe = job_name.replace("-", "_")
    return base / f".{safe}.pid", base / f".{safe}.log"


def _background_running(pid_path: Path) -> int | None:
    """Return running PID if alive, else None."""
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def _launch_background_job(*, remote_dir: Path, job_name: str, cmd: list[str]) -> None:
    from part_io.utils.exec import launch_resolved

    pid_path, log_path = _background_paths(remote_dir, job_name)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _background_running(pid_path)
    if existing is not None:
        print(f"Already running (pid {existing}).", file=sys.stderr)
        sys.exit(1)

    log_f = log_path.open("a")
    devnull = open(os.devnull, "rb")
    proc = launch_resolved(
        cmd,
        stdout=log_f,
        stderr=log_f,
        stdin=devnull,
        start_new_session=True,
    )
    pid_path.write_text(str(proc.pid))
    print(f"Started {job_name} (pid {proc.pid}).")
    print(f"Log: {log_path}")


_T = TypeVar("_T")


def _resolve_opt(cli_value: _T | None, state_value: _T) -> _T:
    return state_value if cli_value is None else cli_value


def _apply_sticky_review_args(args: argparse.Namespace, state: PipelineState) -> None:
    d = state.settings.detect
    args.step_seconds = float(_resolve_opt(args.step_seconds, d.step_seconds))
    args.workers = int(_resolve_opt(args.workers, d.workers))
    args.max_matches = int(_resolve_opt(args.max_matches, d.max_matches))
    args.overwrite = bool(_resolve_opt(args.overwrite, d.overwrite))

    # Optional in some parsers; only normalize when present.
    if hasattr(args, "no_interactive"):
        args.no_interactive = bool(args.no_interactive)

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


def _cmd_prep_cut(args: argparse.Namespace) -> None:
    """Run interactive quiz only; detection should be prepared by prep-quiz."""
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"
    state = PipelineState.load(state_path)

    if not remote_dir.exists():
        sys.exit(f"Remote dir not found: {remote_dir}")

    # Fail fast if prep-quiz background job is still writing state.
    quiz_pid_path, _ = _background_paths(remote_dir, "prep-quiz")
    if _background_running(quiz_pid_path):
        sys.exit(
            "remote-prep-quiz is still running. "
            "Wait for it to finish before running remote-prep-cut."
        )

    if not state.episodes:
        sys.exit(
            f"No detection data in {state_path}.\n"
            "Run `remote-prep-quiz` first and wait for it to complete."
        )

    _reclassify_all(state)
    n_unc = _count_uncertain(state)
    n_und = sum(1 for ep in state.episodes.values() if not ep.is_detected())
    _emit(f"\n{n_unc} uncertain target(s) to review.")
    if n_und:
        _emit(f"  ({n_und} episode(s) not yet detected — re-run remote-prep-quiz)")
    if not n_unc:
        n_cut = sum(1 for ep in state.episodes.values() if ep.is_cuttable())
        _emit(f"All episodes classified — {n_cut} cuttable. Run remote-execute-cut to proceed.")
        return

    snippets = _resolve_review_snippets(args, remote_dir)
    if not snippets:
        _emit("No snippet audio found for compare playback; [c]ompare is disabled.")

    _run_review_loop(
        state,
        snippets=snippets,
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


def _cmd_execute_cut(args: argparse.Namespace) -> None:
    """Cut episodes in the foreground or detach by default."""
    remote_dir: Path = args.remote_dir
    if args.background:
        cmd = [
            sys.executable,
            "-m",
            "part_io.cli.remote_pipeline",
            "execute-cut",
            str(remote_dir),
            "--no-background",
            "--yes",
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.output_dir is not None:
            cmd += ["--output-dir", str(args.output_dir)]
        if args.min_gap is not None:
            cmd += ["--min-gap", str(args.min_gap)]
        if args.max_gap is not None:
            cmd += ["--max-gap", str(args.max_gap)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.inclusive:
            cmd.append("--inclusive")
        if args.fade is not None:
            cmd += ["--fade", str(args.fade)]
        _launch_background_job(remote_dir=remote_dir, job_name="execute-cut", cmd=cmd)
        return

    _cmd_cut(args)


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


def _add_background_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run detached in background (default: true).",
    )


def _build_execute_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("execute-cut", help="Execute confident cuts from __state__.toml")
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    p.add_argument("--output-dir", type=Path, default=None)
    _add_cut_args(p)
    _add_background_arg(p)


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
    _add_background_arg(p)


def _build_prep_quiz_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "prep-quiz",
        help="Populate __state__.toml by caching and detecting candidate matches.",
    )
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    _add_detect_args(p)
    _add_seed_args(p)
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Pause between episodes while precaching (default: 0)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-detect and re-build cached profiles",
    )
    _add_background_arg(p)


def _build_prep_cut_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prep-cut", help="Run interactive quiz and persist labels to state.")
    _add_remote_dir_arg(p)
    _add_verbose_arg(p)
    _add_seed_args(p)


def _resolve_review_snippets(args: argparse.Namespace, remote_dir: Path) -> dict[str, Path]:
    snippets: dict[str, Path] = {}
    explicit = {
        "open": args.open_seed,
        "close": args.close_seed,
        "intro": args.intro_seed,
        "outro": args.outro_seed,
    }
    for kind, seed in explicit.items():
        if seed is None:
            continue
        if not seed.exists() or not seed.is_file():
            sys.exit(f"Seed file for '{kind}' not found: {seed}")
        snippets[kind] = seed

    if snippets:
        return snippets

    default_dir = remote_dir.parent / "snippets"
    for kind in ("open", "close", "intro", "outro"):
        candidate = default_dir / f"{kind}.mp3"
        if candidate.exists() and candidate.is_file():
            snippets[kind] = candidate
    return snippets


def _cmd_precache(args: argparse.Namespace) -> None:
    import time

    remote_dir: Path = args.remote_dir
    if not remote_dir.exists() or not remote_dir.is_dir():
        sys.exit(f"Remote dir not found: {remote_dir}")

    if args.background:
        cmd = [
            sys.executable,
            "-m",
            "part_io.cli.remote_pipeline",
            "precache",
            str(remote_dir),
            "--no-background",
            "--sleep",
            str(args.sleep),
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.overwrite:
            cmd.append("--overwrite")
        _launch_background_job(remote_dir=remote_dir, job_name="precache", cmd=cmd)
        return

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


def _cmd_prep_quiz(args: argparse.Namespace) -> None:
    remote_dir: Path = args.remote_dir
    state_path = remote_dir / "__state__.toml"

    if args.background:
        cmd = [
            sys.executable,
            "-m",
            "part_io.cli.remote_pipeline",
            "prep-quiz",
            str(remote_dir),
            "--no-background",
            "--sleep",
            str(args.sleep),
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.step_seconds is not None:
            cmd += ["--step-seconds", str(args.step_seconds)]
        if args.workers is not None:
            cmd += ["--workers", str(args.workers)]
        if args.max_matches is not None:
            cmd += ["--max-matches", str(args.max_matches)]
        if args.open_seed is not None:
            cmd += ["--open-seed", str(args.open_seed)]
        if args.close_seed is not None:
            cmd += ["--close-seed", str(args.close_seed)]
        if args.intro_seed is not None:
            cmd += ["--intro-seed", str(args.intro_seed)]
        if args.outro_seed is not None:
            cmd += ["--outro-seed", str(args.outro_seed)]
        if args.overwrite:
            cmd.append("--overwrite")
        _launch_background_job(remote_dir=remote_dir, job_name="prep-quiz", cmd=cmd)
        return

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

    # prep-quiz includes precaching as a guaranteed side effect.
    precache_args = argparse.Namespace(
        remote_dir=remote_dir,
        verbose=args.verbose,
        sleep=args.sleep,
        overwrite=args.overwrite,
        background=False,
    )
    _cmd_precache(precache_args)

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

    n_unc = _count_uncertain(state)
    _emit(f"\nprep-quiz complete: {n_unc} uncertain target(s) ready for prep-cut.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote episode review and ad-cut pipeline.")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    _build_config_init_parser(sub)
    _build_precache_parser(sub)
    _build_prep_quiz_parser(sub)
    _build_prep_cut_parser(sub)
    _build_execute_cut_parser(sub)
    args = parser.parse_args()

    if args.verbose if hasattr(args, "verbose") else False:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.subcommand == "config-init":
        _cmd_config_init(args)
    elif args.subcommand == "precache":
        _cmd_precache(args)
    elif args.subcommand == "prep-quiz":
        _cmd_prep_quiz(args)
    elif args.subcommand == "prep-cut":
        _cmd_prep_cut(args)
    elif args.subcommand == "execute-cut":
        _cmd_execute_cut(args)
    else:
        parser.error(f"Unhandled subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
