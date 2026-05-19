"""Cross-platform project task runner.

This module provides a Python-native alternative to shell-specific Make recipes.
It keeps command behavior consistent across Windows, macOS, and Linux while
reusing discoverable CLI task modules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from part_io.adapters.config.task_registry_loader import (
    load_registry,
    select_tasks,
    tasks_for_profile,
)
from part_io.adapters.process.runner import run_resolved
from part_io.adapters.reporting.json_report_writer import write_lint_report
from part_io.services.lint_orchestrator import (
    execute_lint_tasks,
    select_lint_tasks,
    write_lint_report_if_requested,
)


def _run_cmd(cmd: list[str]) -> int:
    """Run *cmd* and return its exit code."""
    return int(run_resolved(cmd).returncode)


def _run_module(module: str) -> int:
    """Run a Python module with the current interpreter."""
    return _run_cmd([sys.executable, "-m", module])


def _add_profile_arg(cmd: list[str], profile: str | None) -> list[str]:
    """Return *cmd* with optional profile argument appended."""
    if profile:
        return [*cmd, "--profile", profile]
    return cmd


def _print_help() -> None:
    """Print available commands and declared lint tasks."""
    print("Available commands:")
    print("  help            Show this message")
    print("  install         Install development dependencies via poetry")
    print("  test            Run pytest")
    print("  audio-review-batch  Run batch audio review generation")
    print("  audio-ad-detect     Pair open/close detections into ad_segments.json")
    print("  audio-ad-remove     Cut detected ad segments from an episode MP3")
    print("  remote-review       Generate open/close review bundles from downloads/remote/")
    print("  remote-cut          Cut ad segments from labeled remote episodes")
    print("  remote-loop         Generate → review → cut one episode at a time (streamlined)")
    print("  remote-promote      Safely replace remote files with staged cleaned versions")
    print("  compile         Generate Pydantic model schemas into part_io/models/schemas")
    print("  generate-tasks  Regenerate config/generated.mk")
    print("  check-tasks     Verify config/generated.mk is current")
    print("  lint            Run declared lint tasks")
    print("  clean           Remove caches and build artifacts")

    tasks = tasks_for_profile(load_registry())
    if tasks:
        print("\nDeclared lint tasks:")
        for task in tasks:
            print(f"  {task.target:<18} {task.description}")


def _run_lint(
    selected: list[str] | None,
    *,
    profile: str | None = None,
    report_json: Path | None = None,
) -> int:
    """Run declared lint task modules with optional profile and report output."""
    if selected and profile:
        print("Cannot use explicit lint targets together with --profile.", file=sys.stderr)
        return 2

    try:
        registry, chosen = select_lint_tasks(
            selected,
            profile=profile,
            load_registry_fn=load_registry,
            select_tasks_fn=select_tasks,
            tasks_for_profile_fn=tasks_for_profile,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not chosen:
        print("No declared lint tasks found.")
        return 0

    results, exit_code = execute_lint_tasks(
        chosen,
        run_module_fn=_run_module,
        on_task_start=lambda task: print(f"==> {task.target}"),
    )

    write_lint_report_if_requested(
        report_json,
        selected=selected,
        profile=profile,
        default_profile=registry.default_profile,
        results=results,
        exit_code=exit_code,
        write_lint_report_fn=write_lint_report,
    )

    return exit_code


def _run_generate_tasks(profile: str | None, *, check: bool = False) -> int:
    """Run task-target generation command with optional profile."""
    cmd = _add_profile_arg([sys.executable, "-m", "part_io.cli.generate.tasks"], profile)
    if check:
        cmd.append("--check")
    return _run_cmd(cmd)


def _remove_tree(path: Path) -> None:
    """Recursively remove *path* using pathlib operations only."""
    if not path.exists():
        return

    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _clean() -> int:
    """Remove common Python build and cache artifacts."""
    root = Path.cwd()
    for dir_name in ["__pycache__", ".pytest_cache"]:
        for directory in root.rglob(dir_name):
            if directory.is_dir():
                _remove_tree(directory)

    for pyc in root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)

    for extra in [Path("dist"), Path(".coverage")]:
        if extra.is_dir():
            _remove_tree(extra)
        elif extra.exists():
            extra.unlink(missing_ok=True)

    return 0


_PASSTHROUGH_CMDS = {
    "audio-review-batch",
    "audio-ad-detect",
    "audio-ad-remove",
    "remote-review",
    "remote-cut",
    "remote-loop",
    "remote-promote",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform project task runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("help")
    sub.add_parser("install")
    sub.add_parser("test")
    for cmd in sorted(_PASSTHROUGH_CMDS):
        sub.add_parser(cmd)
    generate = sub.add_parser("generate-tasks")
    generate.add_argument("--profile", help="Registry profile to generate task targets for")
    check = sub.add_parser("check-tasks")
    check.add_argument("--profile", help="Registry profile to check generated task targets for")
    lint = sub.add_parser("lint")
    lint.add_argument("--profile", help="Registry profile to run when targets are not specified")
    lint.add_argument("--report-json", type=Path, help="Write lint execution report to JSON file")
    lint.add_argument("targets", nargs="*")
    sub.add_parser("compile")
    sub.add_parser("clean")
    return parser


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace, extra: list[str]) -> None:
    if args.command == "help":
        _print_help()
        return
    if args.command == "install":
        sys.exit(_run_cmd(["poetry", "install", "--with", "dev"]))
    if args.command == "test":
        sys.exit(_run_cmd(["poetry", "run", "pytest"]))
    if args.command == "audio-review-batch":
        sys.exit(_run_cmd([sys.executable, "-m", "part_io.cli.audio_review_batch", *extra]))
    if args.command == "audio-ad-detect":
        sys.exit(_run_cmd([sys.executable, "-m", "part_io.cli.audio_ad_detect", *extra]))
    if args.command == "audio-ad-remove":
        sys.exit(_run_cmd([sys.executable, "-m", "part_io.cli.audio_ad_remove", *extra]))
    if args.command in ("remote-review", "remote-cut", "remote-loop"):
        sub = args.command.split("-", 1)[1]
        sys.exit(_run_cmd([sys.executable, "-m", "part_io.cli.remote_pipeline", sub, *extra]))
    if args.command == "remote-promote":
        sys.exit(_run_cmd([sys.executable, "-m", "part_io.cli.remote_promote", *extra]))
    if args.command == "compile":
        sys.exit(_run_cmd([sys.executable, "-m", "part_io.cli.compile", *extra]))
    if args.command == "generate-tasks":
        sys.exit(_run_generate_tasks(args.profile))
    if args.command == "check-tasks":
        sys.exit(_run_generate_tasks(args.profile, check=True))
    if args.command == "lint":
        sys.exit(_run_lint(args.targets, profile=args.profile, report_json=args.report_json))
    if args.command == "clean":
        sys.exit(_clean())
    parser.error(f"Unhandled command: {args.command}")


def main() -> None:
    """Parse CLI args and execute project tasks."""
    parser = _build_parser()
    args, extra = parser.parse_known_args()
    if args.command not in _PASSTHROUGH_CMDS and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    _dispatch(parser, args, extra)


if __name__ == "__main__":
    main()
