"""Safely promote cleaned episode files into a target remote directory.

Workflow:
1) Produce cleaned files into a staging directory (e.g. downloads/remove).
2) Promote staged files into target directory with per-file temp write + atomic replace.
3) Optionally back up replaced originals.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class PromoteResult:
    replaced: int = 0
    skipped: int = 0
    failed: int = 0


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_tmp_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.partio.tmp")


def _copy_with_fsync(source: Path, target_tmp: Path) -> None:
    with source.open("rb") as src, target_tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def _promote_one(
    source: Path,
    target: Path,
    *,
    backup_root: Path | None,
    dry_run: bool,
) -> tuple[bool, str]:
    if not target.exists():
        return False, f"SKIP {target.name}: target file does not exist"

    if dry_run:
        return True, f"DRY  {target.name}: would replace target"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _safe_tmp_path(target)

    try:
        _copy_with_fsync(source, tmp_path)

        source_size = source.stat().st_size
        tmp_size = tmp_path.stat().st_size
        if source_size != tmp_size:
            raise OSError(
                f"size mismatch while staging {target.name}: source={source_size} tmp={tmp_size}"
            )

        if backup_root is not None:
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_path = backup_root / target.name
            if backup_path.exists():
                backup_path = backup_root / f"{target.stem}.{_timestamp()}{target.suffix}"
            os.replace(target, backup_path)

        os.replace(tmp_path, target)
        return True, f"OK   {target.name}: replaced"
    except OSError as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False, f"FAIL {target.name}: {exc}"


def _collect_staged_files(staging_dir: Path) -> list[Path]:
    return sorted(p for p in staging_dir.glob("*.mp3") if p.is_file())


def _run(
    staging_dir: Path,
    target_dir: Path,
    *,
    backup_dir: Path | None,
    dry_run: bool,
) -> PromoteResult:
    staged = _collect_staged_files(staging_dir)
    if not staged:
        print(f"No staged MP3s found in {staging_dir}", file=sys.stderr)
        return PromoteResult(failed=1)

    replaced = skipped = failed = 0
    print(f"Found {len(staged)} staged MP3(s) in {staging_dir}")

    for source in staged:
        ok, message = _promote_one(
            source,
            target_dir / source.name,
            backup_root=backup_dir,
            dry_run=dry_run,
        )
        print(message)
        if ok:
            if message.startswith("DRY"):
                skipped += 1
            else:
                replaced += 1
        else:
            if "does not exist" in message:
                skipped += 1
            else:
                failed += 1

    return PromoteResult(replaced=replaced, skipped=skipped, failed=failed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely replace remote episodes with staged cleaned files."
    )
    parser.add_argument(
        "staging_dir",
        type=Path,
        nargs="?",
        default=Path("downloads/remove"),
        help="Directory containing staged cleaned MP3s (default: downloads/remove)",
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        nargs="?",
        default=Path("downloads/remote"),
        help="Target directory to replace in-place (default: downloads/remote)",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help=(
            "Directory where originals are moved before replacement. "
            "Default: <target_dir>_backup_<timestamp>"
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not keep backups of replaced originals.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be replaced without writing any files.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.staging_dir.exists():
        sys.exit(f"Staging dir not found: {args.staging_dir}")
    if not args.target_dir.exists():
        sys.exit(f"Target dir not found: {args.target_dir}")

    backup_dir: Path | None
    if args.no_backup:
        backup_dir = None
    elif args.backup_dir is not None:
        backup_dir = args.backup_dir
    else:
        backup_dir = args.target_dir.parent / f"{args.target_dir.name}_backup_{_timestamp()}"

    result = _run(
        args.staging_dir,
        args.target_dir,
        backup_dir=backup_dir,
        dry_run=args.dry_run,
    )

    print(f"Done: {result.replaced} replaced, {result.skipped} skipped, {result.failed} failed")
    if result.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
