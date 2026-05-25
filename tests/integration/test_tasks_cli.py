"""Unit tests for task runner lint execution behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from part_io.cli import tasks as tasks_cli


def _patch_module_runner(monkeypatch, fn: Callable[[str], int]) -> None:
    """Helper: monkeypatch _run_module to use the provided function."""
    monkeypatch.setattr(tasks_cli, "_run_module", fn)


def test_run_lint_uses_selected_profile_and_writes_report(
    monkeypatch,
    test_registry,
    tmp_path: Path,
) -> None:
    """Profile mode should run profile tasks and write JSON report."""
    monkeypatch.setattr(tasks_cli, "load_registry", lambda: test_registry)

    seen_modules: list[str] = []

    def fake_run(module: str) -> int:
        seen_modules.append(module)
        return 0

    _patch_module_runner(monkeypatch, fake_run)

    report_path = tmp_path / "lint-report.json"
    rc = tasks_cli._run_lint([], profile="ci", report_json=report_path)

    assert rc == 0
    assert seen_modules == ["part_io.cli.lint.beta"]

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected_profile"] == "ci"
    assert payload["selected_targets"] == ["lint.beta"]
    assert payload["exit_code"] == 0
    assert payload["task_count"] == 1


def test_run_lint_rejects_profile_and_targets_together() -> None:
    """Target IDs and profile are mutually exclusive."""
    rc = tasks_cli._run_lint(["lint.alpha"], profile="strict")
    assert rc == 2


def test_run_lint_stops_on_first_failure(monkeypatch, test_registry, tmp_path: Path) -> None:
    """Execution should stop at first failed task and report partial results."""
    monkeypatch.setattr(tasks_cli, "load_registry", lambda: test_registry)

    seen_modules: list[str] = []

    def fail_on_alpha(module: str) -> int:
        seen_modules.append(module)
        return 3 if module.endswith("alpha") else 0

    _patch_module_runner(monkeypatch, fail_on_alpha)

    report_path = tmp_path / "lint-report.json"
    rc = tasks_cli._run_lint([], report_json=report_path)

    assert rc == 3
    assert seen_modules == ["part_io.cli.lint.alpha"]

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 3
    assert payload["failed_count"] == 1
    assert payload["task_count"] == 1
    assert payload["selected_targets"] == ["lint.alpha"]


def test_run_lint_returns_2_on_registry_error(monkeypatch) -> None:
    """Lint runner should map registry validation errors to CLI usage code 2."""
    monkeypatch.setattr(
        tasks_cli,
        "load_registry",
        lambda: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert tasks_cli._run_lint([], profile=None, report_json=None) == 2


def test_run_lint_no_tasks_prints_and_succeeds(monkeypatch) -> None:
    """Empty selected task list should return success without execution."""
    monkeypatch.setattr(tasks_cli, "load_registry", lambda: object())
    monkeypatch.setattr(tasks_cli, "tasks_for_profile", lambda _registry, profile=None: [])
    assert tasks_cli._run_lint([], profile=None, report_json=None) == 0


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["part_io-tasks", "install"], ["poetry", "install", "--with", "dev"]),
        (["part_io-tasks", "test"], ["poetry", "run", "pytest"]),
    ],
)
def test_main_dispatches_run_cmd(monkeypatch, argv: list[str], expected: list[str]) -> None:
    """Install and test commands should dispatch through _run_cmd."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        seen.append(cmd)
        return 0

    monkeypatch.setattr(tasks_cli, "_run_cmd", fake_run)
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc:
        tasks_cli.main()

    assert exc.value.code == 0
    assert seen == [expected]


def test_main_dispatches_audio_review_batch_with_passthrough_args(monkeypatch) -> None:
    """Audio review batch command should forward remaining args to batch module."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        seen.append(cmd)
        return 0

    monkeypatch.setattr(tasks_cli, "_run_cmd", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "part_io-tasks",
            "audio-review-batch",
            "--threshold",
            "0.9",
            "--overwrite",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        tasks_cli.main()

    assert exc.value.code == 0
    assert len(seen) == 1
    assert seen[0][:3] == [
        tasks_cli.sys.executable,
        "-m",
        "part_io.cli.audio_review_batch",
    ]
    assert seen[0][3:] == ["--threshold", "0.9", "--overwrite"]


def test_main_help_command(monkeypatch) -> None:
    """Help command should print help and return normally."""
    called = {"help": False}

    def fake_help() -> None:
        called["help"] = True

    monkeypatch.setattr(tasks_cli, "_print_help", fake_help)
    monkeypatch.setattr("sys.argv", ["part_io-tasks", "help"])

    tasks_cli.main()
    assert called["help"] is True


def test_main_compile_dispatch(monkeypatch) -> None:
    """Compile command should dispatch to schema compiler module."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        seen.append(cmd)
        return 0

    monkeypatch.setattr(tasks_cli, "_run_cmd", fake_run)
    monkeypatch.setattr("sys.argv", ["part_io-tasks", "compile"])

    with pytest.raises(SystemExit) as exc:
        tasks_cli.main()

    assert exc.value.code == 0
    assert seen == [[tasks_cli.sys.executable, "-m", "part_io.cli.generate.compile"]]


def test_main_remote_promote_dispatch(monkeypatch) -> None:
    """remote-promote command should dispatch to promotion module."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        seen.append(cmd)
        return 0

    monkeypatch.setattr(tasks_cli, "_run_cmd", fake_run)
    monkeypatch.setattr("sys.argv", ["part_io-tasks", "remote-promote", "downloads/remove"])

    with pytest.raises(SystemExit) as exc:
        tasks_cli.main()

    assert exc.value.code == 0
    assert seen == [
        [tasks_cli.sys.executable, "-m", "part_io.cli.remote_promote", "downloads/remove"]
    ]


def test_main_remote_prep_quiz_dispatch(monkeypatch) -> None:
    """remote-prep-quiz should dispatch to remote pipeline with prep-quiz subcommand."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        seen.append(cmd)
        return 0

    monkeypatch.setattr(tasks_cli, "_run_cmd", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["part_io-tasks", "remote-prep-quiz", "downloads/remote", "--workers", "4"],
    )

    with pytest.raises(SystemExit) as exc:
        tasks_cli.main()

    assert exc.value.code == 0
    assert seen == [
        [
            tasks_cli.sys.executable,
            "-m",
            "part_io.cli.remote_pipeline",
            "prep-quiz",
            "downloads/remote",
            "--workers",
            "4",
        ]
    ]


def test_main_lint_dispatch(monkeypatch) -> None:
    """Lint should dispatch to its dedicated handler."""
    called = {"lint": False}

    def fake_run_lint(_targets, *, profile=None, report_json=None) -> int:
        called["lint"] = profile == "strict" and report_json is not None
        return 0

    monkeypatch.setattr(tasks_cli, "_run_lint", fake_run_lint)

    monkeypatch.setattr(
        "sys.argv",
        ["part_io-tasks", "lint", "--profile", "strict", "--report-json", "x.json"],
    )
    with pytest.raises(SystemExit) as lint_exit:
        tasks_cli.main()
    assert lint_exit.value.code == 0

    assert called == {"lint": True}


def test_clean_removes_common_artifacts(tmp_path: Path, monkeypatch) -> None:
    """Clean command helper should remove cache folders and pyc artifacts."""
    pkg = tmp_path / "pkg"
    pycache = pkg / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "a.pyc").write_text("x", encoding="utf-8")
    (pkg / "b.pyc").write_text("x", encoding="utf-8")
    pytest_cache = tmp_path / ".pytest_cache"
    pytest_cache.mkdir()
    dist = tmp_path / "dist"
    dist.mkdir()
    coverage_file = tmp_path / ".coverage"
    coverage_file.write_text("x", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    rc = tasks_cli._clean()

    assert rc == 0
    assert not pycache.exists()
    assert not pytest_cache.exists()
    assert not dist.exists()
    assert not coverage_file.exists()
