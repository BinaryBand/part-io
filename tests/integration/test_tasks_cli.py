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


def _capture_cmd_runner(monkeypatch) -> list[list[str]]:
    """Helper: patch _run_cmd to capture commands and return success."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str]) -> int:
        seen.append(cmd)
        return 0

    monkeypatch.setattr(tasks_cli, "_run_cmd", fake_run)
    return seen


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


def test_run_generate_tasks_appends_profile_argument(monkeypatch) -> None:
    """Profile-aware generate should pass --profile through to generator command."""
    seen = _capture_cmd_runner(monkeypatch)
    rc = tasks_cli._run_generate_tasks("ci")
    assert rc == 0
    assert len(seen) == 1
    assert seen[0][-2:] == ["--profile", "ci"]


def test_run_generate_tasks_check_appends_flags(monkeypatch) -> None:
    """Check mode should include both profile and --check."""
    seen = _capture_cmd_runner(monkeypatch)
    rc = tasks_cli._run_generate_tasks("strict", check=True)
    assert rc == 0
    assert len(seen) == 1
    assert seen[0][-3:] == ["--profile", "strict", "--check"]


def test_main_generate_tasks_profile_passthrough(monkeypatch) -> None:
    """CLI should pass profile flag to generate command path."""
    monkeypatch.setattr(
        tasks_cli,
        "_run_generate_tasks",
        lambda profile, *, check=False: 0 if profile == "ci" and not check else 1,
    )
    monkeypatch.setattr(
        tasks_cli,
        "_run_cmd",
        lambda _cmd: pytest.fail("_run_cmd should not be called directly for generate-tasks"),
    )
    monkeypatch.setattr(tasks_cli, "_print_help", lambda: None)
    monkeypatch.setattr(tasks_cli, "_clean", lambda: 0)
    monkeypatch.setattr(tasks_cli, "_run_lint", lambda *_args, **_kwargs: 0)

    monkeypatch.setattr("sys.argv", ["part_io-tasks", "generate-tasks", "--profile", "ci"])

    with pytest.raises(SystemExit) as exc:
        tasks_cli.main()

    assert exc.value.code == 0


def test_add_profile_arg_helper() -> None:
    """Profile helper should append --profile only when provided."""
    assert tasks_cli._add_profile_arg(["x"], None) == ["x"]
    assert tasks_cli._add_profile_arg(["x"], "ci") == ["x", "--profile", "ci"]


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


def test_main_help_command(monkeypatch) -> None:
    """Help command should print help and return normally."""
    called = {"help": False}

    def fake_help() -> None:
        called["help"] = True

    monkeypatch.setattr(tasks_cli, "_print_help", fake_help)
    monkeypatch.setattr("sys.argv", ["part_io-tasks", "help"])

    tasks_cli.main()
    assert called["help"] is True


def test_main_lint_and_check_tasks_dispatch(monkeypatch) -> None:
    """Lint and check-tasks should dispatch to their dedicated handlers."""
    called = {"lint": False, "check": False}

    def fake_run_lint(_targets, *, profile=None, report_json=None) -> int:
        called["lint"] = profile == "strict" and report_json is not None
        return 0

    def fake_generate(profile, *, check=False) -> int:
        called["check"] = check and profile == "ci"
        return 0

    monkeypatch.setattr(tasks_cli, "_run_lint", fake_run_lint)
    monkeypatch.setattr(tasks_cli, "_run_generate_tasks", fake_generate)

    monkeypatch.setattr(
        "sys.argv",
        ["part_io-tasks", "lint", "--profile", "strict", "--report-json", "x.json"],
    )
    with pytest.raises(SystemExit) as lint_exit:
        tasks_cli.main()
    assert lint_exit.value.code == 0

    monkeypatch.setattr("sys.argv", ["part_io-tasks", "check-tasks", "--profile", "ci"])
    with pytest.raises(SystemExit) as check_exit:
        tasks_cli.main()
    assert check_exit.value.code == 0

    assert called == {"lint": True, "check": True}


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
