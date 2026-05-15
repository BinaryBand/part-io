"""Entry-point tests for lint tool wrappers and shared runner."""

from __future__ import annotations

import pytest

from part_io.cli.lint import cpd, lizard, ruff, semgrep, ty, vulture


@pytest.mark.parametrize(
    ("module", "expected_keys", "code"),
    [
        (cpd, ["cpd"], 0),
        (lizard, ["lizard"], 0),
        (ruff, ["ruff"], 0),
        (semgrep, ["semgrep"], 0),
        (ty, ["ty"], 0),
        (vulture, ["vulture"], 0),
    ],
)
def test_lint_main_wrappers_exit_with_registered_tool_code(
    monkeypatch,
    module,
    expected_keys,
    code,
) -> None:
    """Each wrapper main() should map registered-tool return code to SystemExit."""
    seen: list[str] = []

    def fake_runner(key: str) -> int:
        seen.append(key)
        return code

    monkeypatch.setattr(module, "run_registered_tool", fake_runner)

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == code
    assert seen == expected_keys
