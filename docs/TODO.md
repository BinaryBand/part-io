# TODO

## Cleanup

- [x] Remove dead `_build_cmd` wrappers from the lint entrypoints. Deleted the seven `_build_cmd` defs and their now-unused `build_tool_cmd` imports from `part_io/cli/lint/<tool>.py`, and repointed `tests/unit/adapters/test_lint_adapters.py` and `tests/unit/audio/test_audio_cli.py` at `registry.build_tool_cmd("<tool>", cfg)` so they exercise the real builder.
- [x] Drop the `cli/lint/entrypoints.py` seam entirely. Inlined `sys.exit(run_registered_tool("<tool>"))` into each `main()` and deleted the module plus its test -- both `run_single_tool_entrypoint` (a one-line pass-through) and `run_multi_tool_entrypoint` (no production callers, only its own test) were dead. Also removed the `entrypoints.py` entry from the `part_io-cli-lint-shape` allowlist in `config/semgrep/scaffold.yml`.

## Docs

- [ ] Update docs to remove package-specific references and keep extension instructions package-agnostic.

## Constraints To Keep Enforced

- One obvious module path per concern.
- No compatibility shims in steady state.
- No import-time side effects outside entrypoints.
- Explicit, narrow error boundaries.
- Prefer small, standard libraries over custom plumbing.
