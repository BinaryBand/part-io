# TODO

## Cleanup

- [x] Remove dead `_build_cmd` wrappers from the lint entrypoints. Deleted the seven `_build_cmd` defs and their now-unused `build_tool_cmd` imports from `part_io/cli/lint/<tool>.py`, and repointed `tests/unit/adapters/test_lint_adapters.py` and `tests/unit/audio/test_audio_cli.py` at `registry.build_tool_cmd("<tool>", cfg)` so they exercise the real builder.
- [ ] (Optional) Drop the `run_single_tool_entrypoint` seam in `cli/lint/entrypoints.py`. It is a one-line pass-through (`return run_tool_fn(tool_key)`); each `main()` could call `sys.exit(run_registered_tool("<tool>"))` directly, letting the single-tool helper go. Keep `run_multi_tool_entrypoint` (the fail-fast loop earns its keep). Lower priority -- this is a deliberate contract seam.

## Docs

- [ ] Update docs to remove package-specific references and keep extension instructions package-agnostic.

## Constraints To Keep Enforced

- One obvious module path per concern.
- No compatibility shims in steady state.
- No import-time side effects outside entrypoints.
- Explicit, narrow error boundaries.
- Prefer small, standard libraries over custom plumbing.
