# TODO

## Features

- [x] Auto-download sample episode(s) from RSS feeds. `adapters/feed/` fetches (httpx) and parses (feedparser) the feed; episodes land in `static/downloads/`.
- [x] Collapse `feed` and `library` into `feed` alone. The library is now virtual: `cli/library/` enumerates every episode of every remembered feed as a `Track` whether or not it is downloaded, and `ensure_local` fetches only the one a picker actually returns. That deleted the whole `library` command group and `feed download` -- 7 menu entries became 3 -- and left `static/library.json` as an internal cache index nothing addresses directly.
- [ ] Tweak CLI to be more guided.

## Cleanup

- [x] Remove dead `_build_cmd` wrappers from the lint entrypoints. Deleted the seven `_build_cmd` defs and their now-unused `build_tool_cmd` imports from `partio/cli/lint/<tool>.py`, and repointed `tests/unit/adapters/test_lint_adapters.py` and `tests/unit/audio/test_audio_cli.py` at `registry.build_tool_cmd("<tool>", cfg)` so they exercise the real builder.
- [x] Drop the `cli/lint/entrypoints.py` seam entirely. Inlined `sys.exit(run_registered_tool("<tool>"))` into each `main()` and deleted the module plus its test -- both `run_single_tool_entrypoint` (a one-line pass-through) and `run_multi_tool_entrypoint` (no production callers, only its own test) were dead. Also removed the `entrypoints.py` entry from the `partio-cli-lint-shape` allowlist in `config/semgrep/scaffold.yml`.

## Docs

- [ ] Update docs to remove package-specific references and keep extension instructions package-agnostic.

## Constraints To Keep Enforced

- One obvious module path per concern.
- No compatibility shims in steady state.
- No import-time side effects outside entrypoints.
- Explicit, narrow error boundaries.
- Prefer small, standard libraries over custom plumbing.
