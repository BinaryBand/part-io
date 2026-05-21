# TODO

## Pattern Standardization

- [x] Define two explicit lint wrapper contracts in docs and tests:

- single-tool wrapper
- multi-phase wrapper
- [x] Add architecture tests to ensure wrappers do not add hidden orchestration except when explicitly declared multi-phase.
- [x] Introduce typed domain exceptions for registry/config/process boundaries and map exit codes only at entrypoints.
- [x] Reduce stringly-typed key drift by centralizing and validating key naming grammar for task IDs and tool keys.
- [x] Separate template mode from instantiated-project mode:
  - placeholders allowed in template assets
  - no unresolved placeholders allowed in instantiated project
- [ ] Update docs to remove package-specific references and keep extension instructions package-agnostic.

## Scaffolding

### Current Pain Points

- [x] CLI layer currently mixes orchestration and adapter concerns.
- [x] Utility modules include behavior that is task-specific rather than cross-cutting.
- [x] Some modules still imply Make-centric workflows rather than generic Python task execution.

### Recommended Target Structure

```text
part_io/
  adapters/
    config/
      task_registry_loader.py
      lint_config_loader.py
    lint/
      runner.py
    process/
      runner.py
    reporting/
      json_report_writer.py
  cli/
    tasks.py
    generate/
      tasks.py
    lint/
      entrypoints.py
      coverage.py
      lizard.py
      ruff.py
      semgrep.py
      ty.py
      vulture.py
      registry.py
  models/
    lint.py
    registry.py
    results.py
    ports/
  services/
    lint_orchestrator.py
  utils/
    exec.py
    coverage.py
```

### Migration Checklist

- [x] Move process execution concerns into adapters/process.
- [x] Move config loading/parsing into adapters/config.
- [x] Move report serialization into adapters/reporting.
- [x] Keep domain models/services free of print/sys.exit/subprocess.
- [x] Keep entrypoints as the only process boundary for printing and exit behavior.

## High-Yield Refactoring

- [x] Standardize lint wrappers with a tiny shared entrypoint helper to reduce repetitive files.
- [x] Split lint adapter responsibilities:
  - config read/parse
  - command execution
  - error translation
- [x] Ensure non-entrypoint modules do not print or call sys.exit.
- [x] Consolidate extension metadata ownership to avoid synchronization across multiple files.
- [x] Remove Make-centric language in generator messaging and docs.
- [x] Add architectural guardrail tests:
  - no import-time side effects outside entrypoints
  - no reverse imports from core/domain into entrypoints
  - one-way dependency flow

## Constraints To Keep Enforced

- [x] One obvious module path per concern.
- [x] No compatibility shims in steady state.
- [x] No import-time side effects outside entrypoints.
- [x] Explicit, narrow error boundaries.
- [x] Prefer small, standard libraries over custom plumbing.

## Audio Alignment Refinement

Manual review identified cases where reported audio onset can be 3+ seconds off true alignment due to coarse feature-frame granularity. See [.alignment_plan.md](.alignment_plan.md) for detailed root-cause analysis and refinement strategy (sub-step refinement, onset anchoring, cross-correlation).

Current policy: the `--refine` path is disabled in active CLIs and moved behind an optional experimental plugin seam. Baseline detection is the default operational path.

- [x] Partition refine into optional plugin boundary (`part_io.utils.refine_plugin`).
- [x] Remove `--refine` from active CLI surfaces.
- [ ] Fix plugin refinement offset bug and verify lag convention.
- [ ] Benchmark plugin refinement on typical episodes; compare against baseline.
- [ ] Re-enable refine only after tests prove net alignment improvement.
