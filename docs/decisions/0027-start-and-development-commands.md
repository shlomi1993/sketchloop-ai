# ADR 0027: Start and development commands

Status: accepted

## Context

The owner wants the primary CLI to emphasize the browser workflow while keeping research and development tools available. Existing terminal commands provide generation, inspection, reconstruction (R06), and benchmarks (R09). The browser page serves R07 and the guided study. The owner approved this command structure and immediate removal of the old paths, without adding tests.

## Decision

Rename `sketchloop ui` to `sketchloop start`, preserving every launch option, including `--seed` and `--candidates`. Bare `sketchloop` still launches the page. Configuration keys such as `[ui]` remain unchanged.

Group `run`, `show`, `rerun`, and `benchmark` under visible `sketchloop dev`. The group and its tools appear in help. Keep the existing leaf command dispatch, settings merge, and implementations, with no new service or persistence changes.

Keep `sketchloop study summary` and `sketchloop study export` visible. Moving these functions to the page was considered and rejected by the owner as unnecessary effort. No UI additions are part of this change.

## Consequences

Old command paths stop working immediately, with no aliases. Users and scripts must replace `ui` with `start` and prefix the four development commands with `dev`. Current usage documentation and existing tests follow the new paths. Historical ADRs and measured experiment transcripts preserve the commands used at the time. Generation, storage, and study behavior remain unchanged. Validate with existing tests and offline checks, without adding tests or invoking real models or hardware.
