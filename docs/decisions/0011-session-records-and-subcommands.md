# ADR 0011: Session records, show, and rerun subcommands

Status: accepted

## Context

T05 (R06, R09) must keep each session's inputs, requested and effective settings, outputs, timings, lineage, and selections on disk, reopen them without a model, and rerun a round as a linked attempt. [ADR 0006](0006-first-backend-store-and-ui.md) chose a standard-library filesystem store, and [ADR 0010](0010-sessions-and-evaluator.md) fixed the folder layout `runs/<session>/round-<n>/`. The domain records ([ADR 0005](0005-domain-records-and-generator-contract.md)) hold `Mapping` fields, so `dataclasses.asdict` cannot serialize them. Diffusers does not guarantee identical output across devices or versions ([ADR 0009](0009-diffusers-backend.md)).

## Decision

Adopt [the T05 design](../design/t05-records.md):

- **Records:** one `session.json` with schema version, session ID, creation time, Python, platform, package versions including sketchloop-ai, and `rerun_of`, but no git revision because people change parameters rather than code, and one `round-<n>/round.json` per round with input and preprocessing steps, iteration and parent IDs, requested request, backend identity, effective settings, candidates with checksums, stage timings, evaluator and scores, selection, and failure. Unknown values are `{"unavailable": "<reason>"}`.
- **Serialization:** explicit to-dict and from-dict functions in a new `sketchloop.experiments` that rebuild domain records through their validating constructors. Atomic writes through a temporary file and `os.replace`, written after each round. `round.json` is replaced exactly once, to add the selection.
- **Loading:** structural faults in schema version, invalid fields, broken lineage, and paths outside the round folder raise `ExperimentRecordError`. Missing or changed artifacts are listed as problems so a damaged session can still be inspected.
- **CLI:** subcommands `run` for today's behavior, `show` to inspect without a model, and `rerun --round N`, which starts a new session linked by `rerun_of`, reuses the stored processed sketch, pins effective controls, mode, and base seed, and reports byte equality and environment differences without claiming exact replay.
- **Failed rounds** are recorded with stage, error class, and a one-line message, then the error is re-raised.

Alternatives considered: write-once files with a separate `selection.json` give stricter immutability but one more file per round. Re-preprocessing the raw sketch on rerun needs less code, but preprocessing changes would confound generation comparisons. A nested `iterations/` and `selections/` layout from the research note does not match the ADR 0010 folders. MLflow was rejected in ADR 0006.

## Consequences

`sketchloop <sketch> --prompt ...` without `run` stops working, so the README and CLI tests change. Sessions saved before T05 have no records and cannot be shown. Rerunning a diffusers round needs the generation extra and model folders, and it reports device, version, and revision differences instead of guaranteeing equal images. The fake backend gives byte-identical reruns, which the tests rely on. A store protocol for replaceable storage is deferred to T07. Changing a persisted field later needs a schema version bump and a migration.
