# ADR 0005: Domain records and generator contract

Status: accepted

## Context

T01a needs the first typed contracts for images, guidance, generation results, iterations, and human selection (R02, R03, R04, R08). Later boundaries depend on them. The package has no application code or runtime dependencies (ADR 0001). The development machine has no GPU and the first real backend may run on Colab (ADR 0004). EXPERIMENTS.md requires distinct requested and effective settings, explicit unavailable values, and explicit selection. The owner asked for the smallest useful slice.

## Decision

Adopt [the T01a design](../design/t01a-contracts.md), standard library only, Python 3.11+:

- `sketchloop.domain`: frozen, keyword-only dataclasses validated on construction, `Literal` enumerations, `uuid4` hex IDs created where records are built, `SketchLoopError` with `InvalidRecordError`, `InvalidSelectionError`, and `UnsupportedConfigurationError` (which lists every problem). Records reference images through `ImageRef` (relative path, size, mode, media type, SHA-256). An `Iteration` holds a request, its result, and an optional parent. Selection is a separate `SelectionEvent` from `select_candidates` or `record_no_selection`.
- `sketchloop.generation`: `ControlSpec`, `GeneratorCapabilities`, `request_issues`, `validate_request`, and a synchronous `Generator` protocol returning a result plus transient image bytes.
- `sketchloop.fakes`: a deterministic `FakeGenerator` producing PGM bytes and identifying itself with `execution="fake"`.

Alternatives considered: Pydantic or attrs (a dependency without a concrete need), `TypedDict` or `NamedTuple` (mutable or positional and unvalidated), bytes inside candidates (records stop being serializable), and a fuller first slice with iteration status, failures, retries, and conditioning (deferred by the owner to T03 and T05).

## Consequences

Contracts are testable offline. Remote and in-process adapters share one interface. Adapters currently build candidates and choose paths `candidates/<id><ext>`, which T05 may revise. Mapping fields make records unhashable and incompatible with `dataclasses.asdict`, so T05 needs explicit serialization and migrations before changing fields. Iterations exist only after successful generation until T05 adds status, failures, and retry links. The deferred list in the design names the target task for each omitted item.
