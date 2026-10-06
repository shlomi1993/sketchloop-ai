# T01a design: domain records and generator contract

Status: implemented. Decision record: [ADR 0005](../decisions/0005-domain-records-and-generator-contract.md).

Serves T01a in [ROADMAP.md](../ROADMAP.md): R02 (uniform generator layer, substitutable test adapter), R03 (explicit feedback for unsupported controls), R04 (multiple candidates, explicit selection), and R08 (typed, replaceable interfaces). Impact: High, because later boundaries reuse these records. Standard library only, Python 3.11+.

## Conventions

- Records are `@dataclass(frozen=True, slots=True, kw_only=True)` and validate in `__post_init__`, raising `InvalidRecordError`. Mapping fields are copied into a read-only `types.MappingProxyType`, so records compare by value but are not hashable or `asdict`-compatible. T05 adds explicit serialization.
- Enumerations are `typing.Literal` aliases, checked at runtime with `typing.get_args`.
- IDs are `uuid.uuid4().hex`, created by a `default_factory` where the record is built. Tests assert format and uniqueness, not exact values.
- Persisted records reference images only through `ImageRef`. Bytes appear only in the transient `GenerationOutput.payloads`.
- Requested settings live in the request. Effective settings are reported by the adapter and never copied from the request. Unknown values are `Unavailable(reason)`, never guesses.
- `Iteration` and `SelectionEvent` carry `schema_version: int = SCHEMA_VERSION` (1).
- `Unavailable` takes its reason positionally, for example `Unavailable("fake backend has no model")`. Other records are keyword-only.

## `src/sketchloop/domain.py`

Imports no other SketchLoop module and no third-party package.

```python
SCHEMA_VERSION: Final = 1
ControlValue: TypeAlias = bool | int | float | str
ColorMode: TypeAlias = Literal["L", "RGB", "RGBA"]  # Pillow-style names, RGB assumed sRGB
ExecutionKind: TypeAlias = Literal["fake", "in_process", "remote"]
IssueCode: TypeAlias = Literal["unknown_control", "wrong_type", "out_of_range", "invalid_choice", "unsupported_feature",
                               "too_many_candidates"]
```

| Record | Fields | Invariants |
| --- | --- | --- |
| `Unavailable` | `reason: str` | Non-empty reason. |
| `ImageRef` | `path: str`, `width: int`, `height: int`, `mode: ColorMode`, `media_type: str`, `sha256: str` | `path` is a relative POSIX key under the experiment artifact root: no leading `/`, no `\`, no drive prefix, no empty, `.`, or `..` segments. Sizes are `int` (not `bool`) and at least 1. `media_type` looks like `type/subtype`. `sha256` is 64 lowercase hex characters. |
| `Guidance` | `prompt: str`, `negative_prompt: str \| None = None`, `controls: Mapping[str, ControlValue] = {}` | `None` means no negative prompt requested, distinct from `""`. Control names match `[a-z][a-z0-9_]*`. Floats are finite. |
| `GenerationRequest` | `sketch: ImageRef`, `guidance: Guidance`, `n_candidates: int = 1`, `seed: int \| None = None` | `n_candidates` is an `int` of at least 1. `seed` is `None` (backend chooses) or a non-negative `int`, not `bool`. |
| `BackendIdentity` | `adapter: str`, `adapter_version: str`, `execution: ExecutionKind`, `model_id: str \| Unavailable` | Non-empty strings. Property `is_fake` returns `execution == "fake"`. |
| `EffectiveSettings` | `prompt: str \| Unavailable`, `negative_prompt: str \| None \| Unavailable`, `controls: Mapping[str, ControlValue \| Unavailable]` | `controls` includes defaults the backend applied. |
| `Candidate` | `id: str = <uuid4 hex>`, `index: int`, `image: ImageRef`, `seed: int \| Unavailable` | `index` at least 0. An unreported seed is `Unavailable`. |
| `GenerationResult` | `candidates: tuple[Candidate, ...]`, `backend: BackendIdentity`, `effective: EffectiveSettings` | At least one candidate. Unique IDs and image paths. Indices are `0..n-1` in order. |
| `Iteration` | `schema_version`, `id: str = <uuid4 hex>`, `parent_id: str \| None`, `request: GenerationRequest`, `result: GenerationResult` | `parent_id != id`. `len(result.candidates) == request.n_candidates`, so returning fewer images is an error. |
| `SelectionEvent` | `schema_version`, `id: str = <uuid4 hex>`, `iteration_id: str`, `selected_candidate_ids: tuple[str, ...]`, `created_at: datetime = <now, UTC>` | IDs unique, in the order given. Empty means explicit no-selection. `created_at` is timezone-aware. Property `is_no_selection`. |

```python
def select_candidates(iteration: Iteration, candidate_ids: Sequence[str]) -> SelectionEvent: ...
def record_no_selection(iteration: Iteration) -> SelectionEvent: ...
```

`select_candidates` raises `InvalidSelectionError` for an empty list (use `record_no_selection`), a repeated ID, or any ID not among `iteration.result.candidates`, naming the offending IDs. Selection never reads scores.

```python
class SketchLoopError(Exception): ...
class InvalidRecordError(SketchLoopError, ValueError): ...  # record invariant violated
class InvalidSelectionError(SketchLoopError, ValueError): ...

@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationIssue:
    field: str  # dotted path, for example "guidance.controls.steps" or "seed"
    code: IssueCode
    message: str

class UnsupportedConfigurationError(SketchLoopError):
    def __init__(self, issues: Sequence[ValidationIssue]) -> None: ...  # at least one issue, message joins them all
    issues: tuple[ValidationIssue, ...]
```

`InvalidRecordError` stays because records are built from adapter output now and loaded from storage in T05, and callers need one catchable SketchLoop type for bad data.

## `src/sketchloop/generation.py`

Imports `domain` only.

| Record | Fields | Invariants |
| --- | --- | --- |
| `ControlSpec` | `name: str`, `kind: Literal["int", "float", "bool", "choice"]`, `minimum: float \| None = None`, `maximum: float \| None = None`, `choices: tuple[str, ...] = ()`, `default: ControlValue \| None = None` | Bounds only for `int` and `float`, with `minimum <= maximum`. `choice` needs unique non-empty choices. A default must pass its own spec. |
| `GeneratorCapabilities` | `controls: tuple[ControlSpec, ...]`, `max_candidates: int`, `supports_negative_prompt: bool`, `supports_seed: bool` | Unique control names. `max_candidates` at least 1. |
| `GenerationOutput` | `result: GenerationResult`, `payloads: Mapping[str, bytes]` | Keys equal the candidate image paths. Each payload's SHA-256 equals its `ImageRef.sha256`. Transient, never persisted. |

```python
def request_issues(request: GenerationRequest, capabilities: GeneratorCapabilities) -> tuple[ValidationIssue, ...]: ...
def validate_request(request: GenerationRequest, capabilities: GeneratorCapabilities) -> None: ...

class Generator(Protocol):
    def capabilities(self) -> GeneratorCapabilities: ...
    def generate(self, request: GenerationRequest) -> GenerationOutput: ...
```

`request_issues` returns every problem, ordered by `n_candidates`, `seed`, `guidance.negative_prompt`, then controls by name. `validate_request` raises `UnsupportedConfigurationError` if any exist.

| Rule | Code |
| --- | --- |
| Control name not declared | `unknown_control` |
| Wrong value type (`bool` is never an `int` or `float`, an `int` is accepted for `float`) | `wrong_type` |
| Value outside inclusive bounds | `out_of_range` |
| Choice not allowed | `invalid_choice` |
| Seed given without `supports_seed`, or negative prompt given without `supports_negative_prompt` | `unsupported_feature` |
| `n_candidates > max_candidates` | `too_many_candidates` |

`generate` is synchronous so it fits both in-process and remote (Colab) backends. It must call `validate_request` first. For now the adapter builds candidates and uses the path `candidates/<candidate id><ext>`. It does not write storage, score, or select.

Paper parameters map as follows: `steps`, `guidance_scale`, and `strength` are controls. Batch size is `n_candidates`. Seed is `seed`. ControlNet input and CLIP filtering are deferred.

## `src/sketchloop/fakes.py`

```python
class FakeGenerator:
    def __init__(self, capabilities: GeneratorCapabilities | None = None, *, size: tuple[int, int] = (8, 8)) -> None: ...
    def capabilities(self) -> GeneratorCapabilities: ...
    def generate(self, request: GenerationRequest) -> GenerationOutput: ...
```

- A deterministic test double, not a model. Default capabilities: `steps` int 1 to 50 (default 4), `guidance_scale` float 0 to 20 (default 7.5), `strength` float 0 to 1 (default 0.75), `max_candidates` 8, negative prompt and seed supported.
- Produces binary PGM images (`image/x-portable-graymap`, mode `L`) whose pixels come from SHA-256 over canonical JSON of the sketch checksum, prompt, negative prompt, sorted controls, index, and effective seed. Same request, same bytes.
- The effective seed is `(request.seed or 0) + index`, which the fake really uses. Effective controls are defaults merged with requested values.
- Identity is `adapter="sketchloop.fake"`, `execution="fake"`, and `model_id=Unavailable("fake backend has no model")`. The T03 milestone must assert `not result.backend.is_fake`.

## Deferred

| Item | Target |
| --- | --- |
| `Failure` record, iteration status, retry link, `start_iteration`, `complete_iteration`, `fail_iteration`, timings | T05 (status and timing may be needed earlier by T04) |
| `BackendError`, `BackendTimeoutError`, fake failure injection, conditioning images (ControlNet), replay-guarantee enum, backend `details`, sketch-byte reader for real adapters | T03 |
| Constraint kinds (style, functional), evaluation, repeated-selection rules, iteration stage tags (D1 to D4) | T04 or later |
| Building candidates outside adapters (`assemble_result`), ID-source injection | Revisit when a second adapter or T05 needs it |
| Capture and preprocessing contracts, serialization, storage, lineage checks against stored iterations, UI types | T02, T05, T06 |

## File plan for the python-developer

Write tests first and run `python3 scripts/check.py` after each step.

1. `tests/test_domain.py`, then `src/sketchloop/domain.py`.
2. `tests/test_generation.py`, then `src/sketchloop/generation.py`.
3. `tests/test_fakes.py`, then `src/sketchloop/fakes.py`, including one end-to-end test that runs the fake, builds an iteration, selects, and builds a child iteration.
4. Update `docs/STATUS.md`, the T01 evidence in `docs/ROADMAP.md`, and this status line.

## Test list for the qa-engineer

Use synthetic values. Compute checksums with `hashlib` or use `"a" * 64`, because committed random hex can trip the publication guard's digit-run check.

| Behavior or failure path | Requirements |
| --- | --- |
| `ImageRef` rejects absolute, backslash, drive, `..`, `.`, and empty-segment paths, zero or `bool` sizes, unknown mode, and bad checksums. It accepts `sketches/s1.png`. | R08 |
| Records are immutable, including mapping fields, and are unaffected by later changes to input dicts. | R08 |
| `Guidance` rejects malformed control names and non-finite floats. `None` and `""` negative prompts stay distinct. | R03 |
| Each validation code is produced by its rule, at both bounds and just outside them (parametrized). | R03 |
| A request with several problems yields all issues in one error, in the documented order, with dotted field paths. Omitted controls and inclusive bounds pass. | R03 |
| `ControlSpec` and `GeneratorCapabilities` reject inconsistent specs and duplicate names. | R02, R03 |
| `FakeGenerator.generate` raises `UnsupportedConfigurationError` for an invalid request. | R02, R03 |
| The fake is deterministic across instances, and output changes when prompt, a control, seed, or sketch checksum changes. | R02 |
| The fake returns `n_candidates` candidates with seeds `seed + index`, payloads matching paths and checksums, `is_fake` true, and `Unavailable` model ID. | R02, R04 |
| With no controls requested, the request controls are empty while effective controls list the defaults. | R02, R03 |
| `Iteration` rejects a result with fewer candidates than requested and `parent_id == id`. Generated IDs are 32-character lowercase hex and unique. | R04, R08 |
| `select_candidates` records valid IDs in order and rejects foreign, unknown, duplicate, and empty selections. `record_no_selection` gives `is_no_selection` true and a timezone-aware time. | R04 |
| A minimal in-test adapter works with validation and selection unchanged. `domain.py` imports no other SketchLoop or third-party module (AST check). | R02, R08 |
