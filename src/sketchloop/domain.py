import math
import re
import uuid

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, get_args


# Persisted records carry this version so later readers can migrate older runs.
SCHEMA_VERSION: Final = 1
ControlValue: TypeAlias = bool | int | float | str
ColorMode: TypeAlias = Literal["L", "RGB", "RGBA"]
ExecutionKind: TypeAlias = Literal["fake", "in_process", "remote"]
IssueCode: TypeAlias = Literal[
    "unknown_control", "wrong_type", "out_of_range", "invalid_choice", "unsupported_feature", "too_many_candidates"
]

# Formats checked with fullmatch, so the patterns need no anchors.
_CONTROL_NAME = re.compile(r"[a-z][a-z0-9_]*")
_MEDIA_TYPE = re.compile(r"[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SketchLoopError(Exception):
    """
    Base class for SketchLoop errors.
    """


class InvalidRecordError(SketchLoopError, ValueError):
    """
    A record was built with values that break its invariants.
    """


class InvalidSelectionError(SketchLoopError, ValueError):
    """
    A selection names candidates outside the iteration or repeats them.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationIssue:
    """
    One problem found while validating a request against backend capabilities.
    """
    field: str
    code: IssueCode
    message: str


class UnsupportedConfigurationError(SketchLoopError):
    """
    A request uses settings the backend cannot honor, listing every issue found.
    """

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        # An error without issues would give the person nothing to fix.
        if not issues:
            raise ValueError("UnsupportedConfigurationError needs at least one issue.")

        # Keep the issues for callers and join them into one readable message.
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        super().__init__("Unsupported request: " + " | ".join(f"{issue.field}: {issue.message}" for issue in self.issues))


def _new_id() -> str:
    return uuid.uuid4().hex


def require(condition: object, message: str) -> None:
    """
    Raise InvalidRecordError with the message when a record invariant does not hold.
    """
    if not condition:
        raise InvalidRecordError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True, slots=True)
class Unavailable:
    """
    Marks an unknown value, with the reason it could not be recorded.
    """

    reason: str

    def __post_init__(self) -> None:
        require(_is_text(self.reason), "Unavailable needs a non-empty reason.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRef:
    """
    Reference to a stored image by relative path, size, color mode, media type, and checksum.
    """
    path: str
    width: int
    height: int
    mode: ColorMode
    media_type: str
    sha256: str

    def __post_init__(self) -> None:
        # Reject Windows separators, drive letters, and traversal so the path stays inside the run folder.
        unsafe = "\\" in self.path or ":" in self.path or {"", ".", ".."} & set(self.path.split("/"))
        require(not unsafe, f"Image path {self.path!r} must be relative POSIX without ':', empty, '.', or '..' parts.")

        # Check the image metadata needed to reopen and verify the stored file.
        require(_is_int(self.width) and self.width >= 1, f"Image width must be an int >= 1, got {self.width!r}.")
        require(_is_int(self.height) and self.height >= 1, f"Image height must be an int >= 1, got {self.height!r}.")
        require(self.mode in get_args(ColorMode), f"Image mode must be L, RGB, or RGBA, got {self.mode!r}.")
        require(_MEDIA_TYPE.fullmatch(self.media_type), f"Media type must be type/subtype, got {self.media_type!r}.")
        require(_SHA256.fullmatch(self.sha256), "Image sha256 must be 64 lowercase hex characters.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Guidance:
    """
    Text guidance and named controls the person supplies for generation.
    """
    prompt: str
    negative_prompt: str | None = None
    controls: Mapping[str, ControlValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep control names portable and values serializable, since NaN and infinity break JSON records.
        for name, value in self.controls.items():
            require(isinstance(name, str) and _CONTROL_NAME.fullmatch(name), f"Control {name!r} must be snake_case.")
            require(not isinstance(value, float) or math.isfinite(value), f"Control {name} must be finite.")

        # Copy into a read-only view so later changes to the caller's dict cannot alter the record.
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationRequest:
    """
    Everything a generator needs to produce candidates from a sketch.
    """
    sketch: ImageRef
    guidance: Guidance
    n_candidates: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        require(_is_int(self.n_candidates) and self.n_candidates >= 1, "n_candidates must be an int >= 1.")
        require(self.seed is None or (_is_int(self.seed) and self.seed >= 0), "seed must be None or an int >= 0.")


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIdentity:
    """
    Which adapter and model produced a result, and how it ran.
    """
    adapter: str
    adapter_version: str
    execution: ExecutionKind
    model_id: str | Unavailable

    def __post_init__(self) -> None:
        require(_is_text(self.adapter) and _is_text(self.adapter_version), "Adapter and version must be non-empty.")
        require(self.execution in get_args(ExecutionKind), "Execution must be fake, in_process, or remote.")
        require(isinstance(self.model_id, Unavailable) or _is_text(self.model_id), "Model ID must be non-empty.")

    @property
    def is_fake(self) -> bool:
        """
        Tell whether the result came from the test-only fake backend.
        """
        return self.execution == "fake"


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveSettings:
    """
    Settings the backend reports it actually used, including defaults it applied.
    """
    prompt: str | Unavailable
    negative_prompt: str | None | Unavailable
    controls: Mapping[str, ControlValue | Unavailable]

    def __post_init__(self) -> None:
        # Copy into a read-only view so the recorded settings cannot change after the fact.
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate:
    """
    One generated alternative with its image and effective seed.
    """
    id: str = field(default_factory=_new_id)
    index: int
    image: ImageRef
    seed: int | Unavailable

    def __post_init__(self) -> None:
        require(_is_int(self.index) and self.index >= 0, f"Candidate index must be an int >= 0, got {self.index!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationResult:
    """
    Candidates from one generation call plus the backend and effective settings.
    """
    candidates: tuple[Candidate, ...]
    backend: BackendIdentity
    effective: EffectiveSettings

    def __post_init__(self) -> None:
        # Candidates need distinct IDs and files so a selection and its stored image are unambiguous.
        count = len(self.candidates)
        require(count >= 1, "A generation result needs at least one candidate.")
        require(len({candidate.id for candidate in self.candidates}) == count, "Candidate IDs must be unique.")
        require(len({candidate.image.path for candidate in self.candidates}) == count, "Image paths must be unique.")

        # Indices must match display order with no gaps.
        indices = [candidate.index for candidate in self.candidates]
        require(indices == list(range(count)), f"Candidate indices must be 0..{count - 1} in order, got {indices}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Iteration:
    """
    One round of the loop, linking a request, its result, and the parent iteration.
    """
    schema_version: int = SCHEMA_VERSION
    id: str = field(default_factory=_new_id)
    parent_id: str | None
    request: GenerationRequest
    result: GenerationResult

    def __post_init__(self) -> None:
        # Block a self-loop in the lineage and a backend that silently returned a different candidate count.
        require(self.parent_id != self.id, "An iteration cannot be its own parent.")
        expected, actual = self.request.n_candidates, len(self.result.candidates)
        require(actual == expected, f"Backend returned {actual} candidates but {expected} were requested.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionEvent:
    """
    The person's explicit choice of candidates, or an explicit choice of none.
    """
    schema_version: int = SCHEMA_VERSION
    id: str = field(default_factory=_new_id)
    iteration_id: str
    selected_candidate_ids: tuple[str, ...]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        # Require unique picks and an aware timestamp so timings compare correctly across machines.
        selected = self.selected_candidate_ids
        require(len(set(selected)) == len(selected), f"Selected candidate IDs must be unique, got {selected}.")
        require(self.created_at.utcoffset() is not None, "Selection created_at must be timezone-aware.")

    @property
    def is_no_selection(self) -> bool:
        """
        Tell whether the person explicitly selected nothing.
        """
        return not self.selected_candidate_ids


def select_candidates(iteration: Iteration, candidate_ids: Sequence[str]) -> SelectionEvent:
    """
    Record the person's selection after checking each ID belongs to the iteration and appears once.

    Args:
        iteration (Iteration): Iteration whose candidates were shown.
        candidate_ids (Sequence[str]): Selected candidate IDs, in the person's order.

    Returns:
        SelectionEvent: The recorded selection.
    """
    # An empty list is ambiguous, so choosing nothing must go through record_no_selection.
    if not candidate_ids:
        raise InvalidSelectionError("No candidate IDs given. Use record_no_selection for an explicit no-selection.")

    # Reject IDs that belong to another iteration.
    known = {candidate.id for candidate in iteration.result.candidates}
    unknown = [candidate_id for candidate_id in candidate_ids if candidate_id not in known]
    if unknown:
        raise InvalidSelectionError(f"Not candidates of iteration {iteration.id}: {', '.join(unknown)}.")

    # Reject repeats here so the error can name them, rather than the generic record check.
    repeated = sorted({candidate_id for candidate_id in candidate_ids if candidate_ids.count(candidate_id) > 1})
    if repeated:
        raise InvalidSelectionError(f"Candidates selected more than once: {', '.join(repeated)}. Select each once.")

    return SelectionEvent(iteration_id=iteration.id, selected_candidate_ids=tuple(candidate_ids))


def record_no_selection(iteration: Iteration) -> SelectionEvent:
    """
    Record that the person chose none of the iteration's candidates.
    """
    return SelectionEvent(iteration_id=iteration.id, selected_candidate_ids=())
