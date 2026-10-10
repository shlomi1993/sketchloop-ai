import hashlib

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, get_args

from sketchloop.domain import (ControlValue, GenerationRequest, GenerationResult, InvalidRecordError, IssueCode,
                               UnsupportedConfigurationError, ValidationIssue, require)

ControlKind: TypeAlias = Literal["int", "float", "bool", "choice"]

# Python types each control kind accepts, where float controls also take ints.
_ACCEPTED_TYPES_BY_KIND: dict[str, tuple[type, ...]] = {"int": (int,), "float": (int, float), "bool": (bool,), "choice": (str,)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlSpec:
    """
    Describes one backend control: its kind, bounds or choices, and default.
    """
    name: str
    kind: ControlKind
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    default: ControlValue | None = None

    def __post_init__(self) -> None:
        # Only numeric controls may have bounds, and the bounds must be ordered.
        require(self.kind in get_args(ControlKind), f"Control {self.name} kind must be int, float, bool, or choice.")
        numeric = self.kind in ("int", "float")
        has_bounds = self.minimum is not None or self.maximum is not None
        require(numeric or not has_bounds, f"Control {self.name} may have bounds only if it is an int or float.")
        bounds_ordered = self.minimum is None or self.maximum is None or self.minimum <= self.maximum
        require(bounds_ordered, f"Control {self.name} minimum must not exceed its maximum.")

        # Choice controls, and only they, need a list of distinct options.
        require((self.kind == "choice") == bool(self.choices), f"Control {self.name} needs choices exactly when a choice.")
        require(len(set(self.choices)) == len(self.choices), f"Control {self.name} choices must be unique.")

        # A default must pass the same checks as a value the person supplies.
        problem = None if self.default is None else _control_problem(self, self.default)
        if problem is not None:
            raise InvalidRecordError(f"Control {self.name} default is invalid. {problem[1]}")


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorCapabilities:
    """
    What a backend supports, used to validate requests before generating.
    """
    controls: tuple[ControlSpec, ...]
    max_candidates: int
    supports_negative_prompt: bool
    supports_seed: bool

    def __post_init__(self) -> None:
        # Duplicate names would make request validation ambiguous.
        names = [spec.name for spec in self.controls]
        require(len(set(names)) == len(names), f"Control names must be unique, got {names}.")

        # Exclude bool explicitly, since it is a subclass of int.
        valid_limit = isinstance(self.max_candidates, int) and not isinstance(self.max_candidates, bool)
        require(valid_limit and self.max_candidates >= 1, "max_candidates must be an int >= 1.")


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationOutput:
    """
    A generation result plus each candidate's image bytes, checked against its checksum.
    """
    result: GenerationResult
    payloads: Mapping[str, bytes]

    def __post_init__(self) -> None:
        # Freeze the payloads, then require exactly one payload per candidate image.
        object.__setattr__(self, "payloads", MappingProxyType(dict(self.payloads)))
        images = {candidate.image.path: candidate.image for candidate in self.result.candidates}
        require(set(self.payloads) == set(images), "Payload keys must equal the candidate image paths.")

        # Catch a backend whose bytes do not match the checksum it recorded.
        for path, payload in self.payloads.items():
            require(hashlib.sha256(payload).hexdigest() == images[path].sha256, f"Payload {path} fails its checksum.")


class Generator(Protocol):
    """
    Interface every generation backend implements, local or remote.
    """

    @property
    def capabilities(self) -> GeneratorCapabilities: ...

    def generate(self, request: GenerationRequest, sketch_payload: bytes) -> GenerationOutput: ...


def _control_problem(spec: ControlSpec, value: ControlValue) -> tuple[IssueCode, str] | None:
    # Bools pass isinstance checks for int and float, so reject them unless the control is a bool.
    if not isinstance(value, _ACCEPTED_TYPES_BY_KIND[spec.kind]) or (isinstance(value, bool) and spec.kind != "bool"):
        return "wrong_type", f"Expected {spec.kind}, got {type(value).__name__}."

    # Choice values must be one of the listed options.
    if spec.kind == "choice" and value not in spec.choices:
        return "invalid_choice", f"Use one of {', '.join(spec.choices)}."

    # Numeric values must fall within the declared bounds.
    if spec.minimum is not None and value < spec.minimum:
        return "out_of_range", f"{value} is below the minimum {spec.minimum}."
    if spec.maximum is not None and value > spec.maximum:
        return "out_of_range", f"{value} is above the maximum {spec.maximum}."

    return None


def request_issues(request: GenerationRequest, capabilities: GeneratorCapabilities) -> tuple[ValidationIssue, ...]:
    """
    List every way a request exceeds the backend's capabilities.

    Args:
        request (GenerationRequest): Request to check.
        capabilities (GeneratorCapabilities): What the backend supports.

    Returns:
        tuple[ValidationIssue, ...]: Issues found, empty when the request is valid.
    """
    # Check the candidate limit first.
    issues = []
    if request.n_candidates > capabilities.max_candidates:
        message = f"Requested {request.n_candidates}, the backend allows at most {capabilities.max_candidates}."
        issues.append(ValidationIssue(field="n_candidates", code="too_many_candidates", message=message))

    # Report optional features the backend lacks instead of silently ignoring them.
    if request.seed is not None and not capabilities.supports_seed:
        issues.append(ValidationIssue(field="seed", code="unsupported_feature", message="Omit the seed."))

    if request.guidance.negative_prompt is not None and not capabilities.supports_negative_prompt:
        message = "Negative prompts are unsupported. Omit it."
        issues.append(ValidationIssue(field="guidance.negative_prompt", code="unsupported_feature", message=message))

    # Check each control against its spec, in sorted order so the issue list is stable.
    specs = {spec.name: spec for spec in capabilities.controls}
    for name in sorted(request.guidance.controls):
        field = f"guidance.controls.{name}"

        # Name the supported controls so the person can correct a typo.
        if name not in specs:
            message = f"Unknown control. Supported: {', '.join(sorted(specs)) or 'none'}."
            issues.append(ValidationIssue(field=field, code="unknown_control", message=message))
            continue

        # Record a type, choice, or range problem for a known control.
        problem = _control_problem(specs[name], request.guidance.controls[name])
        if problem is not None:
            issues.append(ValidationIssue(field=field, code=problem[0], message=problem[1]))

    return tuple(issues)


def validate_request(request: GenerationRequest, capabilities: GeneratorCapabilities) -> None:
    """
    Reject a request the backend cannot honor, reporting all issues at once.

    Args:
        request (GenerationRequest): Request to check.
        capabilities (GeneratorCapabilities): What the backend supports.
    """
    # Raise one error listing every issue, so the person can fix them all in one pass.
    issues = request_issues(request, capabilities)
    if issues:
        raise UnsupportedConfigurationError(issues)
