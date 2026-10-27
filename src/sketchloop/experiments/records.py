import math

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final

from sketchloop.domain import (
    GenerationRequest, ImageRef, Iteration, SelectionEvent, SketchLoopError, SketchSource, Unavailable, require)
from sketchloop.preprocessing import PreprocessingStep


SESSION_FILE: Final = "session.json"
ROUND_FILE: Final = "round.json"
MISSING_SOURCE_REASON: Final = "recorded before input sources were tracked"
MISSING_HARDWARE_REASON: Final = "recorded before hardware was tracked"


class RoundStage(StrEnum):
    """
    A timed stage of a round, recorded as a stage timing and as the stage a failure happened in.
    """
    PREPROCESSING = "preprocessing"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    SAVING = "saving"


class RoundStatus(StrEnum):
    """
    Whether a round produced candidates or failed.
    """
    COMPLETE = "complete"
    FAILED = "failed"


class ExperimentRecordError(SketchLoopError, ValueError):
    """
    A saved session record is missing, malformed, from another schema version, or inconsistent.
    """


class StoreError(SketchLoopError):
    """
    A store could not save a session record, a round record, or a file.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class RerunLink:
    """
    The original round a rerun session repeats.
    """
    session_id: str  # Original session folder name
    round_number: int  # Original round number
    iteration_id: str  # Original round's iteration ID


@dataclass(frozen=True, slots=True, kw_only=True)
class Hardware:
    """
    The processor, logical core count, and memory of the machine a session ran on, never its name or serial number.
    """
    cpu: str | Unavailable  # Processor model name, such as "Apple M2 Pro"
    logical_cores: int | Unavailable  # Logical CPU count the operating system reports
    memory_bytes: int | Unavailable  # Total physical memory

    def __post_init__(self) -> None:
        # Each fact is a real value or unavailable with a reason, never an empty name or an impossible count.
        valid_cpu = isinstance(self.cpu, Unavailable) or (isinstance(self.cpu, str) and bool(self.cpu.strip()))
        require(valid_cpu, f"Hardware cpu must be non-empty text or unavailable, got {self.cpu!r}.")
        for name in ("logical_cores", "memory_bytes"):
            value = getattr(self, name)
            is_count = isinstance(value, int) and not isinstance(value, bool) and value >= 1
            require(is_count or isinstance(value, Unavailable), f"Hardware {name} must be an int >= 1, got {value!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Environment:
    """
    Python, platform, package versions, and hardware a session ran with, including sketchloop-ai itself.
    """
    py_version: str  # Python version, such as 3.12.7
    platform: str  # Operating system, release, and machine, never host or user names
    packages: Mapping[str, str | Unavailable]  # Version by distribution name
    hardware: Hardware | Unavailable  # CPU, cores, and memory of the machine

    def __post_init__(self) -> None:
        # Copy into a read-only view so the recorded versions cannot change after the fact.
        object.__setattr__(self, "packages", MappingProxyType(dict(self.packages)))


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionRecord:
    """
    Contents of session.json: identity, creation time, environment, and the round a rerun repeats.
    """
    schema_version: int  # Record format version
    session_id: str  # Session folder name
    created_at: datetime  # Timezone-aware creation time
    environment: Environment  # Code and library versions
    rerun_of: RerunLink | None  # Original round for a rerun session, otherwise None

    def __post_init__(self) -> None:
        require(self.created_at.utcoffset() is not None, "Session created_at must be timezone-aware.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundFailure:
    """
    Where and how a round failed.
    """
    stage: RoundStage  # Stage that raised the error
    error: str  # Error class name
    message: str  # First line of the error message

    def __post_init__(self) -> None:
        # Hold a known stage as the enum, so a stage loaded from JSON text matches a recorded one.
        require(self.stage in list(RoundStage), f"Failure stage must be one of {', '.join(RoundStage)}, got {self.stage!r}.")
        object.__setattr__(self, "stage", RoundStage(self.stage))


def require_known_source(source: object) -> None:
    """
    Refuse a round source that is neither a known value nor Unavailable.
    """
    valid_source = isinstance(source, Unavailable) or source in list(SketchSource)
    require(valid_source, f"Round source must be file, camera, webcam, or upload, got {source!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundRecord:
    """
    Contents of round.json: input, preprocessing, request, result, timings, scores, selection, and any failure.
    """
    schema_version: int  # Record format version
    number: int  # Round number, starting at 1
    status: RoundStatus  # complete, or failed with a failure record
    started_at: datetime  # Timezone-aware start time
    raw_sketch: ImageRef  # Raw sketch as loaded or captured
    source: SketchSource | Unavailable  # Where the raw sketch bytes came from, unavailable in older records
    use_raw: bool  # Whether generation used the raw sketch and skipped preprocessing
    rotation: int  # Clockwise rotation in degrees applied before preprocessing
    steps: tuple[PreprocessingStep, ...]  # Preprocessing steps applied, empty with --raw
    request: GenerationRequest | None  # What was asked, None when preprocessing failed
    iteration: Iteration | None  # Request, result, and parent link, None when the round failed
    timings_seconds: Mapping[str, float]  # Duration by stage
    evaluator: str | None  # Evaluator class name, None when the round failed
    scores: Mapping[str, float]  # Score by candidate ID
    selection: SelectionEvent | None  # The person's choice, None when not recorded
    failure: RoundFailure | None  # Why the round failed, None when complete

    def __post_init__(self) -> None:
        # A complete round has an iteration built from its request, and a failed one has only the failure.
        complete = self.iteration is not None and self.failure is None
        require(self.status in list(RoundStatus), f"Round status must be complete or failed, got {self.status!r}.")
        object.__setattr__(self, "status", RoundStatus(self.status))
        require((self.status == RoundStatus.COMPLETE) == complete, "Round status must be complete exactly when it has a result.")
        require(not complete or self.request is self.iteration.request, "A complete round's request must be its iteration's.")
        require(self.started_at.utcoffset() is not None, "Round started_at must be timezone-aware.")
        require_known_source(self.source)
        require(all(math.isfinite(score) for score in self.scores.values()), "Scores must be finite numbers.")

        # A selection must name this round's iteration and candidates.
        if self.selection is not None:
            candidate_ids = {candidate.id for candidate in self.iteration.result.candidates} if self.iteration else set()
            belongs = self.iteration is not None and self.selection.iteration_id == self.iteration.id
            require(belongs and set(self.selection.selected_candidate_ids) <= candidate_ids, "Selection is outside its round.")

        # Hold a known source as the enum, so a source loaded from JSON text compares and prints like a recorded one.
        if not isinstance(self.source, Unavailable):
            object.__setattr__(self, "source", SketchSource(self.source))

        # Copy into read-only views so the recorded values cannot change after the fact.
        object.__setattr__(self, "timings_seconds", MappingProxyType(dict(self.timings_seconds)))
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@dataclass(frozen=True, slots=True, kw_only=True)
class SavedSession:
    """
    A loaded session folder with its records and any artifact problems found.
    """
    folder: Path  # Session folder
    session: SessionRecord  # Contents of session.json
    rounds: tuple[RoundRecord, ...]  # Rounds with a round.json, in order
    problems: tuple[str, ...]  # Missing or changed files and interrupted rounds
    checked_files: int  # Number of files whose checksum matched


@dataclass(frozen=True, slots=True, kw_only=True)
class RerunComparison:
    """
    How a rerun round compares with the original round.
    """
    identical_candidates: tuple[int, ...]  # 1-based numbers of candidates with identical bytes
    candidate_count: int  # Number of candidates compared
    differences: tuple[str, ...]  # One line per changed backend, setting, or environment value
