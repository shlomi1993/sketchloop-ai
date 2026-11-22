"""
Session and round record types, with the checks that keep each record valid.
"""

import math
import re

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from sketchloop.domain import (
    GenerationRequest, ImageRef, Iteration, SelectionEvent, SketchLoopError, SketchSource, Unavailable, require,
    require_duration)
from sketchloop.experiments.prompts import (ORIGIN_NOT_RECORDED, PromptOrigin, PromptSuggestionRecord, check_prompt_origin,
                                            check_prompt_suggestions)
from sketchloop.preprocessing import PreprocessingStep

# Drawing records build on these records, so the loaded session names their type only for type checkers.
if TYPE_CHECKING:
    from sketchloop.experiments.drawings import DrawingRecord


SESSION_FILE: Final = "session.json"
ROUND_FILE: Final = "round.json"
MISSING_SOURCE_REASON: Final = "recorded before input sources were tracked"
MISSING_TOTAL_REASON: Final = "recorded before total round time was measured"
MISSING_HARDWARE_REASON: Final = "recorded before hardware was tracked"
SETUP_NOT_MEASURED_REASON: Final = "not measured by the caller"
SETUP_REUSED_REASON: Final = "generator reused from an earlier session"
MISSING_SETUP_REASON: Final = "recorded before generator setup was timed"
NOT_STREAMED_REASON: Final = "not a streamed webcam frame"
UNKNOWN_SOURCE_REASON: Final = "the sketch source is unknown"
FRAME_AGE_NOT_MEASURED_REASON: Final = "not measured by the caller"
MISSING_FRAME_AGE_REASON: Final = "recorded before frame age was tracked"

# Experiment names are portable folder names: ASCII letters, digits, "_", and "-", never a Windows device name.
EXPERIMENT_NAME_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
WINDOWS_DEVICE_NAMES: Final = frozenset({"con", "prn", "aux", "nul", *(f"com{digit}" for digit in range(1, 10)),
                                         *(f"lpt{digit}" for digit in range(1, 10))})
EXPERIMENT_NAME_RULE: Final = 'Experiment name must be 1 to 64 letters, digits, "_" or "-", starting with a letter or digit. Use a code such as participant1_car.'  # noqa: E501


class RoundStage(StrEnum):
    """
    A timed stage of a round, recorded as a stage timing and as the stage a failure happened in, in print order.
    """
    CAPTURE = "capture"
    PREPROCESSING = "preprocessing"
    MODEL_LOAD = "model_load"
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


class ExperimentNameError(SketchLoopError, ValueError):
    """
    An experiment name breaks the naming rules.
    """


class ExperimentNameTakenError(StoreError):
    """
    A folder of the experiment's name already exists in the runs folder.
    """


def is_experiment_name(name: object) -> bool:
    """
    Tell whether a name follows the experiment naming rules exactly, without surrounding spaces.
    """
    return isinstance(name, str) and bool(EXPERIMENT_NAME_PATTERN.fullmatch(name)) and name.lower() not in WINDOWS_DEVICE_NAMES


def check_experiment_name(name: str) -> str:
    """
    Return the experiment name with surrounding spaces stripped, or raise ExperimentNameError naming the rule.
    """
    stripped = name.strip() if isinstance(name, str) else name
    if not is_experiment_name(stripped):
        raise ExperimentNameError(EXPERIMENT_NAME_RULE)

    return stripped


def resolve_frame_age(source: SketchSource | Unavailable, frame_age_seconds: float | None) -> float | Unavailable:
    """
    Decide the frame age a record keeps, refusing an age for a source that is not a streamed webcam frame.

    Args:
        source (SketchSource | Unavailable): Where the raw sketch came from.
        frame_age_seconds (float | None): Seconds from the frame's arrival to its use, or None when not measured.

    Returns:
        float | Unavailable: The age for a webcam frame, otherwise why it is unavailable.
    """
    if source != SketchSource.WEBCAM:
        require(frame_age_seconds is None, f"Only a webcam round has a frame age, got source {source!r}.")
        return Unavailable(UNKNOWN_SOURCE_REASON if isinstance(source, Unavailable) else NOT_STREAMED_REASON)

    if frame_age_seconds is None:
        return Unavailable(FRAME_AGE_NOT_MEASURED_REASON)

    require_duration(frame_age_seconds, "Frame age")
    return frame_age_seconds


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
class BenchmarkPlan:
    """
    How many warm-up and measured rounds a benchmark session runs.
    """
    warmup_rounds: int  # Rounds run first and left out of the statistics, such as the one that loads the model
    measured_rounds: int  # Rounds the statistics cover, after the warm-up

    def __post_init__(self) -> None:
        # Exclude bool explicitly, since it is a subclass of int.
        counts = (self.warmup_rounds, self.measured_rounds)
        are_ints = all(isinstance(count, int) and not isinstance(count, bool) for count in counts)
        valid_counts = are_ints and self.warmup_rounds >= 0 and self.measured_rounds >= 1
        require(valid_counts, f"Benchmark needs 0+ warm-up and 1+ measured rounds, got {counts}.")


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
    Contents of session.json: identity, creation time, environment, generator setup, rerun link, and benchmark plan.
    """
    schema_version: int  # Record format version
    session_id: str  # Session folder name
    created_at: datetime  # Timezone-aware creation time
    environment: Environment  # Code and library versions
    generator_setup_seconds: float | Unavailable  # Time to create the session's first generator, before round 1
    rerun_of: RerunLink | None  # Original round for a rerun session, otherwise None
    benchmark: BenchmarkPlan | None  # Warm-up and measured round counts for a benchmark session, otherwise None
    name: str | None = None  # Experiment name the experimenter entered, also the folder name, None for a time-based folder

    def __post_init__(self) -> None:
        require(self.created_at.utcoffset() is not None, "Session created_at must be timezone-aware.")
        valid_name = self.name is None or (is_experiment_name(self.name) and self.name == self.session_id)
        require(valid_name, f"Experiment name {self.name!r} must follow the naming rules and equal the session ID.")
        if not isinstance(self.generator_setup_seconds, Unavailable):
            require_duration(self.generator_setup_seconds, "Generator setup time")


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
    frame_age_seconds: float | Unavailable  # Seconds from the webcam frame's arrival to the start of the round
    use_raw: bool  # Whether generation used the raw sketch and skipped preprocessing
    rotation: int  # Clockwise rotation in degrees applied before preprocessing
    steps: tuple[PreprocessingStep, ...]  # Preprocessing steps applied, empty with --raw
    request: GenerationRequest | None  # What was asked, None when preprocessing failed
    iteration: Iteration | None  # Request, result, and parent link, None when the round failed
    timings_seconds: Mapping[str, float]  # Duration by stage, holding only stages that ran and were timed
    total_seconds: float | Unavailable  # Capture time plus wall time of run_round until the files were saved
    evaluator: str | None  # Evaluator class name, None when the round failed
    score_details: Mapping[str, str]  # Model, revision, and meaning of the scores, empty when the evaluator gave none
    scores: Mapping[str, float]  # Score by candidate ID
    selection: SelectionEvent | None  # The person's choice, None when not recorded
    failure: RoundFailure | None  # Why the round failed, None when complete
    prompt_suggestions: tuple[PromptSuggestionRecord, ...] | None = None  # Shown outside any drawing since the previous round
    prompt_origin: PromptOrigin | Unavailable = Unavailable(ORIGIN_NOT_RECORDED)  # How the round's prompt came about

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

        # Durations are finite seconds of at least 0, and only a streamed webcam frame has an age.
        for stage, seconds in self.timings_seconds.items():
            require_duration(seconds, f"Stage {stage} time")
        if not isinstance(self.total_seconds, Unavailable):
            require_duration(self.total_seconds, "Total round time")
        if not isinstance(self.frame_age_seconds, Unavailable):
            require_duration(self.frame_age_seconds, "Frame age")
            require(self.source == SketchSource.WEBCAM, f"Only a webcam round has a frame age, got source {self.source!r}.")

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
        object.__setattr__(self, "score_details", MappingProxyType(dict(self.score_details)))
        self._check_prompts()

    def _check_prompts(self) -> None:
        # Suggestions shown outside a drawing are numbered in order and follow no capture.
        if self.prompt_suggestions is not None:
            object.__setattr__(self, "prompt_suggestions", tuple(self.prompt_suggestions))
        suggestions = self.prompt_suggestions or ()
        check_prompt_suggestions(suggestions, f"Round {self.number}")
        outside_drawing = all(item.after_capture is None for item in suggestions)
        require(outside_drawing, f"Round {self.number} prompt suggestions were shown outside drawings, so follow no capture.")

        # The origin may name a suggestion of an earlier round or a drawing, so only one in this round's list is checked here.
        origin = self.prompt_origin
        valid_origin = isinstance(origin, PromptOrigin | Unavailable)
        require(valid_origin, f"Round {self.number} prompt_origin must be an origin or unavailable, got {origin!r}.")
        if isinstance(origin, PromptOrigin) and origin.prompt_suggestion in {item.id for item in suggestions}:
            prompt = self.request.guidance.prompt if self.request else None
            check_prompt_origin(origin, prompt, self.started_at, suggestions, f"Round {self.number} prompt origin")


@dataclass(frozen=True, slots=True, kw_only=True)
class SavedSession:
    """
    A loaded session folder with its records and any artifact problems found.
    """
    folder: Path  # Session folder
    session: SessionRecord  # Contents of session.json
    rounds: tuple[RoundRecord, ...]  # Rounds with a round.json, in order
    problems: tuple[str, ...]  # Missing or changed files and interrupted rounds or drawings
    checked_files: int  # Number of files whose checksum matched
    drawings: tuple["DrawingRecord", ...] = ()  # Drawings with a drawing.json, in order, none in sessions saved before T09


@dataclass(frozen=True, slots=True, kw_only=True)
class RerunComparison:
    """
    How a rerun round compares with the original round.
    """
    identical_candidates: tuple[int, ...]  # 1-based numbers of candidates with identical bytes
    candidate_count: int  # Number of candidates compared
    differences: tuple[str, ...]  # One line per changed backend, setting, or environment value
