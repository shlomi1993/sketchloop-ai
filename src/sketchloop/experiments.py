import hashlib
import json
import math
import os
import platform
import re

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Final, TypeVar

from sketchloop.domain import (SCHEMA_VERSION, BackendIdentity, Candidate, ControlValue, EffectiveSettings,
                               GenerationRequest, GenerationResult, Guidance, ImageRef, Iteration, SelectionEvent,
                               SketchLoopError, SketchSource, Unavailable, first_line, require)
from sketchloop.generation import GeneratorCapabilities
from sketchloop.preprocessing import PreprocessingStep


SESSION_FILE: Final = "session.json"
ROUND_FILE: Final = "round.json"
ROUND_FOLDER_PATTERN: Final = re.compile(r"round-([1-9][0-9]*)")
MISSING_SOURCE_REASON: Final = "recorded before input sources were tracked"

# Failure messages are split into quoted strings, kept whole since paths may contain spaces, and runs of other text.
MESSAGE_TOKEN_PATTERN: Final = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")

# Punctuation that can surround an unquoted path, such as a closing parenthesis or a trailing comma.
PATH_PUNCTUATION: Final = "()[]{},;:."

# Distributions whose versions can change generated output or how records are read.
TRACKED_PACKAGES: Final = (
    "sketchloop-ai",
    "numpy",
    "opencv-python",
    "rich",
    "diffusers",
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "huggingface_hub",
)

RecordT = TypeVar("RecordT")


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
class Environment:
    """
    Python, platform, and package versions a session ran with, including sketchloop-ai itself.
    """
    py_version: str  # Python version, such as 3.12.7
    platform: str  # Operating system, release, and machine, never host or user names
    packages: Mapping[str, str | Unavailable]  # Version by distribution name

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


def read_package_version(name: str) -> str | Unavailable:
    """
    Return an installed distribution's version, or why it is unavailable.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return Unavailable("not installed")


def collect_environment() -> Environment:
    """
    Record the Python, platform, and tracked package versions of this process.

    Returns:
        Environment: The current environment, with Unavailable where a version cannot be read.
    """
    system = f"{platform.system()} {platform.release()} {platform.machine()}"
    packages = {name: read_package_version(name) for name in TRACKED_PACKAGES}
    return Environment(py_version=platform.python_version(), platform=system, packages=packages)


def is_absolute_path(text: str) -> bool:
    """
    Tell whether text is an absolute POSIX or Windows path, including a network share.
    """
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def redact_token(match: re.Match[str]) -> str:
    """
    Replace a message token that is an absolute path with a placeholder, keeping its quotes or punctuation.

    Args:
        match (re.Match[str]): A quoted string or a run of non-space text from the message.

    Returns:
        str: The token, with an absolute path replaced by <path>.
    """
    # A quoted token is a path when its whole content is one.
    token = match.group()
    if len(token) > 1 and token[0] in "'\"" and token[-1] == token[0]:
        return f"{token[0]}<path>{token[0]}" if is_absolute_path(token[1:-1]) else token

    # An unquoted token is a path once the punctuation around it is set aside.
    core = token.strip(PATH_PUNCTUATION)
    return token.replace(core, "<path>", 1) if core and is_absolute_path(core) else token


def redact_paths(text: str) -> str:
    """
    Replace absolute paths, which can contain the user name, with a placeholder before the text is saved.
    """
    return MESSAGE_TOKEN_PATTERN.sub(redact_token, text)


def describe_failure(stage: RoundStage, error: BaseException) -> RoundFailure:
    """
    Summarize an exception as a round failure with its class name and the first line of its message, paths redacted.
    """
    return RoundFailure(stage=stage, error=type(error).__name__, message=redact_paths(first_line(error)))


def encode_value(value: object) -> object:
    """
    Encode an Unavailable value as {"unavailable": reason} and leave other values unchanged.
    """
    return {"unavailable": value.reason} if isinstance(value, Unavailable) else value


def decode_value(value: object) -> object:
    """
    Turn {"unavailable": reason} back into Unavailable and leave other values unchanged.
    """
    return Unavailable(value["unavailable"]) if isinstance(value, dict) else value


def parse_time(text: object) -> datetime:
    """
    Parse an ISO 8601 time, rejecting values that are not text.
    """
    require(isinstance(text, str), f"Time must be ISO 8601 text, got {text!r}.")
    return datetime.fromisoformat(text)


def image_from_dict(data: Mapping[str, object]) -> ImageRef:
    """
    Rebuild an image reference, refusing paths that leave the round folder.

    Args:
        data (Mapping[str, object]): Serialized image reference.

    Returns:
        ImageRef: The validated image reference.
    """
    # Name traversal explicitly, since it may be an attempt to read files outside the session.
    path = data["path"]
    if isinstance(path, str) and (path.startswith(("/", "\\")) or ".." in re.split(r"[/\\]", path)):
        raise ExperimentRecordError(f"references {path} outside its round folder. Refusing to load it.")

    return ImageRef(**data)


def request_to_dict(request: GenerationRequest) -> dict[str, object]:
    """
    Serialize a generation request.
    """
    guidance = request.guidance
    return {
        "sketch": asdict(request.sketch),
        "guidance": {
            "prompt": guidance.prompt,
            "negative_prompt": guidance.negative_prompt,
            "controls": dict(guidance.controls),
        },
        "n_candidates": request.n_candidates,
        "seed": request.seed,
    }


def request_from_dict(data: Mapping[str, object]) -> GenerationRequest:
    """
    Rebuild a generation request through its validating constructors.
    """
    guidance_data = data["guidance"]
    guidance = Guidance(prompt=guidance_data["prompt"], negative_prompt=guidance_data["negative_prompt"],
                        controls=guidance_data["controls"])
    return GenerationRequest(sketch=image_from_dict(data["sketch"]), guidance=guidance,
                             n_candidates=data["n_candidates"], seed=data["seed"])


def result_to_dict(result: GenerationResult) -> dict[str, object]:
    """
    Serialize a generation result: backend identity, effective settings, and candidates.

    Args:
        result (GenerationResult): Result to serialize.

    Returns:
        dict[str, object]: JSON-ready result.
    """
    backend = result.backend
    effective = result.effective
    candidates = [
        {
            "id": candidate.id,
            "index": candidate.index,
            "seed": encode_value(candidate.seed),
            "image": asdict(candidate.image),
        }
        for candidate in result.candidates
    ]
    return {
        "backend": {
            "adapter": backend.adapter,
            "adapter_version": backend.adapter_version,
            "execution": backend.execution,
            "model_id": encode_value(backend.model_id),
        },
        "effective": {
            "prompt": encode_value(effective.prompt),
            "negative_prompt": encode_value(effective.negative_prompt),
            "controls": {name: encode_value(value) for name, value in effective.controls.items()},
        },
        "candidates": candidates,
    }


def result_from_dict(data: Mapping[str, object]) -> GenerationResult:
    """
    Rebuild a generation result through its validating constructors.

    Args:
        data (Mapping[str, object]): Serialized result.

    Returns:
        GenerationResult: The validated result.
    """
    backend_data, effective_data = data["backend"], data["effective"]
    backend = BackendIdentity(adapter=backend_data["adapter"], adapter_version=backend_data["adapter_version"],
                              execution=backend_data["execution"], model_id=decode_value(backend_data["model_id"]))
    controls = {name: decode_value(value) for name, value in effective_data["controls"].items()}
    effective = EffectiveSettings(prompt=decode_value(effective_data["prompt"]),
                                  negative_prompt=decode_value(effective_data["negative_prompt"]), controls=controls)
    candidates = tuple(Candidate(id=item["id"], index=item["index"], seed=decode_value(item["seed"]),
                                 image=image_from_dict(item["image"])) for item in data["candidates"])
    return GenerationResult(candidates=candidates, backend=backend, effective=effective)


def round_to_dict(record: RoundRecord) -> dict[str, object]:
    """
    Serialize a round record in the fixed key order of round.json.

    Args:
        record (RoundRecord): Round to serialize.

    Returns:
        dict[str, object]: JSON-ready round record.
    """
    iteration, selection, failure = record.iteration, record.selection, record.failure
    steps = [{"name": step.name, "params": dict(step.params)} for step in record.steps]
    evaluation = {"evaluator": record.evaluator, "scores": dict(record.scores)}
    selection_data = {
        "id": selection.id,
        "candidate_ids": list(selection.selected_candidate_ids),
        "created_at": selection.created_at.isoformat(),
    } if selection else None
    failure_data = asdict(failure) if failure else None
    return {
        "schema_version": record.schema_version,
        "round": record.number,
        "status": record.status,
        "started_at": record.started_at.isoformat(),
        "input": {
            "source": encode_value(record.source),
            "raw_sketch": asdict(record.raw_sketch),
            "use_raw": record.use_raw,
            "rotation": record.rotation,
            "preprocessing": steps,
        },
        "iteration_id": iteration.id if iteration else None,
        "parent_id": iteration.parent_id if iteration else None,
        "request": request_to_dict(record.request) if record.request else None,
        "result": result_to_dict(iteration.result) if iteration else None,
        "timings_seconds": dict(record.timings_seconds),
        "evaluation": evaluation if iteration else None,
        "selection": selection_data,
        "failure": failure_data,
    }


def check_schema_version(data: Mapping[str, object]) -> None:
    """
    Refuse a record written in a schema version this code cannot read.
    """
    version = data["schema_version"]
    if version != SCHEMA_VERSION:
        raise ExperimentRecordError(f"has schema version {version}, but this sketchloop reads {SCHEMA_VERSION}. Upgrade it.")


def round_from_dict(data: Mapping[str, object]) -> RoundRecord:
    """
    Rebuild a round record through the validating domain constructors.

    Args:
        data (Mapping[str, object]): Contents of round.json.

    Returns:
        RoundRecord: The validated round record.
    """
    check_schema_version(data)

    # Rebuild the request, and the iteration around it when the round completed.
    input_data, request_data, result_data = data["input"], data["request"], data["result"]
    request = request_from_dict(request_data) if request_data is not None else None
    iteration = None
    if result_data is not None:
        iteration = Iteration(id=data["iteration_id"], parent_id=data["parent_id"], request=request,
                              result=result_from_dict(result_data))

    # Rebuild the selection against this round's iteration, and the failure if the round failed.
    selection_data, failure_data, evaluation = data["selection"], data["failure"], data["evaluation"] or {}
    selection = None
    if selection_data is not None:
        selection = SelectionEvent(id=selection_data["id"], iteration_id=iteration.id if iteration else "",
                                   selected_candidate_ids=tuple(selection_data["candidate_ids"]),
                                   created_at=parse_time(selection_data["created_at"]))
    failure = RoundFailure(**failure_data) if failure_data is not None else None
    steps = tuple(PreprocessingStep(name=step["name"], params=step["params"]) for step in input_data["preprocessing"])

    # Records written before sources were tracked have no source, which is unknown rather than a guessed default.
    source = decode_value(input_data["source"]) if "source" in input_data else Unavailable(MISSING_SOURCE_REASON)
    return RoundRecord(schema_version=data["schema_version"], number=data["round"], status=data["status"],
                       started_at=parse_time(data["started_at"]), raw_sketch=image_from_dict(input_data["raw_sketch"]),
                       source=source, use_raw=input_data["use_raw"], rotation=input_data["rotation"], steps=steps,
                       request=request, iteration=iteration, timings_seconds=data["timings_seconds"],
                       evaluator=evaluation.get("evaluator"), scores=evaluation.get("scores", {}), selection=selection,
                       failure=failure)


def session_to_dict(record: SessionRecord) -> dict[str, object]:
    """
    Serialize a session record in the fixed key order of session.json.

    Args:
        record (SessionRecord): Session to serialize.

    Returns:
        dict[str, object]: JSON-ready session record.
    """
    environment, link = record.environment, record.rerun_of
    packages = {name: encode_value(version) for name, version in environment.packages.items()}
    link_data = {
        "session_id": link.session_id,
        "round": link.round_number,
        "iteration_id": link.iteration_id,
    } if link else None
    return {
        "schema_version": record.schema_version,
        "session_id": record.session_id,
        "created_at": record.created_at.isoformat(),
        "environment": {
            "python": environment.py_version,
            "platform": environment.platform,
            "packages": packages,
        },
        "rerun_of": link_data,
    }


def session_from_dict(data: Mapping[str, object]) -> SessionRecord:
    """
    Rebuild a session record from the contents of session.json.

    Args:
        data (Mapping[str, object]): Contents of session.json.

    Returns:
        SessionRecord: The validated session record.
    """
    check_schema_version(data)
    environment_data, link_data = data["environment"], data["rerun_of"]
    packages = {name: decode_value(version) for name, version in environment_data["packages"].items()}
    environment = Environment(py_version=environment_data["python"], platform=environment_data["platform"], packages=packages)
    link = None
    if link_data is not None:
        link = RerunLink(session_id=link_data["session_id"], round_number=link_data["round"],
                         iteration_id=link_data["iteration_id"])
    return SessionRecord(schema_version=data["schema_version"], session_id=data["session_id"],
                         created_at=parse_time(data["created_at"]), environment=environment, rerun_of=link)


def write_json_atomic(path: Path, data: Mapping[str, object]) -> None:
    """
    Write JSON through a temporary file in the same folder, so a crash never leaves a half-written record.

    Args:
        path (Path): Target file.
        data (Mapping[str, object]): JSON-ready data.
    """
    # Flush to disk before the rename, so the replaced file is complete even after a power loss.
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)
    except BaseException:
        # Remove the partial temporary file, so a failed save leaves only the previous record.
        temporary.unlink(missing_ok=True)
        raise


def write_files(folder: Path, files: Mapping[str, bytes]) -> None:
    """
    Write each payload to its relative path under the folder, creating subfolders as needed.

    Args:
        folder (Path): Folder the relative paths resolve against.
        files (Mapping[str, bytes]): Payload by relative POSIX path.
    """
    # Create each file's subfolder before writing it.
    for path, payload in files.items():
        target_file = folder / path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(payload)


class ExperimentStore(ABC):
    """
    Write side of experiment storage: where sessions live, and saving session and round records with their files.
    """

    @property
    @abstractmethod
    def root(self) -> Path | None:
        """
        Return the local folder that holds all saved sessions, or None when files are not kept on disk.
        """
        ...

    @abstractmethod
    def locate_session(self, session_id: str) -> Path | None:
        """
        Return the session's folder, which may not exist yet, or None when files are not kept on disk.
        """
        ...

    @abstractmethod
    def locate_round(self, session_id: str, number: int) -> Path | None:
        """
        Return the round's folder, or None when files are not kept on disk.
        """
        ...

    @abstractmethod
    def save_session(self, record: SessionRecord) -> None:
        """
        Create the session and save its record, raising StoreError when it cannot.
        """
        ...

    @abstractmethod
    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        Save the files, then the record, replacing an earlier record of the same round number.

        Args:
            session_id (str): Session the round belongs to.
            record (RoundRecord): Round record to save.
            files (Mapping[str, bytes]): Payload by relative POSIX path, empty to replace only the record.
        """
        ...


class FolderStore(ExperimentStore):
    """
    Store that keeps each session in a folder under the runs folder, with session.json and round-<n>/round.json.
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir

    @property
    def root(self) -> Path | None:
        """
        Return the runs folder.
        """
        return self.runs_dir

    def locate_session(self, session_id: str) -> Path | None:
        """
        Return the session's folder under the runs folder.
        """
        return self.runs_dir / session_id

    def locate_round(self, session_id: str, number: int) -> Path | None:
        """
        Return the round's folder inside its session folder.
        """
        return self.runs_dir / session_id / f"round-{number}"

    @contextmanager
    def _report_failure(self, target: Path) -> Iterator[None]:
        # Name the file relative to the runs folder, which keeps user names out of the message.
        try:
            yield
        except OSError as error:
            label = target.relative_to(self.runs_dir).as_posix()
            reason = error.strerror or type(error).__name__
            message = f"Cannot save {label}: {reason}. Free disk space or check write access to the runs folder, then retry."
            raise StoreError(message) from error
        except (TypeError, ValueError) as error:
            label = target.relative_to(self.runs_dir).as_posix()
            raise StoreError(f"Cannot save {label}: {first_line(error)}. Check the record values, then retry.") from error

    def save_session(self, record: SessionRecord) -> None:
        """
        Create the session folder and atomically write session.json.

        Args:
            record (SessionRecord): Session record to save.
        """
        folder = self.runs_dir / record.session_id
        with self._report_failure(folder / SESSION_FILE):
            folder.mkdir(parents=True, exist_ok=True)
            write_json_atomic(folder / SESSION_FILE, session_to_dict(record))

    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        Write the round's files, then atomically write round.json, so the record never names an unsaved file.

        Args:
            session_id (str): Session the round belongs to.
            record (RoundRecord): Round record to save.
            files (Mapping[str, bytes]): Payload by relative POSIX path, empty to replace only the record.
        """
        # Write the files one at a time, so a failure names the file that could not be saved.
        round_folder = self.runs_dir / session_id / f"round-{record.number}"
        for path, payload in files.items():
            with self._report_failure(round_folder / path):
                write_files(round_folder, {path: payload})

        # Write the record last.
        with self._report_failure(round_folder / ROUND_FILE):
            round_folder.mkdir(parents=True, exist_ok=True)
            write_json_atomic(round_folder / ROUND_FILE, round_to_dict(record))


class MemoryStore(ExperimentStore):
    """
    Store that keeps records and file bytes in process memory, for tests and library callers that need no files.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, SessionRecord] = {}
        self.rounds: dict[tuple[str, int], RoundRecord] = {}
        self.files: dict[tuple[str, int, str], bytes] = {}

    @property
    def root(self) -> Path | None:
        """
        Return None, since no files are kept on disk.
        """
        return None

    def locate_session(self, session_id: str) -> Path | None:
        """
        Return None, since sessions have no folder.
        """
        return None

    def locate_round(self, session_id: str, number: int) -> Path | None:
        """
        Return None, since rounds have no folder.
        """
        return None

    def _check_encodable(self, label: str, encode: Callable[[], Mapping[str, object]]) -> None:
        # Encode as the folder store would, discarding the text, so a record it could not write fails here too.
        try:
            json.dumps(encode(), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise StoreError(f"Cannot keep {label}: {first_line(error)}. Check the record values, then retry.") from error

    def save_session(self, record: SessionRecord) -> None:
        """
        Keep the session record after checking that it encodes as JSON.

        Args:
            record (SessionRecord): Session record to keep.
        """
        self._check_encodable(f"{record.session_id}/{SESSION_FILE}", lambda: session_to_dict(record))
        self.sessions[record.session_id] = record

    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        Keep the round's files, then its record, replacing an earlier record of the same round number.

        Args:
            session_id (str): Session the round belongs to.
            record (RoundRecord): Round record to keep.
            files (Mapping[str, bytes]): Payload by relative POSIX path, empty to replace only the record.
        """
        self.files |= {(session_id, record.number, path): payload for path, payload in files.items()}
        self._check_encodable(f"{session_id}/round-{record.number}/{ROUND_FILE}", lambda: round_to_dict(record))
        self.rounds[(session_id, record.number)] = record


def read_json(path: Path, label: str) -> Mapping[str, object]:
    """
    Read a JSON object from a record file.

    Args:
        path (Path): Record file.
        label (str): Name used in error messages, relative to the session folder.

    Returns:
        Mapping[str, object]: The parsed object.
    """
    # Name the decoding error, which includes the line of a syntax error, so the person can find it.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExperimentRecordError(f"{label} is not valid JSON: {error}. Restore the file or inspect it by hand.") from None

    if not isinstance(data, dict):
        raise ExperimentRecordError(f"{label}: the top level is invalid: expected an object.")

    return data


def convert_record(label: str, data: Mapping[str, object], convert: Callable[[Mapping[str, object]], RecordT]) -> RecordT:
    """
    Convert record data, turning any validation fault into one ExperimentRecordError that names the file.

    Args:
        label (str): File name used in error messages.
        data (Mapping[str, object]): Parsed record.
        convert (Callable[[Mapping[str, object]], RecordT]): From-dict function.

    Returns:
        RecordT: The converted record.
    """
    try:
        return convert(data)
    except ExperimentRecordError as error:
        raise ExperimentRecordError(f"{label} {error}") from None
    except KeyError as error:
        raise ExperimentRecordError(f"{label}: {error.args[0]} is invalid: it is missing.") from None
    except (AttributeError, TypeError, ValueError) as error:
        raise ExperimentRecordError(f"{label}: a field is invalid: {error}") from None


def list_round_folders(folder: Path) -> list[tuple[int, Path]]:
    """
    List round-<n> folders by number, refusing gaps in the numbering.

    Args:
        folder (Path): Session folder.

    Returns:
        list[tuple[int, Path]]: Round numbers and folders, in order.
    """
    rounds = []
    for path in folder.iterdir():
        match = ROUND_FOLDER_PATTERN.fullmatch(path.name)
        if match and path.is_dir():
            rounds.append((int(match.group(1)), path))

    # Require round-1 to round-N, since a missing folder means the records were edited or mixed.
    rounds.sort()
    numbers = [number for number, _ in rounds]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ExperimentRecordError(f"Round folders {numbers} have a gap. The records were edited or mixed.")

    return rounds


def list_round_images(record: RoundRecord) -> list[ImageRef]:
    """
    List the raw sketch, the generation sketch, and the candidates a round recorded, without duplicates.
    """
    images = [record.raw_sketch] + ([record.request.sketch] if record.request else [])
    images += [candidate.image for candidate in record.iteration.result.candidates] if record.iteration else []
    return list({image.path: image for image in images}.values())


def verify_round_files(record: RoundRecord, round_folder: Path) -> tuple[list[str], int]:
    """
    Compare each recorded image file with its checksum.

    Args:
        record (RoundRecord): Round whose files to check.
        round_folder (Path): Folder holding the round's files.

    Returns:
        tuple[list[str], int]: Problems found, and the number of files that matched.
    """
    problems, matched = [], 0
    resolved_folder = round_folder.resolve()
    for image in list_round_images(record):
        # Refuse a link that resolves outside the round folder, such as a planted symlink.
        target = round_folder / image.path
        if not target.resolve().is_relative_to(resolved_folder):
            raise ExperimentRecordError(f"{round_folder.name}/{ROUND_FILE} references {image.path} outside its round folder. Refusing to load it.")  # noqa: E501

        # List a missing or changed file as a problem, so the rest of the record can still be shown.
        label = f"{round_folder.name}/{image.path}"
        if not target.is_file():
            problems.append(f"{label} is missing.")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != image.sha256:
            problems.append(f"{label} does not match its recorded checksum.")
        else:
            matched += 1

    return problems, matched


def load_rounds(folder: Path) -> tuple[list[RoundRecord], list[str], int]:
    """
    Load and check every round of a session folder.

    Args:
        folder (Path): Session folder.

    Returns:
        tuple[list[RoundRecord], list[str], int]: The rounds, the artifact problems, and the number of matching files.
    """
    rounds, problems, matched = [], [], 0
    parent_id = None
    for number, round_folder in list_round_folders(folder):
        # A round folder without round.json was interrupted before its record was written.
        label = f"{round_folder.name}/{ROUND_FILE}"
        if not (round_folder / ROUND_FILE).is_file():
            problems.append(f"{round_folder.name} has no {ROUND_FILE}, so the round was interrupted.")
            continue

        # Check the number and the link to the previous complete round.
        record = convert_record(label, read_json(round_folder / ROUND_FILE, label), round_from_dict)
        if record.number != number:
            raise ExperimentRecordError(f"{label}: round is invalid: {record.number} does not match its folder.")
        if record.iteration is not None and record.iteration.parent_id != parent_id:
            raise ExperimentRecordError(f"round-{number} does not link to round-{number - 1}. The records were edited or mixed.")

        # Verify the files, then remember this round as the parent of the next complete one.
        round_problems, round_matched = verify_round_files(record, round_folder)
        problems += round_problems
        matched += round_matched
        parent_id = record.iteration.id if record.iteration else parent_id
        rounds.append(record)

    return rounds, problems, matched


def load_session(folder: Path) -> SavedSession:
    """
    Load a session folder's records, failing on structural faults and listing artifact problems.

    Args:
        folder (Path): Session folder created by sketchloop run.

    Returns:
        SavedSession: The session, its rounds, and any problems found.
    """
    if not (folder / SESSION_FILE).is_file():
        raise ExperimentRecordError(f"No {SESSION_FILE} in {folder.as_posix()}. Give a session folder created by sketchloop run.")

    session = convert_record(SESSION_FILE, read_json(folder / SESSION_FILE, SESSION_FILE), session_from_dict)
    rounds, problems, matched = load_rounds(folder)
    return SavedSession(folder=folder, session=session, rounds=tuple(rounds), problems=tuple(problems), checked_files=matched)


def rebuild_request(record: RoundRecord, capabilities: GeneratorCapabilities) -> GenerationRequest:
    """
    Rebuild a completed round's request, pinning the effective controls and the first candidate's seed.

    Args:
        record (RoundRecord): Completed round to repeat.
        capabilities (GeneratorCapabilities): Capabilities of the new generator.

    Returns:
        GenerationRequest: The request to repeat, with every declared control set to its recorded effective value.
    """
    # Pin every control the generator declares, so changed defaults cannot leak into the rerun.
    request, result = record.request, record.iteration.result
    effective = result.effective.controls
    controls: dict[str, ControlValue] = {spec.name: effective[spec.name] for spec in capabilities.controls
                                         if not isinstance(effective.get(spec.name, Unavailable("absent")), Unavailable)}

    # Candidate seeds count up from the first one, so the first seed restores them all.
    first_seed = result.candidates[0].seed
    seed = None if isinstance(first_seed, Unavailable) else first_seed
    guidance = Guidance(prompt=request.guidance.prompt, negative_prompt=request.guidance.negative_prompt, controls=controls)
    return GenerationRequest(sketch=request.sketch, guidance=guidance, n_candidates=request.n_candidates, seed=seed)


def format_value(value: object) -> str:
    """
    Format a recorded value for a difference line.
    """
    return f"unavailable ({value.reason})" if isinstance(value, Unavailable) else "none" if value is None else str(value)


def list_differences(original: Mapping[str, object], rerun: Mapping[str, object]) -> list[str]:
    """
    Describe each value that differs between two flat mappings, such as "device: cuda -> cpu".
    """
    names = sorted(set(original) | set(rerun))
    return [f"{name}: {format_value(original.get(name))} -> {format_value(rerun.get(name))}" for name in names
            if original.get(name) != rerun.get(name)]


def describe_conditions(record: RoundRecord, environment: Environment) -> dict[str, object]:
    """
    Flatten a round's backend, effective settings, seeds, and environment into named values to compare.

    Args:
        record (RoundRecord): Completed round.
        environment (Environment): Environment of the round's session.

    Returns:
        dict[str, object]: Value by name.
    """
    result = record.iteration.result
    backend, effective = result.backend, result.effective
    conditions: dict[str, object] = {
        "backend": backend.adapter,
        "adapter version": backend.adapter_version,
        "execution": backend.execution,
        "model": backend.model_id,
        "prompt": effective.prompt,
        "negative prompt": effective.negative_prompt,
        "candidate seeds": " ".join(format_value(candidate.seed) for candidate in result.candidates),
        "python": environment.py_version,
        "platform": environment.platform,
    }

    # Add the settings and packages, skipping a backend's seeds setting, which candidate seeds already cover.
    conditions |= {name: value for name, value in effective.controls.items() if name != "seeds"}
    conditions |= {f"package {name}": version for name, version in environment.packages.items()}
    return conditions


def compare_rounds(original: SavedSession, round_number: int, rerun: SavedSession) -> RerunComparison:
    """
    Compare a rerun session's round with the original round by candidate bytes, settings, and environment.

    Args:
        original (SavedSession): Session holding the original round.
        round_number (int): Number of the original round.
        rerun (SavedSession): Rerun session, whose first round repeats the original.

    Returns:
        RerunComparison: Identical candidate numbers and one line per difference.
    """
    # Compare only completed rounds, since a failed one has no candidates.
    source = next((record for record in original.rounds if record.number == round_number), None)
    repeat = rerun.rounds[0] if rerun.rounds else None
    if source is None or repeat is None or source.iteration is None or repeat.iteration is None:
        return RerunComparison(identical_candidates=(), candidate_count=0, differences=("A round did not complete.",))

    # Match candidates by position and compare their recorded checksums.
    pairs = zip(source.iteration.result.candidates, repeat.iteration.result.candidates)
    identical = tuple(first.index + 1 for first, second in pairs if first.image.sha256 == second.image.sha256)
    original_conditions = describe_conditions(source, original.session.environment)
    rerun_conditions = describe_conditions(repeat, rerun.session.environment)
    differences = list_differences(original_conditions, rerun_conditions)
    count = len(source.iteration.result.candidates)
    return RerunComparison(identical_candidates=identical, candidate_count=count, differences=tuple(differences))
