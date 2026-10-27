import re

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime

from sketchloop.domain import (SCHEMA_VERSION, BackendIdentity, Candidate, EffectiveSettings, GenerationRequest,
                               GenerationResult, Guidance, ImageRef, Iteration, SelectionEvent, Unavailable, require)
from sketchloop.experiments.records import (MISSING_HARDWARE_REASON, MISSING_SOURCE_REASON, Environment, ExperimentRecordError,
                                            Hardware, RerunLink, RoundFailure, RoundRecord, SessionRecord)
from sketchloop.preprocessing import PreprocessingStep


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


def hardware_to_dict(hardware: Hardware | Unavailable) -> dict[str, object]:
    """
    Serialize hardware facts, each through encode_value, or the whole record as unavailable.
    """
    if isinstance(hardware, Unavailable):
        return {"unavailable": hardware.reason}

    return {
        "cpu": encode_value(hardware.cpu),
        "logical_cores": encode_value(hardware.logical_cores),
        "memory_bytes": encode_value(hardware.memory_bytes),
    }


def hardware_from_dict(data: Mapping[str, object]) -> Hardware | Unavailable:
    """
    Rebuild hardware facts, checking for a whole unavailable record first, since decode_value treats any dict as one.
    """
    if "unavailable" in data:
        return decode_value(data)

    return Hardware(cpu=decode_value(data["cpu"]), logical_cores=decode_value(data["logical_cores"]),
                    memory_bytes=decode_value(data["memory_bytes"]))


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
            "hardware": hardware_to_dict(environment.hardware),
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

    # Sessions written before hardware was tracked lack it, which is unknown rather than a default.
    packages = {name: decode_value(version) for name, version in environment_data["packages"].items()}
    hardware = hardware_from_dict(environment_data.get("hardware", {"unavailable": MISSING_HARDWARE_REASON}))
    environment = Environment(py_version=environment_data["python"], platform=environment_data["platform"], packages=packages,
                              hardware=hardware)

    # Rebuild the rerun link.
    link = None
    if link_data is not None:
        link = RerunLink(session_id=link_data["session_id"], round_number=link_data["round"],
                         iteration_id=link_data["iteration_id"])
    return SessionRecord(schema_version=data["schema_version"], session_id=data["session_id"],
                         created_at=parse_time(data["created_at"]), environment=environment, rerun_of=link)
