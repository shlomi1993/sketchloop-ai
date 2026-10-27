"""
Rebuilding a saved round's request and comparing a rerun with the original round.
"""

from collections.abc import Mapping

from sketchloop.domain import ControlValue, GenerationRequest, Guidance, Unavailable
from sketchloop.experiments.records import Environment, RerunComparison, RoundRecord, SavedSession
from sketchloop.generation import GeneratorCapabilities


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

    # Add the hardware facts, each unavailable for the same reason when the whole record is.
    hardware = environment.hardware
    for fact in ("cpu", "logical_cores", "memory_bytes"):
        conditions[fact.replace("_", " ")] = hardware if isinstance(hardware, Unavailable) else getattr(hardware, fact)

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
