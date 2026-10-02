import dataclasses
import hashlib
import pytest

from typing import Any

from sketchloop.domain import (ControlValue, GenerationRequest, Guidance, ImageRef, InvalidRecordError,
                               InvalidSelectionError, Iteration, Unavailable, UnsupportedConfigurationError,
                               select_candidates)
from sketchloop.fakes import FakeGenerator
from sketchloop.generation import ControlSpec, GenerationOutput, GeneratorCapabilities


SKETCH = ImageRef(path="sketches/s1.png", width=8, height=8, mode="L", media_type="image/png", sha256="a" * 64)


def make_request(controls: dict[str, ControlValue] | None = None, sketch: ImageRef = SKETCH, **changes: Any) -> GenerationRequest:
    """
    Build a request for the shared test sketch with optional controls and field overrides.
    """
    return GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair", controls=controls or {}), **changes)


def payloads_of(output: GenerationOutput) -> list[bytes]:
    """
    Return the candidates' image bytes in candidate order.
    """
    return [output.payloads[candidate.image.path] for candidate in output.result.candidates]


def test_fake_loop_generates_selects_and_builds_child_iteration() -> None:
    """
    One full round with the fake: generate, record provenance, select, and continue.
    """
    # The fake must be deterministic per request yet sensitive to the seed.
    generator = FakeGenerator()
    request = make_request(num_candidates=2, seed=7)
    output = generator.generate(request)
    assert payloads_of(FakeGenerator().generate(request)) == payloads_of(output), "Same request must give same images"
    assert payloads_of(generator.generate(dataclasses.replace(request, seed=9))) != payloads_of(output), "Seed must change images"

    # Check the recorded provenance: checksums, per-candidate seeds, and the fake label.
    parent = Iteration(parent_id=None, request=request, result=output.result)
    first, second = parent.result.candidates
    checksums = [hashlib.sha256(payload).hexdigest() for payload in payloads_of(output)]
    assert checksums == [first.image.sha256, second.image.sha256], "Image checksums must match payload bytes"
    assert [first.seed, second.seed] == [7, 8], f"Expected seed + index, got {[first.seed, second.seed]}"
    assert parent.result.backend.is_fake, "Fake backend must label itself as fake"
    assert isinstance(parent.result.backend.model_id, Unavailable), "Fake model ID must be Unavailable"

    # Defaults belong in the effective settings only, keeping requested and effective values apart.
    assert request.guidance.controls == {}, "Requested controls must not gain defaults"
    effective = parent.result.effective.controls
    assert effective == {"steps": 4, "guidance_scale": 7.5, "strength": 0.75}, f"Defaults missing from effective: {effective}"

    # Select a candidate and continue from it to show lineage across rounds.
    selection = select_candidates(parent, [second.id])
    child_request = make_request({"steps": 20}, sketch=second.image)
    child_result = generator.generate(child_request).result
    child = Iteration(parent_id=selection.iteration_id, request=child_request, result=child_result)
    assert child.parent_id == parent.id, "Child must link to its parent iteration"
    assert child.result.effective.controls["steps"] == 20, "Requested control must override the default"


def test_every_unsupported_setting_is_reported_together() -> None:
    """
    A request that breaks every capability must fail once, listing all issues in order.
    """
    # Build a restrictive backend and a request that breaks every rule it has.
    controls = (
        ControlSpec(name="steps", kind="int", minimum=1, maximum=50),
        ControlSpec(name="scale", kind="float", minimum=0, maximum=20),
        ControlSpec(name="sampler", kind="choice", choices=("euler", "ddim"))
    )
    capabilities = GeneratorCapabilities(controls=controls, max_candidates=8, supports_negative_prompt=False, supports_seed=False)
    invalid = {"steps": 51, "scale": True, "sampler": "x", "eta": 1}
    guidance = Guidance(prompt="chair", negative_prompt="", controls=invalid)
    request = GenerationRequest(sketch=SKETCH, guidance=guidance, num_candidates=9, seed=0)
    with pytest.raises(UnsupportedConfigurationError) as raised:
        FakeGenerator(capabilities).generate(request)

    # Request-level issues come first, then controls in name order.
    expected = [
        ("num_candidates", "too_many_candidates"),
        ("seed", "unsupported_feature"),
        ("guidance.negative_prompt", "unsupported_feature"),
        ("guidance.controls.eta", "unknown_control"),
        ("guidance.controls.sampler", "invalid_choice"),
        ("guidance.controls.scale", "wrong_type"),
        ("guidance.controls.steps", "out_of_range")
    ]
    actual = [(issue.field, issue.code) for issue in raised.value.issues]
    assert actual == expected, f"Every issue must be reported in order, got {actual}"


def test_select_candidates_rejects_foreign_and_duplicate_ids() -> None:
    """
    Selecting a foreign or repeated candidate ID must fail and name that ID.
    """
    # Take a foreign ID from a second generation, since its candidates get fresh IDs.
    request = make_request(num_candidates=2)
    iteration = Iteration(parent_id=None, request=request, result=FakeGenerator().generate(request).result)
    own, foreign = iteration.result.candidates[0].id, FakeGenerator().generate(request).result.candidates[0].id

    # Each error must name the offending ID so the person knows what to fix.
    for candidate_ids, offender in [([foreign], foreign), ([own, own], own)]:
        with pytest.raises(InvalidSelectionError, match=offender):
            select_candidates(iteration, candidate_ids)


@pytest.mark.parametrize("path", ["/sketches/s1.png", "sketches\\s1.png", "C:/s1.png", "sketches/../s1.png"])
def test_image_ref_rejects_unsafe_paths(path: str) -> None:
    """
    Image paths must be relative POSIX paths that cannot escape the artifact folder.
    """
    with pytest.raises(InvalidRecordError):
        dataclasses.replace(SKETCH, path=path)


def test_iteration_rejects_fewer_candidates_than_requested() -> None:
    """
    An iteration must reject a result with fewer candidates than requested.
    """
    # Pair a two-candidate result with a three-candidate request to mimic a backend that returned too few.
    result = FakeGenerator().generate(make_request(num_candidates=2)).result
    with pytest.raises(InvalidRecordError, match="2 candidates but 3"):
        Iteration(parent_id=None, request=make_request(num_candidates=3), result=result)
