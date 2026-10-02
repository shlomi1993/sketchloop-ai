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


def make_request(controls: dict[str, ControlValue] | None = None, sketch: ImageRef = SKETCH,
                 **changes: Any) -> GenerationRequest:
    return GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair", controls=controls or {}), **changes)


def payloads_of(output: GenerationOutput) -> list[bytes]:
    return [output.payloads[candidate.image.path] for candidate in output.result.candidates]


def test_fake_loop_generates_selects_and_builds_child_iteration() -> None:
    generator = FakeGenerator()
    request = make_request(num_candidates=2, seed=7)
    output = generator.generate(request)
    assert payloads_of(FakeGenerator().generate(request)) == payloads_of(output)
    assert payloads_of(generator.generate(dataclasses.replace(request, seed=9))) != payloads_of(output)

    parent = Iteration(parent_id=None, request=request, result=output.result)
    first, second = parent.result.candidates
    checksums = [hashlib.sha256(payload).hexdigest() for payload in payloads_of(output)]
    assert checksums == [first.image.sha256, second.image.sha256] and [first.seed, second.seed] == [7, 8]
    assert parent.result.backend.is_fake and isinstance(parent.result.backend.model_id, Unavailable)
    assert request.guidance.controls == {}
    assert parent.result.effective.controls == {"steps": 4, "guidance_scale": 7.5, "strength": 0.75}

    selection = select_candidates(parent, [second.id])
    child_request = make_request({"steps": 20}, sketch=second.image)
    child_result = generator.generate(child_request).result
    child = Iteration(parent_id=selection.iteration_id, request=child_request, result=child_result)
    assert child.parent_id == parent.id and child.result.effective.controls["steps"] == 20


def test_every_unsupported_setting_is_reported_together() -> None:
    controls = (ControlSpec(name="steps", kind="int", minimum=1, maximum=50),
                ControlSpec(name="scale", kind="float", minimum=0, maximum=20),
                ControlSpec(name="sampler", kind="choice", choices=("euler", "ddim")))
    capabilities = GeneratorCapabilities(controls=controls, max_candidates=8, supports_negative_prompt=False,
                                         supports_seed=False)
    invalid = {"steps": 51, "scale": True, "sampler": "x", "eta": 1}
    guidance = Guidance(prompt="chair", negative_prompt="", controls=invalid)
    request = GenerationRequest(sketch=SKETCH, guidance=guidance, num_candidates=9, seed=0)
    with pytest.raises(UnsupportedConfigurationError) as raised:
        FakeGenerator(capabilities).generate(request)

    expected = [("num_candidates", "too_many_candidates"), ("seed", "unsupported_feature"),
                ("guidance.negative_prompt", "unsupported_feature"), ("guidance.controls.eta", "unknown_control"),
                ("guidance.controls.sampler", "invalid_choice"), ("guidance.controls.scale", "wrong_type"),
                ("guidance.controls.steps", "out_of_range")]
    assert [(issue.field, issue.code) for issue in raised.value.issues] == expected


def test_select_candidates_rejects_foreign_and_duplicate_ids() -> None:
    request = make_request(num_candidates=2)
    iteration = Iteration(parent_id=None, request=request, result=FakeGenerator().generate(request).result)
    own, foreign = iteration.result.candidates[0].id, FakeGenerator().generate(request).result.candidates[0].id
    for candidate_ids, offender in [([foreign], foreign), ([own, own], own)]:
        with pytest.raises(InvalidSelectionError, match=offender):
            select_candidates(iteration, candidate_ids)


@pytest.mark.parametrize("path", ["/sketches/s1.png", "sketches\\s1.png", "C:/s1.png", "sketches/../s1.png"])
def test_image_ref_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(InvalidRecordError):
        dataclasses.replace(SKETCH, path=path)


def test_iteration_rejects_fewer_candidates_than_requested() -> None:
    result = FakeGenerator().generate(make_request(num_candidates=2)).result
    with pytest.raises(InvalidRecordError, match="2 candidates but 3"):
        Iteration(parent_id=None, request=make_request(num_candidates=3), result=result)
