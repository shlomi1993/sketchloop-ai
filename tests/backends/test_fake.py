import dataclasses
import hashlib

from typing import Any

from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import (ControlValue, GenerationRequest, Guidance, ImageRef, Iteration, Unavailable,
                               select_candidates)
from sketchloop.generation import GenerationOutput


SKETCH = ImageRef(path="sketches/s1.png", width=8, height=8, mode="L", media_type="image/png", sha256="a" * 64)


SKETCH_BYTES = b"unused by the fake"


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
    request = make_request(n_candidates=2, seed=7)
    output = generator.generate(request, SKETCH_BYTES)
    repeated = FakeGenerator().generate(request, SKETCH_BYTES)
    assert payloads_of(repeated) == payloads_of(output), "Same request must give same images"
    reseeded = generator.generate(dataclasses.replace(request, seed=9), SKETCH_BYTES)
    assert payloads_of(reseeded) != payloads_of(output), "Seed must change images"

    # Check the recorded provenance: checksums, per-candidate seeds, and the fake label.
    parent = Iteration(parent_id=None, request=request, result=output.result)
    first, second = parent.result.candidates
    checksums = [hashlib.sha256(payload).hexdigest() for payload in payloads_of(output)]
    assert checksums == [first.image.sha256, second.image.sha256], "Image checksums must match payload bytes"
    assert [first.seed, second.seed] == [7, 8], f"Expected seed + index, got {[first.seed, second.seed]}"
    assert parent.result.backend.is_fake, "Fake backend must label itself as fake"
    assert isinstance(parent.result.backend.model_id, Unavailable), "Fake model ID must be Unavailable"
    assert output.load_seconds is None, f"The fake loads no model, got load_seconds {output.load_seconds}"

    # Defaults belong in the effective settings only, keeping requested and effective values apart.
    assert request.guidance.controls == {}, "Requested controls must not gain defaults"
    effective = parent.result.effective.controls
    assert effective == {"steps": 4, "guidance_scale": 7.5, "strength": 0.75}, f"Defaults missing from effective: {effective}"

    # Select a candidate and continue from it to show lineage across rounds.
    selection = select_candidates(parent, [second.id])
    child_request = make_request({"steps": 20}, sketch=second.image)
    child_result = generator.generate(child_request, SKETCH_BYTES).result
    child = Iteration(parent_id=selection.iteration_id, request=child_request, result=child_result)
    assert child.parent_id == parent.id, "Child must link to its parent iteration"
    assert child.result.effective.controls["steps"] == 20, "Requested control must override the default"
