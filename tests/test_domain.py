import dataclasses
import pytest

from typing import Any

from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import (ControlValue, GenerationRequest, Guidance, ImageRef, InvalidRecordError,
                               InvalidSelectionError, Iteration, select_candidates)


SKETCH = ImageRef(path="sketches/s1.png", width=8, height=8, mode="L", media_type="image/png", sha256="a" * 64)


SKETCH_BYTES = b"unused by the fake"


def make_request(controls: dict[str, ControlValue] | None = None, sketch: ImageRef = SKETCH, **changes: Any) -> GenerationRequest:
    """
    Build a request for the shared test sketch with optional controls and field overrides.
    """
    return GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair", controls=controls or {}), **changes)


def test_select_candidates_rejects_foreign_and_duplicate_ids() -> None:
    """
    Selecting a foreign or repeated candidate ID must fail and name that ID.
    """
    # Take a foreign ID from a second generation, since its candidates get fresh IDs.
    request = make_request(n_candidates=2)
    iteration = Iteration(parent_id=None, request=request, result=FakeGenerator().generate(request, SKETCH_BYTES).result)
    own, foreign = iteration.result.candidates[0].id, FakeGenerator().generate(request, SKETCH_BYTES).result.candidates[0].id

    # Each error must name the offending ID so the person knows what to fix.
    for candidate_ids, offender in [([foreign], foreign), ([own, own], own)]:
        with pytest.raises(InvalidSelectionError, match=offender):
            select_candidates(iteration, candidate_ids)


@pytest.mark.parametrize("path", ["/sketches/s1.png", "sketches\\s1.png", "C:/s1.png", "sketches/../s1.png"])
def test_image_ref_rejects_unsafe_paths(path: str) -> None:
    """
    Image paths must be relative POSIX paths that cannot escape the artifact folder.
    """
    # Giving the image reference an unsafe path fails validation.
    with pytest.raises(InvalidRecordError):
        dataclasses.replace(SKETCH, path=path)


def test_iteration_rejects_fewer_candidates_than_requested() -> None:
    """
    An iteration must reject a result with fewer candidates than requested.
    """
    # Pair a two-candidate result with a three-candidate request to mimic a backend that returned too few.
    result = FakeGenerator().generate(make_request(n_candidates=2), SKETCH_BYTES).result

    # Building the iteration fails with an error naming both candidate counts.
    with pytest.raises(InvalidRecordError, match="2 candidates but 3"):
        Iteration(parent_id=None, request=make_request(n_candidates=3), result=result)
