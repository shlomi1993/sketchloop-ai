"""
Tests of the next-stroke suggester contract: its request, part, suggestion, and output records.
"""

import hashlib
import pytest

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from sketchloop.domain import (BackendIdentity, ColorMode, EffectiveSettings, ExecutionKind, ImageRef, InvalidRecordError,
                               Unavailable)
from sketchloop.stroke_suggestion import (LINE_DRAWING_FILE, StrokeSuggestion, StrokeSuggestionOutput,
                                          StrokeSuggestionRequest, SuggestedPart, name_part_file)


CANVAS = ImageRef(path="strokes.png", width=512, height=360, mode="L", media_type="image/png", sha256="a" * 64)
IDENTITY = BackendIdentity(adapter="tests.strokes", adapter_version="1", execution=ExecutionKind.FAKE,
                           model_id=Unavailable("test suggester has no model"))
EFFECTIVE = EffectiveSettings(prompt="a chair", negative_prompt=None, controls={})


def make_image(path: str, payload: bytes, width: int = 512, height: int = 360, mode: ColorMode = ColorMode.L) -> ImageRef:
    """
    Build an image reference whose checksum matches the payload.
    """
    return ImageRef(path=path, width=width, height=height, mode=mode, media_type="image/png",
                    sha256=hashlib.sha256(payload).hexdigest())


def make_part(index: int, **changes: Any) -> SuggestedPart:
    """
    Build a valid part at the canvas size, with optional field overrides.
    """
    mask = make_image(name_part_file(index), f"mask {index}".encode())
    values = {"index": index, "mask": mask, "box": (10, 20, 30, 40), "pixel_count": 300, "skeleton_pixels": 100} | changes
    return SuggestedPart(**values)


def make_suggestion(part_count: int = 2, **changes: Any) -> StrokeSuggestion:
    """
    Build a valid suggestion with the given number of parts and a line drawing, with optional field overrides.
    """
    parts = tuple(make_part(index) for index in range(part_count))
    line_drawing = make_image(LINE_DRAWING_FILE, b"line drawing")
    suggestion = StrokeSuggestion(suggester=IDENTITY, effective=EFFECTIVE, seed=3, line_drawing=line_drawing, parts=parts,
                                  timings_seconds={"generation": 0.5, "extraction": 0.01})
    return replace(suggestion, **changes)


def make_payloads(part_count: int = 2) -> dict[str, bytes]:
    """
    Return the bytes matching make_suggestion's line drawing and masks, by path.
    """
    masks = {name_part_file(index): f"mask {index}".encode() for index in range(part_count)}
    return {LINE_DRAWING_FILE: b"line drawing", **masks}


def test_request_refuses_a_canvas_off_the_latent_grid() -> None:
    """
    A request whose canvas is not a multiple of 8 on each side is refused with a message naming the size.
    """
    with pytest.raises(InvalidRecordError, match="multiples of 8, got 500x360"):
        StrokeSuggestionRequest(canvas=replace(CANVAS, width=500), prompt="a chair")


def test_part_refuses_a_box_outside_its_mask() -> None:
    """
    A part whose box runs past its mask's right edge is refused with a message naming the mask size.
    """
    with pytest.raises(InvalidRecordError, match="must lie inside its 512x360 mask"):
        make_part(0, box=(500, 20, 30, 40))


@pytest.mark.parametrize(("changes", "message"), [
    ({"parts": (make_part(0), make_part(2))}, r"Part indexes must be 0\.\.1 in order, got \[0, 2\]"),
    ({"parts": (make_part(0), make_part(1, mask=make_image("part-2.png", b"mask 1", width=256)))}, "share one size"),
], ids=["index-gap", "mask-size"])
def test_suggestion_refuses_invalid_values(changes: Mapping[str, Any], message: str) -> None:
    """
    A suggestion with an index gap or masks of different sizes is refused.
    """
    with pytest.raises(InvalidRecordError, match=message):
        make_suggestion(**changes)


@pytest.mark.parametrize(("payloads", "message"), [
    ({LINE_DRAWING_FILE: b"line drawing", "part-1.png": b"mask 0"}, "Payload keys must equal"),
    (make_payloads() | {"part-2.png": b"changed"}, "Payload part-2.png fails its checksum")
], ids=["missing-mask", "changed-mask"])
def test_output_refuses_payloads_that_do_not_match(payloads: Mapping[str, bytes], message: str) -> None:
    """
    The output needs exactly one payload per line drawing and mask, each matching the checksum the suggestion recorded.
    """
    with pytest.raises(InvalidRecordError, match=message):
        StrokeSuggestionOutput(suggestion=make_suggestion(), payloads=payloads)
