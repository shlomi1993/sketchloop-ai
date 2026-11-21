"""
Tests of the next-stroke suggester contract: its request, part, suggestion, and output records and the base class.
"""

import hashlib
import math
import pytest

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from sketchloop.domain import (BackendIdentity, ColorMode, EffectiveSettings, ExecutionKind, ImageRef, InvalidRecordError,
                               Unavailable)
from sketchloop.generation import ControlSpec
from sketchloop.stroke_suggestion import (LINE_DRAWING_FILE, MAX_PARTS, PART_COLORS, StrokeSuggester, StrokeSuggestion,
                                          StrokeSuggestionOutput, StrokeSuggestionRequest, SuggestedPart, name_part_file)


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


@pytest.mark.parametrize("prompt", ["", "   ", "  a red sports car, side view  "], ids=["empty", "spaces", "padded"])
def test_request_keeps_prompt_exactly_as_typed(prompt: str) -> None:
    """
    Any prompt text, empty or padded with spaces, is accepted and kept unchanged, with the defaults filled in.
    """
    request = StrokeSuggestionRequest(canvas=CANVAS, prompt=prompt)

    # The prompt is not stripped, and the defaults ask for every part with a random seed.
    assert request.prompt == prompt, f"The prompt must stay exactly as typed, got {request.prompt!r}"
    assert request.seed is None and request.max_parts == MAX_PARTS, f"Unexpected defaults: {request}"
    assert dict(request.controls) == {}, f"Default controls must be empty, got {dict(request.controls)}"


def test_request_freezes_controls() -> None:
    """
    The request copies its controls into a read-only view, so a later change to the caller's dict cannot alter it.
    """
    controls = {"strength": 0.85}
    request = StrokeSuggestionRequest(canvas=CANVAS, prompt="", controls=controls)
    controls["strength"] = 0.1

    # The record keeps the value it was built with and refuses changes.
    assert request.controls["strength"] == 0.85, f"The request must copy its controls, got {dict(request.controls)}"
    with pytest.raises(TypeError):
        request.controls["strength"] = 0.5


@pytest.mark.parametrize(("changes", "message"), [
    ({"max_parts": 0}, "max_parts must be an int from 1 to 4"),
    ({"max_parts": MAX_PARTS + 1}, "max_parts must be an int from 1 to 4"),
    ({"max_parts": True}, "max_parts must be an int from 1 to 4"),
    ({"seed": -1}, "seed must be None or an int >= 0"),
    ({"prompt": None}, "prompt must be text"),
    ({"canvas": replace(CANVAS, width=500)}, "multiples of 8, got 500x360"),
    ({"canvas": replace(CANVAS, height=362)}, "multiples of 8, got 512x362"),
    ({"canvas": replace(CANVAS, mode=ColorMode.RGB)}, "canvas must be mode L"),
    ({"controls": {"strength": math.nan}}, "must be finite"),
    ({"controls": {"Strength": 0.5}}, "must be snake_case")
], ids=["no-parts", "too-many-parts", "bool-parts", "negative-seed", "no-prompt", "width-500", "height-362", "rgb-canvas",
        "nan-control", "control-name"])
def test_request_refuses_invalid_values(changes: Mapping[str, Any], message: str) -> None:
    """
    A request with a bad part limit, seed, prompt, canvas, or control is refused with a message naming the problem.
    """
    values = {"canvas": CANVAS, "prompt": "a chair"} | dict(changes)
    with pytest.raises(InvalidRecordError, match=message):
        StrokeSuggestionRequest(**values)


def test_part_holds_a_box_loaded_as_a_list_as_a_tuple() -> None:
    """
    A box read back from JSON as a list is held as a tuple, so equal parts compare equal.
    """
    part = make_part(0, box=[10, 20, 30, 40])
    assert part.box == (10, 20, 30, 40), f"The box must be held as a tuple, got {part.box!r}"


@pytest.mark.parametrize(("changes", "message"), [
    ({"index": -1}, "index must be an int >= 0"),
    ({"box": (10, 20, 30)}, "box must be four ints"),
    ({"box": (-1, 20, 30, 40)}, "box must be four ints"),
    ({"box": (10, 20, 0, 40)}, "must lie inside its 512x360 mask"),
    ({"box": (500, 20, 30, 40)}, "must lie inside its 512x360 mask"),
    ({"box": (10, 340, 30, 40)}, "must lie inside its 512x360 mask"),
    ({"pixel_count": 0}, "pixel_count must be an int from 1"),
    ({"skeleton_pixels": 0}, "skeleton_pixels must be an int from 1 to the pixel count"),
    ({"skeleton_pixels": 301}, "skeleton_pixels must be an int from 1 to the pixel count"),
    ({"mask": make_image("part-1.png", b"mask 0", mode=ColorMode.RGB)}, "mask must be mode L"),
    ({"label": " "}, "label must be None or non-empty text")
], ids=["negative-index", "short-box", "negative-left", "zero-width", "past-right", "past-bottom", "no-pixels", "no-skeleton",
        "skeleton-above-pixels", "rgb-mask", "blank-label"])
def test_part_refuses_invalid_values(changes: Mapping[str, Any], message: str) -> None:
    """
    A part with a bad index, box, count, mask mode, or label is refused with a message naming the problem.
    """
    with pytest.raises(InvalidRecordError, match=message):
        make_part(**({"index": 0} | dict(changes)))


def test_suggestion_without_parts_is_valid() -> None:
    """
    No new parts is a valid suggestion, and its output needs only the line drawing's bytes.
    """
    suggestion = make_suggestion(part_count=0)
    output = StrokeSuggestionOutput(suggestion=suggestion, payloads=make_payloads(part_count=0))
    assert output.suggestion.parts == (), f"Expected no parts, got {output.suggestion.parts}"


@pytest.mark.parametrize(("changes", "message"), [
    ({"parts": (make_part(0), make_part(2))}, r"Part indexes must be 0\.\.1 in order, got \[0, 2\]"),
    ({"parts": (make_part(1), make_part(0))}, r"Part indexes must be 0\.\.1 in order, got \[1, 0\]"),
    ({"parts": tuple(make_part(index) for index in range(MAX_PARTS + 1))}, "at most 4 parts, got 5"),
    ({"parts": (make_part(0), make_part(1, mask=make_image("part-2.png", b"mask 1", width=256)))}, "share one size"),
    ({"line_drawing": make_image(LINE_DRAWING_FILE, b"line drawing", height=512)}, "share one size"),
    ({"line_drawing": make_image(name_part_file(0), b"line drawing")}, "image paths must be unique"),
    ({"seed": -2}, "seed must be an int >= 0 or Unavailable"),
    ({"timings_seconds": {"generation": -1.0}}, "timings_seconds.generation must be finite seconds")
], ids=["index-gap", "index-order", "too-many-parts", "mask-size", "line-drawing-size", "shared-path", "negative-seed",
        "negative-timing"])
def test_suggestion_refuses_invalid_values(changes: Mapping[str, Any], message: str) -> None:
    """
    A suggestion with an index gap, too many parts, mismatched sizes, shared paths, or a bad seed or timing is refused.
    """
    with pytest.raises(InvalidRecordError, match=message):
        make_suggestion(**changes)


def test_suggestion_accepts_unavailable_seed_and_line_drawing() -> None:
    """
    A suggester that cannot report its seed or keep its line drawing records both as unavailable, leaving only the masks.
    """
    suggestion = make_suggestion(seed=Unavailable("no seed"), line_drawing=Unavailable("not kept"))
    paths = [image.path for image in suggestion.images]
    assert paths == ["part-1.png", "part-2.png"], f"Expected only the masks, got {paths}"


@pytest.mark.parametrize(("payloads", "message"), [
    ({LINE_DRAWING_FILE: b"line drawing", "part-1.png": b"mask 0"}, "Payload keys must equal"),
    (make_payloads() | {"extra.png": b"x"}, "Payload keys must equal"),
    (make_payloads() | {"part-2.png": b"changed"}, "Payload part-2.png fails its checksum")
], ids=["missing-mask", "extra-file", "changed-mask"])
def test_output_refuses_payloads_that_do_not_match(payloads: Mapping[str, bytes], message: str) -> None:
    """
    The output needs exactly one payload per line drawing and mask, each matching the checksum the suggestion recorded.
    """
    with pytest.raises(InvalidRecordError, match=message):
        StrokeSuggestionOutput(suggestion=make_suggestion(), payloads=payloads)


def test_output_checks_load_seconds_and_freezes_payloads() -> None:
    """
    A reported load time must be finite seconds, and the payloads are copied into a read-only view.
    """
    payloads = make_payloads()
    output = StrokeSuggestionOutput(suggestion=make_suggestion(), payloads=payloads, load_seconds=2.5)
    payloads.clear()

    # The output keeps its own copy and refuses changes.
    assert len(output.payloads) == 3, f"The output must copy its payloads, got {list(output.payloads)}"
    with pytest.raises(TypeError):
        output.payloads["part-1.png"] = b"x"

    # A negative load time is refused.
    with pytest.raises(InvalidRecordError, match="load_seconds must be finite seconds"):
        StrokeSuggestionOutput(suggestion=make_suggestion(), payloads=make_payloads(), load_seconds=-1.0)


class EmptyStrokeSuggester(StrokeSuggester):
    """
    Test-only suggester that implements only the abstract methods, to check the base class defaults.
    """

    @property
    def controls(self) -> tuple[ControlSpec, ...]:
        """
        Return no controls.
        """
        return ()

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Return a suggestion without parts.
        """
        return StrokeSuggestionOutput(suggestion=make_suggestion(part_count=0), payloads=make_payloads(part_count=0))

    def describe(self) -> str:
        """
        Name the test suggester.
        """
        return "Using the empty test suggester."


def test_base_class_requires_suggest_and_skips_load() -> None:
    """
    The base class cannot be created without its abstract methods, and its default load does nothing.
    """
    # Every abstract method must be implemented.
    with pytest.raises(TypeError):
        StrokeSuggester()

    # The default load accepts the canvas size and returns without loading anything.
    suggester = EmptyStrokeSuggester()
    assert suggester.load(512, 360) is None, "The default load must do nothing"
    assert suggester.suggest(StrokeSuggestionRequest(canvas=CANVAS, prompt=""), b"").suggestion.parts == (), "Expected no parts"


def test_part_colors_cover_every_part() -> None:
    """
    There is one distinct overlay color per possible part, and part files are numbered from 1.
    """
    assert len(set(PART_COLORS)) == MAX_PARTS, f"Expected {MAX_PARTS} distinct colors, got {PART_COLORS}"
    assert name_part_file(0) == "part-1.png", f"Part files must be numbered from 1, got {name_part_file(0)}"
