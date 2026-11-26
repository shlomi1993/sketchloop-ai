"""
Tests of the deterministic fake stroke suggester, which draws known shapes next to the strokes.
"""

import cv2
import hashlib
import numpy as np
import pytest

from sketchloop.capture import InvalidSketchError
from sketchloop.domain import ColorMode, ImageRef, Unavailable, UnsupportedConfigurationError
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.stroke_suggestion import LINE_DRAWING_FILE, StrokeSuggestionOutput, StrokeSuggestionRequest


CANVAS_WIDTH = 512
CANVAS_HEIGHT = 360
STROKE_BOX = (100, 60, 101, 81)  # Left, top, width, and height of the test rectangle's strokes


def encode_canvas(rectangle: tuple[int, int, int, int] | None = STROKE_BOX, width: int = CANVAS_WIDTH,
                  height: int = CANVAS_HEIGHT) -> bytes:
    """
    Encode a white canvas as PNG, with a black rectangle outline at the given left, top, width, and height, or blank.
    """
    canvas = np.full((height, width), 255, dtype=np.uint8)
    if rectangle is not None:
        left, top, box_width, box_height = rectangle
        cv2.rectangle(canvas, (left, top), (left + box_width - 1, top + box_height - 1), 0, 1)

    return cv2.imencode(".png", canvas)[1].tobytes()


def make_request(payload: bytes, **changes: object) -> StrokeSuggestionRequest:
    """
    Build a request for the given canvas bytes, with optional field overrides.
    """
    height, width = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).shape
    canvas = ImageRef(path="strokes.png", width=width, height=height, mode=ColorMode.L, media_type="image/png",
                      sha256=hashlib.sha256(payload).hexdigest())
    return StrokeSuggestionRequest(**({"canvas": canvas, "prompt": "a car"} | changes))


def suggest(payload: bytes, **changes: object) -> StrokeSuggestionOutput:
    """
    Run the fake suggester on the canvas bytes with a request built from them.
    """
    return FakeStrokeSuggester().suggest(make_request(payload, **changes), payload)


def decode_mask(output: StrokeSuggestionOutput, path: str) -> np.ndarray:
    """
    Decode one stored image of the output as a grayscale array.
    """
    return cv2.imdecode(np.frombuffer(output.payloads[path], dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def test_same_canvas_gives_same_parts_and_bytes() -> None:
    """
    The fake is deterministic for a canvas whatever the seed or prompt, and records a missing seed as unavailable.
    """
    payload = encode_canvas()
    first = suggest(payload)
    second = suggest(payload, seed=9, prompt="")
    assert first.suggestion.parts == second.suggestion.parts, "The same canvas must give the same parts"
    assert dict(first.payloads) == dict(second.payloads), "The same canvas must give the same image bytes"

    # Without a seed, the fake records that it used none instead of inventing one.
    assert isinstance(first.suggestion.seed, Unavailable), f"Expected an unavailable seed, got {first.suggestion.seed!r}"


def test_parts_lie_below_and_right_of_the_strokes() -> None:
    """
    A ring lies below the strokes' box and a line to its right, both inside the canvas, largest first.
    """
    output = suggest(encode_canvas())
    parts = output.suggestion.parts
    assert len(parts) == 2, f"Expected a ring and a line, got {len(parts)} parts"

    # The ring is the larger part and starts below the strokes' bottom edge.
    left, top, width, height = STROKE_BOX
    ring, line = parts
    assert ring.pixel_count > line.pixel_count, f"Parts must be largest first, got {ring.pixel_count} then {line.pixel_count}"
    assert ring.box[1] >= top + height, f"The ring must start below the strokes, got box {ring.box}"

    # The line starts right of the strokes and is wider than tall.
    assert line.box[0] >= left + width, f"The line must start right of the strokes, got box {line.box}"
    assert line.box[2] > line.box[3], f"The line must be horizontal, got box {line.box}"

    # Every mask is at the canvas size, holds only 0 and 255, and counts its pixels.
    for part in parts:
        mask = decode_mask(output, part.mask.path)
        assert mask.shape == (CANVAS_HEIGHT, CANVAS_WIDTH), f"Part {part.index} mask has size {mask.shape}"
        assert set(np.unique(mask)) <= {0, 255}, f"Part {part.index} mask must be binary"
        assert int(np.count_nonzero(mask)) == part.pixel_count, f"Part {part.index} pixel_count must match its mask"
        assert part.skeleton_pixels < part.pixel_count, f"Part {part.index} skeleton must be thinner than the part"


@pytest.mark.parametrize("rectangle", [None, (440, 300, 72, 60), (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)],
                         ids=["blank", "bottom-right-corner", "whole-canvas"])
def test_parts_stay_inside_the_canvas(rectangle: tuple[int, int, int, int] | None) -> None:
    """
    A blank canvas, or strokes reaching the canvas edge, still gives two non-empty parts inside the canvas.
    """
    output = suggest(encode_canvas(rectangle))
    assert len(output.suggestion.parts) == 2, f"Expected two parts, got {len(output.suggestion.parts)}"

    # Each box lies inside the canvas, which the record also checks, and each part has pixels.
    for part in output.suggestion.parts:
        left, top, width, height = part.box
        assert left + width <= CANVAS_WIDTH and top + height <= CANVAS_HEIGHT, f"Part {part.index} box {part.box} is outside"
        assert part.pixel_count >= 1, f"Part {part.index} must have pixels"


def test_records_label_the_fake_and_keep_the_request() -> None:
    """
    The suggestion is labeled fake, keeps the prompt as typed and the given seed, and stores a line drawing of canvas size.
    """
    output = suggest(encode_canvas(), prompt="  ", seed=5)
    suggestion = output.suggestion

    # The fake label and the absent model are recorded.
    assert suggestion.suggester.is_fake, "The fake suggester must label itself as fake"
    assert suggestion.suggester.adapter == FakeStrokeSuggester.ADAPTER_NAME, f"Unexpected adapter {suggestion.suggester.adapter}"
    assert isinstance(suggestion.suggester.model_id, Unavailable), "The fake has no model ID"
    assert output.load_seconds is None, f"The fake loads no model, got {output.load_seconds}"

    # The prompt and seed are recorded unchanged, with no controls.
    assert suggestion.effective.prompt == "  ", f"The prompt must stay as typed, got {suggestion.effective.prompt!r}"
    assert suggestion.seed == 5, f"The given seed must be recorded, got {suggestion.seed}"
    assert dict(suggestion.effective.controls) == {}, f"The fake has no controls, got {dict(suggestion.effective.controls)}"
    assert set(suggestion.timings_seconds) == {"generation", "extraction"}, f"Got timings {dict(suggestion.timings_seconds)}"

    # The line drawing is the canvas with the parts drawn on it.
    line_drawing = decode_mask(output, LINE_DRAWING_FILE)
    assert line_drawing.shape == (CANVAS_HEIGHT, CANVAS_WIDTH), f"Line drawing has size {line_drawing.shape}"
    ring_mask = decode_mask(output, suggestion.parts[0].mask.path)
    assert np.all(line_drawing[ring_mask > 0] == 0), "The ring must be drawn black on the line drawing"


def test_max_parts_keeps_the_largest() -> None:
    """
    Asking for one part returns only the largest, the ring.
    """
    output = suggest(encode_canvas(), max_parts=1)
    assert len(output.suggestion.parts) == 1, f"Expected one part, got {len(output.suggestion.parts)}"
    assert set(output.payloads) == {LINE_DRAWING_FILE, "part-1.png"}, f"Expected one mask, got {list(output.payloads)}"


def test_any_control_is_refused_through_the_shared_check() -> None:
    """
    The fake has no controls, so a requested control is refused with the issue the shared control check names.
    """
    with pytest.raises(UnsupportedConfigurationError) as raised:
        suggest(encode_canvas(), controls={"strength": 0.85})
    fields = [(issue.field, issue.code) for issue in raised.value.issues]
    assert fields == [("controls.strength", "unknown_control")], f"Expected one unknown control, got {fields}"


@pytest.mark.parametrize("payload", [b"not an image", encode_canvas(width=256)], ids=["garbage", "wrong-size"])
def test_unreadable_or_mismatched_canvas_is_refused(payload: bytes) -> None:
    """
    Canvas bytes that cannot be decoded, or whose size differs from the request's canvas, are refused before suggesting.
    """
    request = make_request(encode_canvas())
    with pytest.raises(InvalidSketchError, match="canvas"):
        FakeStrokeSuggester().suggest(request, payload)
