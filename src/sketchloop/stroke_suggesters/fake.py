"""
Deterministic fake next-stroke suggester, labeled as fake, for tests and offline runs.
"""

import cv2
import hashlib
import numpy as np
import time

from collections.abc import Callable
from typing import Final

from sketchloop.capture import InvalidSketchError
from sketchloop.domain import (BackendIdentity, ColorMode, EffectiveSettings, ExecutionKind, ImageRef, Unavailable,
                               UnsupportedConfigurationError)
from sketchloop.generation import ControlSpec, control_issues
from sketchloop.stroke_suggestion import (LINE_DRAWING_FILE, StrokeSuggester, StrokeSuggestion, StrokeSuggestionOutput,
                                          StrokeSuggestionRequest, SuggestedPart, name_part_file)


# Pixels darker than this count as strokes when the fake looks for the drawing.
DARK_LEVEL: Final = 128

# Shape sizes: the ring's radius as a share of the canvas's shorter side, the gap to the strokes, and the line thickness.
RING_RADIUS_SHARE: Final = 0.1
MIN_RING_RADIUS_PIXELS: Final = 4
GAP_PIXELS: Final = 8
SHAPE_THICKNESS_PIXELS: Final = 3

# Draws one shape on an image in the given gray level and thickness.
ShapeDrawer = Callable[[np.ndarray, int, int], None]


def find_stroke_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Return the left, top, width, and height of the dark pixels, or None when the canvas has none.
    """
    dark = (gray < DARK_LEVEL).astype(np.uint8)
    return cv2.boundingRect(dark) if cv2.countNonZero(dark) else None


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """
    Move a point onto the nearest canvas pixel, so a shape anchored there always has pixels on the canvas.
    """
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def plan_shapes(gray: np.ndarray) -> tuple[ShapeDrawer, ShapeDrawer]:
    """
    Plan a ring centred below the strokes and a horizontal line to their right, or both around the centre of a blank canvas.

    Args:
        gray (np.ndarray): Grayscale canvas, black strokes on white.

    Returns:
        tuple[ShapeDrawer, ShapeDrawer]: Functions that draw the ring and the line.
    """
    height, width = gray.shape
    radius = max(MIN_RING_RADIUS_PIXELS, round(min(width, height) * RING_RADIUS_SHARE))

    # Anchor the shapes to the strokes' box, or to the canvas centre when nothing is drawn.
    box = find_stroke_box(gray)
    if box is None:
        ring_centre = (width // 2, height // 2)
        line_start = (width // 2 + radius + GAP_PIXELS, height // 2)
    else:
        left, top, box_width, box_height = box
        ring_centre = (left + box_width // 2, top + box_height + GAP_PIXELS + radius)
        line_start = (left + box_width + GAP_PIXELS, top + box_height // 2)

    # Clamp the anchors onto the canvas, so strokes at an edge still leave both shapes partly visible.
    ring_centre = clamp_point(*ring_centre, width, height)
    line_start = clamp_point(*line_start, width, height)
    line_end = (line_start[0] + 2 * radius, line_start[1])

    def draw_ring(image: np.ndarray, level: int, thickness: int) -> None:
        cv2.circle(image, ring_centre, radius, level, thickness)

    def draw_line(image: np.ndarray, level: int, thickness: int) -> None:
        cv2.line(image, line_start, line_end, level, thickness)

    return draw_ring, draw_line


def encode_png(image: np.ndarray, path: str) -> tuple[ImageRef, bytes]:
    """
    Encode a grayscale image as PNG and describe it under the given bare file name.

    Args:
        image (np.ndarray): Grayscale image.
        path (str): Bare file name the caller places, such as part-1.png.

    Returns:
        tuple[ImageRef, bytes]: The image reference and the PNG bytes.
    """
    payload = cv2.imencode(".png", image)[1].tobytes()
    height, width = image.shape
    image_ref = ImageRef(path=path, width=width, height=height, mode=ColorMode.L, media_type="image/png",
                         sha256=hashlib.sha256(payload).hexdigest())
    return image_ref, payload


def decode_canvas(request: StrokeSuggestionRequest, canvas_payload: bytes) -> np.ndarray:
    """
    Decode the canvas bytes to grayscale, refusing bytes that are not an image of the request's canvas size.

    Args:
        request (StrokeSuggestionRequest): Request whose canvas describes the expected size.
        canvas_payload (bytes): PNG bytes of the canvas.

    Returns:
        np.ndarray: Grayscale canvas.
    """
    # OpenCV refuses empty bytes with its own error, so treat them as undecodable here.
    gray = cv2.imdecode(np.frombuffer(canvas_payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if canvas_payload else None
    if gray is None:
        raise InvalidSketchError("Suggestion canvas cannot be decoded. Capture the paper again and retry.")

    # A canvas of another size would give masks that do not overlay the recorded canvas.
    actual = f"{gray.shape[1]}x{gray.shape[0]}"
    expected = f"{request.canvas.width}x{request.canvas.height}"
    if actual != expected:
        raise InvalidSketchError(f"Suggestion canvas is {actual}, not the requested {expected}. Capture again and retry.")

    return gray


class FakeStrokeSuggester(StrokeSuggester):
    """
    Deterministic test-only suggester that draws a ring below the strokes and a line to their right, labeled as fake.
    """

    # Saved with every suggestion, so never rename it.
    ADAPTER_NAME: Final = "sketchloop.fake-strokes"

    @property
    def controls(self) -> tuple[ControlSpec, ...]:
        """
        Return no controls, since the fake's shapes depend only on the canvas.
        """
        return ()

    def describe(self) -> str:
        """
        Say that this suggester draws test shapes, not real AI suggestions.
        """
        return "Using the fake stroke suggester, which draws test shapes, not real AI suggestions."

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Validate the request and suggest a ring and a line placed by the strokes on the canvas.

        Args:
            request (StrokeSuggestionRequest): Request to suggest for, whose prompt and seed are recorded and otherwise unused.
            canvas_payload (bytes): PNG bytes of the request's canvas.

        Returns:
            StrokeSuggestionOutput: The suggestion and the bytes of its line drawing and masks.
        """
        # Refuse controls like a real suggester, before reading the canvas.
        issues = control_issues(request.controls, self.controls, "controls")
        if issues:
            raise UnsupportedConfigurationError(issues)

        gray = decode_canvas(request, canvas_payload)

        # Draw each shape thick as its mask, thin as its skeleton, and black on the line drawing.
        started = time.perf_counter()
        line_drawing = gray.copy()
        masks = []
        for draw_shape in plan_shapes(gray):
            mask, skeleton = np.zeros_like(gray), np.zeros_like(gray)
            draw_shape(mask, 255, SHAPE_THICKNESS_PIXELS)
            draw_shape(skeleton, 255, 1)
            draw_shape(line_drawing, 0, SHAPE_THICKNESS_PIXELS)
            masks.append((mask, cv2.countNonZero(skeleton)))
        generation_seconds = time.perf_counter() - started

        # Keep the largest parts first, up to the requested number, and encode every image.
        started = time.perf_counter()
        masks.sort(key=lambda entry: cv2.countNonZero(entry[0]), reverse=True)
        line_drawing_ref, line_drawing_payload = encode_png(line_drawing, LINE_DRAWING_FILE)
        parts = []
        payloads = {LINE_DRAWING_FILE: line_drawing_payload}
        for index, (mask, skeleton_pixels) in enumerate(masks[:request.max_parts]):
            mask_ref, payloads[name_part_file(index)] = encode_png(mask, name_part_file(index))
            box = tuple(int(value) for value in cv2.boundingRect(mask))
            parts.append(SuggestedPart(index=index, mask=mask_ref, box=box, pixel_count=cv2.countNonZero(mask),
                                       skeleton_pixels=skeleton_pixels))
        extraction_seconds = time.perf_counter() - started

        # Label the suggester as fake, and record a seed only when one was given, since the fake draws none.
        suggester = BackendIdentity(adapter=self.ADAPTER_NAME, adapter_version="1", execution=ExecutionKind.FAKE,
                                    model_id=Unavailable("fake stroke suggester has no model"))
        seed = request.seed if request.seed is not None else Unavailable("fake stroke suggester draws no random seed")
        effective = EffectiveSettings(prompt=request.prompt, negative_prompt=None, controls={})
        timings_seconds = {"generation": generation_seconds, "extraction": extraction_seconds}
        suggestion = StrokeSuggestion(suggester=suggester, effective=effective, seed=seed, line_drawing=line_drawing_ref,
                                      parts=tuple(parts), timings_seconds=timings_seconds)
        return StrokeSuggestionOutput(suggestion=suggestion, payloads=payloads)
