"""
Tests of follow detection's alignment: darkness maps, occluders, ECC on synthetic captures, and the alignment records.
"""

import cv2
import numpy as np
import pytest

from collections.abc import Callable
from typing import Final

from sketchloop.domain import InvalidRecordError, Unavailable
from sketchloop.follow import (ALIGNED_STROKES_METHOD, NOT_CONVERGED_REASON, AlignedStrokeFollowDetector, Alignment,
                               CaptureScores, describe_follow_method, find_occluders)


CANVAS_SIZE: Final = (512, 360)
PAPER_LEVEL: Final = 225
HAND_CENTER: Final = (420, 320)
IDENTITY: Final = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
CHECK_POINTS: Final = ((100, 80), (400, 80), (100, 280), (400, 280), (256, 180))


def draw_canvas(with_new_leg: bool = False) -> np.ndarray:
    """
    Draw a synthetic paper-frame canvas with a box, a circle, and a leg, optionally with a second, newly drawn leg.
    """
    canvas = np.full(CANVAS_SIZE[::-1], PAPER_LEVEL, dtype=np.uint8)
    cv2.rectangle(canvas, (150, 100), (350, 230), 40, 3)
    cv2.circle(canvas, (300, 160), 30, 40, 2)
    cv2.line(canvas, (160, 230), (160, 300), 40, 3)
    cv2.line(canvas, (60, 60), (120, 80), 50, 2)
    if with_new_leg:
        cv2.line(canvas, (340, 230), (340, 300), 40, 3)

    return canvas


def make_camera_motion(shift: tuple[float, float], angle_degrees: float) -> np.ndarray:
    """
    Return the 2x3 affine map from baseline to capture pixels for a rotation about the centre plus a shift.
    """
    motion = cv2.getRotationMatrix2D((CANVAS_SIZE[0] / 2, CANVAS_SIZE[1] / 2), angle_degrees, 1.0)
    motion[:, 2] += shift
    return motion


def photograph(canvas: np.ndarray, motion: np.ndarray, *, uneven_light: bool = False, hand: bool = False,
               seed: int = 0) -> bytes:
    """
    Render a canvas as a webcam would see it after the corner warp: moved, lit, noisy, JPEG-compressed, and as PNG bytes.
    """
    width, height = CANVAS_SIZE
    moved = cv2.warpAffine(canvas, motion, (width, height), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    # Darken one side, as a lamp beside the desk does, and lay a hand-like blob over the lower right.
    if uneven_light:
        moved *= np.linspace(0.75, 1.05, width, dtype=np.float32)[None, :]
    if hand:
        cv2.ellipse(moved, HAND_CENTER, (90, 60), 20, 0, 360, 110, -1)

    # Add sensor noise and JPEG blocks, then hand the result over as the PNG canvas the drawing stores.
    noisy = np.clip(moved + np.random.default_rng(seed).normal(0, 2, moved.shape), 0, 255).astype(np.uint8)
    jpeg = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
    return cv2.imencode(".png", cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE))[1].tobytes()


def measure_alignment_error(alignment: Alignment, motion: np.ndarray) -> float:
    """
    Return the largest distance in pixels between baseline points and their capture points mapped back by the warp.
    """
    baseline_points = np.array(CHECK_POINTS, dtype=np.float64)
    capture_points = baseline_points @ motion[:, :2].T + motion[:, 2]
    mapped = cv2.perspectiveTransform(capture_points[None], np.array(alignment.warp))[0]
    return float(np.linalg.norm(mapped - baseline_points, axis=1).max())


@pytest.fixture
def baseline_canvas() -> bytes:
    """
    Provide the baseline capture's canvas, photographed without camera motion.
    """
    return photograph(draw_canvas(), make_camera_motion((0, 0), 0), seed=1)


@pytest.mark.parametrize(("shift", "angle_degrees", "uneven_light"), [
    ((0, 0), 0.0, False),
    ((3, -2), 0.0, False),
    ((4, 3), 1.0, False),
    ((-3, 2), -1.5, True),
])
def test_alignment_recovers_camera_motion(baseline_canvas: bytes, shift: tuple[float, float], angle_degrees: float,
                                          uneven_light: bool) -> None:
    """
    A shifted, slightly rotated, or differently lit capture of the same drawing aligns within 2 px and has no new strokes.
    """
    motion = make_camera_motion(shift, angle_degrees)
    capture = photograph(draw_canvas(), motion, uneven_light=uneven_light, seed=2)
    alignment, new_stroke_pixels = AlignedStrokeFollowDetector().align(baseline_canvas, capture)

    # ECC converges with a strong correlation, and the warp maps the capture back onto the baseline.
    assert alignment.converged and alignment.correlation > 0.9, f"Expected a strong alignment, got {alignment}"
    error = measure_alignment_error(alignment, motion)
    assert error < 2.0, f"Warp misses the baseline by {error:.2f} px"

    # Nothing was drawn and nothing covers the paper.
    assert new_stroke_pixels < 60, f"An unchanged drawing must have almost no new strokes, got {new_stroke_pixels}"
    assert alignment.occluded_share == 0.0, f"Expected no occluder, got {alignment.occluded_share}"


def test_alignment_masks_hand_and_counts_new_strokes(baseline_canvas: bytes) -> None:
    """
    A hand over the paper is flagged as an occluded share and not counted as strokes, while a new leg is.
    """
    motion = make_camera_motion((2, 2), 0.5)
    detector = AlignedStrokeFollowDetector()
    with_hand, hand_pixels = detector.align(baseline_canvas, photograph(draw_canvas(), motion, hand=True, seed=3))
    with_leg, leg_pixels = detector.align(baseline_canvas, photograph(draw_canvas(True), motion, hand=True, seed=4))

    # The hand covers about a tenth of the canvas and does not steer the alignment.
    assert 0.05 < with_hand.occluded_share < 0.2, f"Expected the hand flagged, got share {with_hand.occluded_share}"
    assert with_hand.converged and measure_alignment_error(with_hand, motion) < 2.0, f"Hand broke alignment {with_hand}"

    # The hand alone adds no new strokes, and a new 70 px leg adds a few hundred pixels.
    assert hand_pixels < 60, f"A hand must not count as new strokes, got {hand_pixels}"
    assert 200 < leg_pixels < 1500, f"Expected the new leg counted, got {leg_pixels}"
    assert with_leg.converged, f"Expected the capture with a new leg to align, got {with_leg}"


@pytest.mark.parametrize("capture", [
    np.full(CANVAS_SIZE[::-1], PAPER_LEVEL, dtype=np.uint8),
    np.random.default_rng(5).normal(PAPER_LEVEL, 2, CANVAS_SIZE[::-1]).clip(0, 255).astype(np.uint8),
])
def test_alignment_failure_is_recorded(baseline_canvas: bytes, capture: np.ndarray) -> None:
    """
    A blank capture does not align, which is recorded with the reason instead of raised.
    """
    capture_canvas = cv2.imencode(".png", capture)[1].tobytes()
    alignment, new_stroke_pixels = AlignedStrokeFollowDetector().align(baseline_canvas, capture_canvas)

    # The alignment names why it failed and has no warp or new strokes.
    assert not alignment.converged and alignment.warp is None, f"Expected a failed alignment, got {alignment}"
    assert alignment.correlation.reason.startswith(NOT_CONVERGED_REASON), f"Unexpected reason {alignment.correlation}"
    assert new_stroke_pixels is None, f"A failed alignment counts no new strokes, got {new_stroke_pixels}"


def test_alignment_refuses_canvases_of_different_sizes(baseline_canvas: bytes) -> None:
    """
    Canvases of different sizes are a caller error, raised rather than recorded.
    """
    small = cv2.imencode(".png", np.full((64, 64), PAPER_LEVEL, dtype=np.uint8))[1].tobytes()
    with pytest.raises(ValueError, match="not the baseline's 512x360"):
        AlignedStrokeFollowDetector().align(baseline_canvas, small)


def test_find_occluders_ignores_strokes() -> None:
    """
    Thick strokes and a dense scribble are not occluders, so the drawing itself is never masked.
    """
    canvas = draw_canvas(True)
    cv2.rectangle(canvas, (20, 250), (120, 340), 40, 8)
    for offset in range(0, 60, 6):
        cv2.line(canvas, (400, 40 + offset), (490, 60 + offset), 40, 2)

    # No pixel is masked.
    assert not find_occluders(canvas).any(), "Strokes must not be taken for an occluder"


def test_detector_describes_its_method() -> None:
    """
    The detector's method record holds its name, version, and every threshold as plain values.
    """
    method = describe_follow_method(AlignedStrokeFollowDetector())

    # The name and version identify the method, and the thresholds include the ECC settings and stroke contrast.
    assert (method.name, method.version) == ALIGNED_STROKES_METHOD, f"Unexpected method {method}"
    expected = {"stroke_contrast", "ecc_iterations", "ecc_epsilon", "occluder_contrast", "baseline_grow_pixels"}
    assert expected <= set(method.params), f"Missing parameters {sorted(expected - set(method.params))}"


@pytest.mark.parametrize(("build", "message"), [
    (lambda: Alignment(converged=True, correlation=Unavailable("x"), warp=IDENTITY, occluded_share=0.0), "finite correlation"),
    (lambda: Alignment(converged=True, correlation=0.8, warp=None, occluded_share=0.0), "3x3 warp"),
    (lambda: Alignment(converged=True, correlation=0.8, warp=IDENTITY[:2], occluded_share=0.0), "3x3 warp"),
    (lambda: Alignment(converged=True, correlation=0.8, warp=IDENTITY, occluded_share=1.5), "from 0 to 1"),
    (lambda: Alignment(converged=False, correlation=0.8, warp=None, occluded_share=0.0), "has no correlation"),
    (lambda: Alignment(converged=False, correlation=Unavailable("x"), warp=IDENTITY, occluded_share=0.0), "has no warp"),
    (lambda: CaptureScores(suggestion=1, seconds_since_shown=1.0, new_stroke_pixels=None,
                           alignment=Alignment(converged=True, correlation=0.8, warp=IDENTITY, occluded_share=0.0)),
     "needs new_stroke_pixels"),
    (lambda: CaptureScores(suggestion=0, seconds_since_shown=1.0, new_stroke_pixels=None,
                           alignment=Alignment(converged=False, correlation=Unavailable("x"), warp=None, occluded_share=0.0)),
     "suggestion must be an int >= 1"),
])
def test_alignment_records_refuse_invalid_values(build: Callable[[], object], message: str) -> None:
    """
    Alignment and capture score records refuse values that contradict whether ECC converged.
    """
    with pytest.raises(InvalidRecordError, match=message):
        build()


def test_alignment_keeps_warp_as_float_rows() -> None:
    """
    A warp given as lists of ints, as JSON loads it, is kept as a tuple of float rows.
    """
    alignment = Alignment(converged=True, correlation=0.7, warp=[[1, 0, 2], [0, 1, -1], [0, 0, 1]], occluded_share=0.1)

    # The warp is immutable and holds floats.
    assert alignment.warp == ((1.0, 0.0, 2.0), (0.0, 1.0, -1.0), (0.0, 0.0, 1.0)), f"Unexpected warp {alignment.warp}"
    assert all(type(value) is float for row in alignment.warp for value in row), "Warp values must be floats"
