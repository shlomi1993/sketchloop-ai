"""
Tests of follow detection: darkness maps, occluders, ECC on synthetic captures, per-part scores, and their records.
"""

import cv2
import numpy as np
import pytest

from collections.abc import Callable
from typing import Final

from sketchloop.domain import InvalidRecordError, Unavailable
from sketchloop.follow import (ALIGNED_STROKES_METHOD, IMPLAUSIBLE_WARP_REASON, NO_PARTS_REASON, NO_USABLE_CAPTURE_REASON,
                               NOT_CONVERGED_REASON, NOTHING_NEW_REASON, PART_OCCLUDED_REASON, PROVISIONAL_RULE,
                               AlignedStrokeFollowDetector, Alignment, AlignmentRegion, CaptureScores, FollowDecision,
                               FollowResult, FollowRule, PartDecision, PartScores, align_to_baseline, choose_alignment_pixels,
                               decide_follow, describe_follow_method, find_implausible_warp, find_occluders, make_darkness_map)
from sketchloop.preprocessing import LOCAL_STROKE_CONTRAST


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
               seed: int = 0, hand_center: tuple[int, int] = HAND_CENTER) -> bytes:
    """
    Render a canvas as a webcam would see it after the corner warp: moved, lit, noisy, JPEG-compressed, and as PNG bytes.
    """
    width, height = CANVAS_SIZE
    moved = cv2.warpAffine(canvas, motion, (width, height), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    # Darken one side, as a lamp beside the desk does, and lay a hand-like blob over the lower right.
    if uneven_light:
        moved *= np.linspace(0.75, 1.05, width, dtype=np.float32)[None, :]
    if hand:
        cv2.ellipse(moved, hand_center, (90, 60), 20, 0, 360, 110, -1)

    # Add sensor noise and JPEG blocks, then hand the result over as the PNG canvas the drawing stores.
    noisy = np.clip(moved + np.random.default_rng(seed).normal(0, 2, moved.shape), 0, 255).astype(np.uint8)
    jpeg = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
    return cv2.imencode(".png", cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE))[1].tobytes()


def measure_warp_error(capture_to_baseline: np.ndarray, motion: np.ndarray) -> float:
    """
    Return the largest distance in pixels between baseline points and their capture points mapped back by the warp.
    """
    baseline_points = np.array(CHECK_POINTS, dtype=np.float64)
    capture_points = baseline_points @ motion[:, :2].T + motion[:, 2]
    mapped = cv2.perspectiveTransform(capture_points[None], capture_to_baseline)[0]
    return float(np.linalg.norm(mapped - baseline_points, axis=1).max())


def measure_alignment_error(alignment: Alignment, motion: np.ndarray) -> float:
    """
    Return the largest distance in pixels between baseline points and their capture points mapped back by the alignment.
    """
    return measure_warp_error(np.array(alignment.warp), motion)


@pytest.fixture
def baseline_canvas() -> bytes:
    """
    Provide the baseline capture's canvas, photographed without camera motion.
    """
    return photograph(draw_canvas(), make_camera_motion((0, 0), 0), seed=1)


@pytest.mark.parametrize(("shift", "angle_degrees", "uneven_light"), [
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
    expected = {"stroke_contrast", "ecc_iterations", "ecc_epsilon", "ecc_stroke_reach_pixels", "min_ecc_stroke_pixels",
                "occluder_contrast", "baseline_grow_pixels"}
    assert expected <= set(method.params), f"Missing parameters {sorted(expected - set(method.params))}"


def build_part_scores(**changes: object) -> PartScores:
    """
    Build valid scores of part 0 with new strokes near it, with the given fields changed.
    """
    fields = {
        "part": 0,
        "occluded_share": 0.0,
        "new_pixels_near": 120,
        "coverage": {4: 0.2, 8: 0.5, 16: 0.9, 24: 1.0},
        "precision": {4: 0.3, 8: 0.6, 16: 0.95, 24: 1.0},
        "chamfer_part_to_new": 6.5,
        "chamfer_new_to_part": 3.25,
    }
    return PartScores(**(fields | changes))


@pytest.mark.parametrize(("build", "message"), [
    (lambda: Alignment(converged=True, correlation=Unavailable("x"), warp=IDENTITY, occluded_share=0.0), "finite correlation"),
    (lambda: Alignment(converged=True, correlation=0.8, warp=None, occluded_share=0.0), "3x3 warp"),
    (lambda: Alignment(converged=True, correlation=0.8, warp=IDENTITY, occluded_share=1.5), "from 0 to 1"),
    (lambda: Alignment(converged=False, correlation=0.8, warp=None, occluded_share=0.0), "has no correlation"),
    (lambda: Alignment(converged=False, correlation=Unavailable("x"), warp=IDENTITY, occluded_share=0.0), "has no warp"),
    (lambda: Alignment(converged=False, correlation=Unavailable("x"), warp=None, occluded_share=0.1, filled_share=0.2),
     "filled_share must be at most"),
    (lambda: CaptureScores(suggestion=1, seconds_since_shown=1.0, new_stroke_pixels=None, parts=(),
                           alignment=Alignment(converged=True, correlation=0.8, warp=IDENTITY, occluded_share=0.0)),
     "needs new_stroke_pixels"),
    (lambda: CaptureScores(suggestion=1, seconds_since_shown=1.0, new_stroke_pixels=None, parts=(build_part_scores(),),
                           alignment=Alignment(converged=False, correlation=Unavailable("x"), warp=None, occluded_share=0.0)),
     "did not align has no part scores"),
    (lambda: CaptureScores(suggestion=1, seconds_since_shown=1.0, new_stroke_pixels=5, parts=(build_part_scores(part=1),),
                           alignment=Alignment(converged=True, correlation=0.8, warp=IDENTITY, occluded_share=0.0)),
     "numbered from 0"),
    (lambda: build_part_scores(new_pixels_near=0), "precision exactly when new strokes are near"),
    (lambda: build_part_scores(coverage={4: 0.5}), "same tolerances"),
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


PART_LINE: Final = ((340, 230), (340, 300))  # Suggested part: a leg under the right end of the seat


def encode_mask(mask: np.ndarray) -> bytes:
    """
    Encode a boolean mask as the PNG a part mask or filled mask is stored as, 255 where true.
    """
    return cv2.imencode(".png", np.where(mask, 255, 0).astype(np.uint8))[1].tobytes()


def draw_part_mask() -> bytes:
    """
    Draw the suggested leg as a 5 px thick part mask PNG at the canvas size.
    """
    mask = np.zeros(CANVAS_SIZE[::-1], dtype=np.uint8)
    cv2.line(mask, *PART_LINE, 255, 5)
    return encode_mask(mask > 0)


def draw_followed(canvas: np.ndarray) -> None:
    """
    Redraw the suggested leg 3 px to the right and a little short, as a person copying it from the screen would.
    """
    cv2.line(canvas, (343, 234), (343, 296), 40, 3)


def draw_elsewhere(canvas: np.ndarray) -> None:
    """
    Draw a line on the far left, well away from the suggested leg.
    """
    cv2.line(canvas, (60, 250), (60, 330), 40, 3)


def draw_different_shape(canvas: np.ndarray) -> None:
    """
    Draw a small circle where the suggested leg is, a different shape in the same place.
    """
    cv2.circle(canvas, (340, 265), 18, 40, 2)


def test_scores_a_followed_part(baseline_canvas: bytes) -> None:
    """
    A part redrawn with a small offset has high coverage and precision at every tolerance from 8 px on.
    """
    capture = draw_canvas()
    draw_followed(capture)
    alignment, new_stroke_pixels, parts = AlignedStrokeFollowDetector().score(
        baseline_canvas, photograph(capture, make_camera_motion((2, -2), 0.5), seed=6), [draw_part_mask()])
    part = parts[0]

    # The capture aligns, and the one part is scored with nothing hiding it.
    assert alignment.converged and len(parts) == 1, f"Expected one scored part, got {alignment} and {parts}"
    assert part.part == 0 and part.occluded_share == 0.0, f"Nothing hides the part, got {part}"
    assert new_stroke_pixels > 100 and part.new_pixels_near > 100, f"The new leg must count, got {part.new_pixels_near}"

    # The provisional rule's numbers are well met, and both chamfer distances are small.
    assert part.coverage[16] > 0.8 and part.precision[16] > 0.9, f"Expected a followed part, got {part}"
    assert part.coverage[4] <= part.coverage[8] <= part.coverage[16] <= part.coverage[24], f"Coverage must grow, got {part}"
    assert part.chamfer_part_to_new < 8 and part.chamfer_new_to_part < 8, f"Expected small chamfer distances, got {part}"


def test_scores_a_part_drawn_elsewhere(baseline_canvas: bytes) -> None:
    """
    Strokes far from the part leave it uncovered, with no new strokes near it and so no precision.
    """
    capture = draw_canvas()
    draw_elsewhere(capture)
    alignment, new_stroke_pixels, parts = AlignedStrokeFollowDetector().score(
        baseline_canvas, photograph(capture, make_camera_motion((1, 2), 0.0), seed=7), [draw_part_mask()])
    part = parts[0]

    # New strokes exist, but none near the part.
    assert alignment.converged and new_stroke_pixels > 100, f"The far line must count as new, got {new_stroke_pixels}"
    assert part.new_pixels_near == 0 and part.coverage[16] == 0.0, f"Expected an uncovered part, got {part}"

    # Precision and the distance from new strokes to the part are unavailable, while the part's distance is capped.
    assert all(isinstance(value, Unavailable) for value in part.precision.values()), f"Expected no precision, got {part}"
    assert isinstance(part.chamfer_new_to_part, Unavailable), f"Expected no new-to-part distance, got {part}"
    assert part.chamfer_part_to_new == 64, f"The part's distance must be capped at 64 px, got {part.chamfer_part_to_new}"


def test_scores_a_different_shape_in_the_same_place_as_covered(baseline_canvas: bytes) -> None:
    """
    A different shape drawn where the part is passes the provisional rule, since followed means the same place.
    """
    capture = draw_canvas()
    draw_different_shape(capture)
    _, _, parts = AlignedStrokeFollowDetector().score(baseline_canvas, photograph(capture, make_camera_motion((0, 0), 0), seed=8),
                                                      [draw_part_mask()])
    part = parts[0]

    # Coverage and precision at 16 px meet the provisional rule, while the tight 4 px coverage stays low.
    assert part.coverage[16] >= 0.35 and part.precision[16] >= 0.4, f"Expected the circle counted as covering, got {part}"
    assert part.coverage[4] < part.coverage[16], f"The circle must match the leg less closely at 4 px, got {part.coverage}"


def test_scores_a_part_under_a_hand_as_occluded(baseline_canvas: bytes) -> None:
    """
    A hand over the part hides it, so its coverage is unavailable rather than counted against it.
    """
    capture = photograph(draw_canvas(), make_camera_motion((1, 1), 0.0), hand=True, seed=9, hand_center=(340, 265))
    alignment, _, parts = AlignedStrokeFollowDetector().score(baseline_canvas, capture, [draw_part_mask()])
    part = parts[0]

    # The hand covers the whole part, and the capture still aligns.
    assert alignment.converged and alignment.occluded_share > 0.05, f"Expected the hand flagged, got {alignment}"
    assert part.occluded_share == 1.0, f"The whole part must be occluded, got {part.occluded_share}"

    # Coverage and the part's distance are unavailable with the reason, and nothing new is near it.
    reasons = {value.reason for value in part.coverage.values() if isinstance(value, Unavailable)}
    assert reasons == {PART_OCCLUDED_REASON} and len(part.coverage) == 4, f"Expected occluded coverage, got {part.coverage}"
    assert isinstance(part.chamfer_part_to_new, Unavailable) and part.new_pixels_near == 0, f"Unexpected scores {part}"


def test_filled_area_counts_as_occluded(baseline_canvas: bytes) -> None:
    """
    A part inside the area a canvas filled in, as under a hand reaching in from the frame edge, is occluded, not undrawn.
    """
    capture = draw_canvas()
    draw_followed(capture)
    filled = np.zeros(CANVAS_SIZE[::-1], dtype=bool)
    filled[222:, 300:] = True
    capture[filled] = PAPER_LEVEL
    capture_canvas = photograph(capture, make_camera_motion((0, 0), 0), seed=10)
    detector = AlignedStrokeFollowDetector()
    part_masks = [draw_part_mask()]
    with_mask, _, masked_parts = detector.score(baseline_canvas, capture_canvas, part_masks, capture_filled=encode_mask(filled))
    _, _, unmasked_parts = detector.score(baseline_canvas, capture_canvas, part_masks)

    # With the mask the part is wholly occluded and its coverage unknown, and the filled area is in the occluded share.
    part = masked_parts[0]
    assert part.occluded_share == 1.0 and isinstance(part.coverage[16], Unavailable), f"Expected an occluded part, got {part}"
    assert with_mask.converged and with_mask.occluded_share > 0.1, f"The filled area must be occluded, got {with_mask}"

    # Without the mask the same capture would wrongly count the part as not drawn.
    assert unmasked_parts[0].coverage[16] == 0.0, f"Without the mask the part looks undrawn, got {unmasked_parts[0]}"

    # The filled share is recorded apart, so the occluded share minus it is what a hand covered, none here.
    assert with_mask.filled_share == pytest.approx(filled.mean()), f"Expected the filled share apart, got {with_mask}"
    hand_share = with_mask.occluded_share - with_mask.filled_share
    assert hand_share == pytest.approx(0.0, abs=1e-9), f"No hand was in the capture, got a hand share {hand_share}"


def test_hand_is_told_apart_from_the_filled_rim(baseline_canvas: bytes) -> None:
    """
    With a filled rim and a hand inside the sheet, the filled share holds only the rim, and the hand is the rest.
    """
    rim = np.ones(CANVAS_SIZE[::-1], dtype=bool)
    rim[12:-12, 12:-12] = False
    capture = photograph(draw_canvas(), make_camera_motion((1, 0), 0.0), hand=True, seed=12, hand_center=(250, 200))
    alignment, _ = AlignedStrokeFollowDetector().align(baseline_canvas, capture, baseline_filled=encode_mask(rim),
                                                       capture_filled=encode_mask(rim))

    # The rim alone is the filled share, and the hand adds a clearly larger share on top of it.
    assert alignment.filled_share == pytest.approx(rim.mean()), f"Expected the rim as the filled share, got {alignment}"
    assert alignment.occluded_share - alignment.filled_share > 0.05, f"The hand must show beyond the rim, got {alignment}"


def test_part_mask_of_another_size_is_refused(baseline_canvas: bytes) -> None:
    """
    A part mask that is not the canvas size is a caller error, raised rather than recorded.
    """
    small = encode_mask(np.ones((64, 64), dtype=bool))
    with pytest.raises(ValueError, match="Part 0 mask is 64x64"):
        AlignedStrokeFollowDetector().score(baseline_canvas, baseline_canvas, [small])


def draw_hatching(canvas: np.ndarray) -> None:
    """
    Hatch the lower right of the canvas, a new drawing away from the existing strokes.
    """
    for offset in range(0, 80, 8):
        cv2.line(canvas, (380 + offset, 250), (420 + offset, 345), 40, 3)


def decode_canvas(payload: bytes) -> np.ndarray:
    """
    Decode a PNG canvas into a grayscale array.
    """
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def test_new_strokes_away_from_the_drawing_do_not_drag_the_alignment(baseline_canvas: bytes) -> None:
    """
    Hatching drawn away from the drawing, with a small camera shift and turn, aligns within 1 px, while whole-canvas ECC drifts.
    """
    motion = make_camera_motion((-4, 3), -1.0)
    capture = draw_canvas()
    draw_hatching(capture)
    capture_canvas = photograph(capture, motion, seed=13)
    alignment, new_stroke_pixels, parts = AlignedStrokeFollowDetector().score(baseline_canvas, capture_canvas, [draw_part_mask()])

    # ECC on the whole canvas, as version 2 aligned, is pulled more than 2 px off the camera motion by the hatching.
    everywhere = np.ones(CANVAS_SIZE[::-1], dtype=bool)
    darkness = [make_darkness_map(decode_canvas(canvas)) for canvas in (baseline_canvas, capture_canvas)]
    _, whole_canvas_warp = align_to_baseline(*darkness, everywhere, everywhere)
    whole_canvas_error = measure_warp_error(np.linalg.inv(whole_canvas_warp), motion)
    assert whole_canvas_error > 2.0, f"Expected whole-canvas ECC dragged by the hatching, got {whole_canvas_error:.2f} px"

    # Aligning near the baseline's strokes recovers the motion, records the region, and scores the hatching as new.
    assert alignment.converged and alignment.aligned_on == AlignmentRegion.BASELINE_STROKES, f"Unexpected alignment {alignment}"
    error = measure_alignment_error(alignment, motion)
    assert error < 1.0, f"Warp misses the baseline by {error:.2f} px"
    assert new_stroke_pixels > 1000 and len(parts) == 1, f"Expected the hatching counted and scored, got {new_stroke_pixels}"


@pytest.mark.parametrize("draw_baseline", [
    lambda canvas: None,
    lambda canvas: cv2.line(canvas, (240, 170), (270, 180), 40, 3),
])
def test_baseline_with_few_strokes_aligns_on_the_whole_canvas(draw_baseline: Callable[[np.ndarray], object]) -> None:
    """
    A blank baseline or one with a single dash has too few strokes to align on, so ECC uses the whole canvas and says so.
    """
    baseline = np.full(CANVAS_SIZE[::-1], PAPER_LEVEL, dtype=np.uint8)
    draw_baseline(baseline)
    capture = baseline.copy()
    draw_followed(capture)
    baseline_canvas = photograph(baseline, make_camera_motion((0, 0), 0), seed=14)
    capture_canvas = photograph(capture, make_camera_motion((2, 1), 0.0), seed=15)
    alignment, _ = AlignedStrokeFollowDetector().align(baseline_canvas, capture_canvas)

    # The fallback region is recorded whether or not ECC then converged.
    assert alignment.aligned_on == AlignmentRegion.WHOLE_CANVAS, f"Expected the whole-canvas fallback, got {alignment}"


def test_alignment_pixels_skip_hidden_areas_and_new_strokes() -> None:
    """
    ECC uses baseline pixels near its visible strokes and capture pixels away from new strokes, never a hidden pixel.
    """
    baseline, capture = draw_canvas(), draw_canvas()
    draw_hatching(capture)
    hidden = np.zeros(CANVAS_SIZE[::-1], dtype=bool)
    hidden[90:240, 140:220] = True
    capture_darkness = make_darkness_map(capture)
    region, baseline_pixels, capture_pixels = choose_alignment_pixels(make_darkness_map(baseline), capture_darkness, ~hidden,
                                                                      ~hidden)

    # Mark the hatching's strokes, the pixels of it darker than their local paper.
    hatching = np.zeros(CANVAS_SIZE[::-1], dtype=np.uint8)
    draw_hatching(hatching)
    hatching = (hatching > 0) & (capture_darkness > LOCAL_STROKE_CONTRAST)

    # Neither mask includes a hidden pixel, such as a hand or a filled-in rim.
    assert region == AlignmentRegion.BASELINE_STROKES, f"Expected to align near the strokes, got {region}"
    assert not (baseline_pixels & hidden).any() and not (capture_pixels & hidden).any(), "Hidden pixels must not steer ECC"

    # The baseline mask covers the box's visible edge and leaves out bare paper and where the hatching is drawn.
    assert baseline_pixels[100, 300] and not baseline_pixels[20, 480], "Expected the mask near the strokes only"
    assert not (baseline_pixels & hatching).any(), "The hatching area is far from the baseline strokes"

    # The capture mask keeps the old strokes and leaves out the new hatching's strokes.
    assert hatching.sum() > 500 and capture_pixels[100, 300], f"Expected hatching strokes, got {hatching.sum()} pixels"
    assert not (capture_pixels & hatching).any(), "New strokes must not steer ECC"


# The capture-to-baseline warp ECC reported for a webcam frame with a hand reaching in from the edge, a wrong solution.
BOGUS_WARP: Final = ((0.535, 0.021, 127.787), (0.331, 1.462, -164.789), (-0.001, 0.002, 1.0))


@pytest.mark.parametrize(("correlation", "baseline_to_capture", "plausible"), [
    (0.99, np.linalg.inv(np.array(BOGUS_WARP)), False),
    (0.99, np.vstack([make_camera_motion((3, -2), 1.0), [0, 0, 1]]), True),
    (0.99, np.vstack([make_camera_motion((40, 0), 0.0), [0, 0, 1]]), False),
    (0.3, np.eye(3), False),
])
def test_implausible_warps_are_refused(correlation: float, baseline_to_capture: np.ndarray, plausible: bool) -> None:
    """
    A warp that moves a canvas corner over 32 px, or a correlation under 0.4, is not a usable alignment.
    """
    reason = find_implausible_warp(correlation, baseline_to_capture, *CANVAS_SIZE)

    # A plausible warp has no reason, and an implausible one names why.
    if plausible:
        assert reason is None, f"Expected a plausible warp, got {reason}"
    else:
        assert reason.reason.startswith(IMPLAUSIBLE_WARP_REASON), f"Expected an implausible warp, got {reason}"


def test_scaled_capture_is_recorded_as_not_converged(baseline_canvas: bytes) -> None:
    """
    A capture whose canvas is shrunk, as when the paper outline changes, does not align, which is recorded as failed.
    """
    shrink = cv2.getRotationMatrix2D((CANVAS_SIZE[0] / 2, CANVAS_SIZE[1] / 2), 0.0, 0.85)
    capture = photograph(draw_canvas(), shrink, seed=11)
    alignment, new_stroke_pixels, parts = AlignedStrokeFollowDetector().score(baseline_canvas, capture, [draw_part_mask()])

    # The alignment failed with a reason, and nothing was counted or scored.
    assert not alignment.converged and alignment.warp is None, f"Expected a refused alignment, got {alignment}"
    reason = alignment.correlation.reason
    assert reason.startswith((IMPLAUSIBLE_WARP_REASON, NOT_CONVERGED_REASON)), f"Unexpected reason {reason}"
    assert new_stroke_pixels is None and parts == (), f"A refused alignment scores nothing, got {new_stroke_pixels} and {parts}"


def score_window_capture(coverage_16: float | Unavailable = 0.6, precision_16: float | Unavailable = 0.7, *,
                         new_stroke_pixels: int = 300, correlation: float = 0.8, converged: bool = True,
                         occluded_share: float = 0.0, part_count: int = 1) -> CaptureScores:
    """
    Build a capture's scores against suggestion 1 with the given 16 px coverage and precision for every part.
    """
    if not converged:
        alignment = Alignment(converged=False, correlation=Unavailable(NOT_CONVERGED_REASON), warp=None, occluded_share=0.0)
        return CaptureScores(suggestion=1, seconds_since_shown=2.0, alignment=alignment, new_stroke_pixels=None, parts=())

    # Nothing near the part makes precision and the distance from the new strokes unavailable.
    near = not isinstance(precision_16, Unavailable)
    precision = {4: 0.1, 8: 0.2, 16: precision_16, 24: 0.9} if near else dict.fromkeys((4, 8, 16, 24), precision_16)
    coverage = {4: 0.1, 8: 0.2, 16: coverage_16, 24: 0.9}
    parts = tuple(build_part_scores(part=index, occluded_share=occluded_share, new_pixels_near=120 if near else 0,
                                    coverage=coverage, precision=precision, chamfer_new_to_part=3.0 if near else precision_16)
                  for index in range(part_count))
    alignment = Alignment(converged=True, correlation=correlation, warp=IDENTITY, occluded_share=occluded_share)
    return CaptureScores(suggestion=1, seconds_since_shown=2.0, alignment=alignment, new_stroke_pixels=new_stroke_pixels,
                         parts=parts)


@pytest.mark.parametrize(("window", "result", "deciding_capture", "reason"), [
    ([(3, score_window_capture())], FollowResult.FOLLOWED, 3, None),
    ([(3, score_window_capture(coverage_16=0.2))], FollowResult.NOT_FOLLOWED, 3, None),
    ([(3, score_window_capture(precision_16=0.1))], FollowResult.NOT_FOLLOWED, 3, None),
    ([(3, score_window_capture(precision_16=Unavailable("none near")))], FollowResult.NOT_FOLLOWED, 3, None),
    ([(3, score_window_capture(new_stroke_pixels=10))], FollowResult.UNDETERMINED, 3, NOTHING_NEW_REASON),
    ([], FollowResult.UNDETERMINED, None, NO_USABLE_CAPTURE_REASON),
    ([(3, score_window_capture(correlation=0.2))], FollowResult.UNDETERMINED, None, NO_USABLE_CAPTURE_REASON),
    ([(3, score_window_capture(coverage_16=0.1)), (4, score_window_capture(occluded_share=0.8))],
     FollowResult.NOT_FOLLOWED, 3, None),
    ([(4, score_window_capture(converged=False)), (3, score_window_capture())], FollowResult.FOLLOWED, 3, None),
    ([(3, score_window_capture(coverage_16=0.1)), (4, score_window_capture())], FollowResult.FOLLOWED, 4, None),
], ids=["followed", "low-coverage", "low-precision", "nothing-near", "nothing-new", "no-capture", "weak-correlation",
        "occluded-latest-uses-earlier", "unaligned-latest-uses-earlier", "latest-decides"])
def test_decide_follow_outcomes(window: list[tuple[int, CaptureScores]], result: FollowResult, deciding_capture: int | None,
                                reason: str | None) -> None:
    """
    Each part is decided by its latest usable capture, and unusable captures or nothing drawn give undetermined.
    """
    decision = decide_follow(window, 1, PROVISIONAL_RULE, ALIGNED_STROKES_METHOD)
    part = decision.parts[0]

    # The part's result, deciding capture, and reason, and the suggestion's result, follow the rule.
    assert (part.result, part.deciding_capture, part.reason) == (result, deciding_capture, reason), f"Unexpected {part}"
    assert decision.result == result and decision.rule == PROVISIONAL_RULE, f"Unexpected decision {decision}"
    assert (decision.method, decision.method_version) == ALIGNED_STROKES_METHOD, f"Unexpected method {decision}"


def replace_part(part: PartScores, **changes: object) -> PartScores:
    """
    Copy part scores with the given fields changed.
    """
    fields = {name: getattr(part, name) for name in PartScores.__dataclass_fields__}
    return PartScores(**(fields | changes))


@pytest.mark.parametrize(("part_results", "result"), [
    ((FollowResult.FOLLOWED, FollowResult.NOT_FOLLOWED, FollowResult.UNDETERMINED), FollowResult.FOLLOWED),
    ((FollowResult.UNDETERMINED, FollowResult.NOT_FOLLOWED), FollowResult.NOT_FOLLOWED),
    ((FollowResult.UNDETERMINED, FollowResult.UNDETERMINED), FollowResult.UNDETERMINED),
])
def test_suggestion_result_combines_its_parts(part_results: tuple[FollowResult, ...], result: FollowResult) -> None:
    """
    A suggestion is followed when any part is, otherwise not followed when any part is not, otherwise undetermined.
    """
    coverages = {FollowResult.FOLLOWED: 0.9, FollowResult.NOT_FOLLOWED: 0.0, FollowResult.UNDETERMINED: 0.9}
    occlusion = {FollowResult.UNDETERMINED: 0.9}
    window = []
    for capture, part_result in enumerate(part_results, start=3):
        # Each capture scores all parts, and only its own part is visible and decisive there.
        scores = score_window_capture(coverage_16=coverages[part_result], part_count=len(part_results))
        hidden = [replace_part(part, occluded_share=1.0) for part in scores.parts]
        visible = replace_part(scores.parts[capture - 3], occluded_share=occlusion.get(part_result, 0.0))
        hidden[capture - 3] = visible
        window.append((capture, CaptureScores(suggestion=1, seconds_since_shown=2.0, alignment=scores.alignment,
                                              new_stroke_pixels=300, parts=tuple(hidden))))
    decision = decide_follow(window, len(part_results), PROVISIONAL_RULE, ALIGNED_STROKES_METHOD)

    # Each part gets its own result, and the suggestion's result combines them.
    assert tuple(part.result for part in decision.parts) == part_results, f"Unexpected part results {decision.parts}"
    assert decision.result == result, f"Expected {result}, got {decision.result}"


def test_suggestion_without_parts_is_undetermined() -> None:
    """
    A suggestion with no parts has nothing to follow, so it is undetermined with its own reason.
    """
    decision = decide_follow([(3, score_window_capture(part_count=0))], 0, PROVISIONAL_RULE, ALIGNED_STROKES_METHOD)

    # No part decisions, an undetermined result, and the reason.
    assert (decision.result, decision.parts, decision.reason) == (FollowResult.UNDETERMINED, (), NO_PARTS_REASON), f"{decision}"


def test_decide_follow_recomputes_with_another_rule() -> None:
    """
    The same stored scores decided with a stricter rule change the result, without reading any image.
    """
    window = [(3, score_window_capture(coverage_16=0.5, precision_16=0.5))]
    strict = FollowRule(min_coverage=0.6)
    loose_tolerance = FollowRule(coverage_tolerance_pixels=24, precision_tolerance_pixels=24, min_coverage=0.85)

    # The provisional rule follows, the stricter one does not, and each decision keeps the rule it used.
    provisional = decide_follow(window, 1, PROVISIONAL_RULE, ALIGNED_STROKES_METHOD)
    stricter = decide_follow(window, 1, strict, ALIGNED_STROKES_METHOD)
    assert provisional.result == FollowResult.FOLLOWED, f"Expected followed by the provisional rule, got {provisional}"
    assert stricter.result == FollowResult.NOT_FOLLOWED and stricter.rule == strict, f"Expected not followed, got {stricter}"

    # Another recorded tolerance works, and one never measured is refused.
    assert decide_follow(window, 1, loose_tolerance, ALIGNED_STROKES_METHOD).result == FollowResult.FOLLOWED, "24 px must work"
    with pytest.raises(ValueError, match="tolerance of 12 px was not measured"):
        decide_follow(window, 1, FollowRule(coverage_tolerance_pixels=12), ALIGNED_STROKES_METHOD)


@pytest.mark.parametrize(("build", "message"), [
    (lambda: FollowRule(min_coverage=1.5), "min_coverage must be a number from 0 to 1"),
    (lambda: PartDecision(part=0, result=FollowResult.FOLLOWED, deciding_capture=None, reason=None), "without a deciding"),
    (lambda: PartDecision(part=0, result=FollowResult.UNDETERMINED, deciding_capture=None, reason=None), "reason exactly"),
    (lambda: FollowDecision(method="aligned-strokes", method_version="2", rule=PROVISIONAL_RULE,
                            result=FollowResult.NOT_FOLLOWED, parts=(PartDecision(part=0, result=FollowResult.FOLLOWED,
                                                                                  deciding_capture=3, reason=None),)),
     "must combine its parts into followed"),
    (lambda: FollowDecision(method="aligned-strokes", method_version="2", rule=PROVISIONAL_RULE,
                            result=FollowResult.UNDETERMINED, parts=()), "reason must be 'no parts suggested'"),
])
def test_decision_records_refuse_invalid_values(build: Callable[[], object], message: str) -> None:
    """
    Rules and decisions refuse values outside their ranges and results that contradict their parts.
    """
    with pytest.raises(InvalidRecordError, match=message):
        build()
