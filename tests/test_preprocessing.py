"""
Tests of the preprocessing steps on synthetic sketch photos.
"""

import cv2
import numpy as np
import pytest

from sketchloop.capture import InvalidSketchError
from sketchloop.preprocessing import extract_canvas_strokes, make_paper_canvas, preprocess_minimally, preprocess_sketch


# A4 in landscape at about 1 px per 0.42 mm, and the default guided sketching canvas.
SHEET_SIZE = (707, 500)
CANVAS_SIZE = (512, 360)


def make_noisy_scene() -> np.ndarray:
    """
    Build an upright webcam-like scene with a checkered tablecloth, paper running off the right edge, a hand, and a faint house.
    """
    rng = np.random.default_rng(0)
    rows, columns = np.indices((480, 640))
    scene = np.where((rows // 20 + columns // 20) % 2 == 0, 40, 90) + rng.integers(-10, 11, (480, 640))
    scene[60:420, 150:] = 220 + rng.integers(-4, 5, (360, 490))
    scene = scene.astype(np.uint8)
    cv2.ellipse(scene, (150, 240), (60, 35), 0, 0, 360, 60, -1)
    house = np.array([[330, 200], [400, 130], [470, 200], [470, 330], [330, 330]], dtype=np.int32)
    cv2.polylines(scene, [house], True, 170, 1)
    return scene


def test_preprocessing_crops_to_strokes_resizes_and_records_steps() -> None:
    """
    A page with one off-center rectangle is cropped around it and resized to 512 px, while minimal preprocessing keeps it whole.
    """
    page = np.full((600, 900), 210, dtype=np.uint8)
    cv2.rectangle(page, (500, 100), (700, 200), 30, 4)
    _, encoded = cv2.imencode(".png", page)

    result = preprocess_sketch(encoded.tobytes())
    processed = cv2.imdecode(np.frombuffer(result.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    # Every preprocessing step is recorded in order.
    step_names = [step.name for step in result.steps]
    expected_steps = ["grayscale", "rotate", "isolate_paper", "extract_strokes", "crop_to_drawing", "thicken_strokes", "resize"]
    assert step_names == expected_steps, f"Unexpected steps {step_names}"

    # The crop starts just above and left of the rectangle.
    crop = result.steps[4].params
    crop_hugs_strokes = crop["found"] and 480 <= crop["x"] < 500 and 80 <= crop["y"] < 100
    assert crop_hugs_strokes, f"Crop must hug the strokes, got {crop}"

    # The output is 512 px on its longer side and matches the recorded size.
    size_matches_record = processed.shape == (result.image.height, result.image.width) and max(processed.shape) == 512
    assert size_matches_record, f"Expected a 512 px grayscale image, got {processed.shape}"

    # The image reference names a grayscale PNG.
    image_ref_is_grayscale_png = result.image.path == "sketch.png" and result.image.mode == "L"
    assert image_ref_is_grayscale_png, f"Unexpected image ref {result.image}"

    # The minimal preprocessor records only grayscale, rotate, and resize, with the rotation applied.
    minimal = preprocess_minimally(encoded.tobytes(), rotation=90)
    step_names = [step.name for step in minimal.steps]
    assert step_names == ["grayscale", "rotate", "resize"], f"Unexpected steps {step_names}"
    assert minimal.steps[1].params == {"degrees_clockwise": 90}, f"Unexpected rotation {minimal.steps[1].params}"

    # The whole page is kept, turned upright, so the long side is now the height.
    turned = cv2.imdecode(np.frombuffer(minimal.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert turned.shape == (512, 341), f"Expected the uncropped page turned and resized, got {turned.shape}"


@pytest.mark.parametrize(("angle", "center", "corners_found", "tilt", "aspect_ratio"), [
    (60, (400, 300), True, -30, 1 / 1.8),
    (-25, (180, 320), False, -25, 1.8)
], ids=["visible-60", "partly-visible-minus-25"])
def test_tilted_sheet_straightened(angle: float, center: tuple[int, int], corners_found: bool, tilt: float,
                                   aspect_ratio: float) -> None:
    """
    A tilted sheet is turned by the smallest angle that levels it, by its corners or, when partly off-frame, by its edges.
    """
    # A light 500 x 300 sheet on a dark table with a 160 x 80 rectangle drawn on it, both turned clockwise by the angle.
    photo = np.full((600, 800), 40, dtype=np.uint8)
    cv2.fillConvexPoly(photo, cv2.boxPoints((center, (500, 300), angle)).astype(np.int32), 220)
    cv2.polylines(photo, [cv2.boxPoints((center, (160, 80), angle)).astype(np.int32)], True, 60, 3)
    _, encoded = cv2.imencode(".png", photo)

    result = preprocess_sketch(encoded.tobytes())
    paper, crop = result.steps[2].params, result.steps[4].params

    # The tilt is found from the corners when all are visible, and from the visible edges otherwise.
    assert paper["corners_found"] == corners_found, f"Expected corners_found {corners_found}, got {dict(paper)}"
    assert abs(paper["tilt_degrees"] - tilt) <= 1.5, f"Expected a tilt near {tilt}, got {paper['tilt_degrees']}"

    # The drawn rectangle comes out level, so its crop has the rectangle's proportions, turned upright at 60 degrees.
    crop_ratio = crop["width"] / crop["height"]
    assert abs(crop_ratio - aspect_ratio) < 0.15, f"Expected a crop aspect near {aspect_ratio:.2f}, got {crop_ratio:.2f}"


def test_noisy_capture() -> None:
    """
    An upside-down noisy capture yields only the upright, faint house drawing, without the hand or the tablecloth.
    """
    _, encoded = cv2.imencode(".png", np.rot90(make_noisy_scene(), 2))

    result = preprocess_sketch(encoded.tobytes(), rotation=180)
    processed = cv2.imdecode(np.frombuffer(result.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    dark_rows, dark_columns = np.nonzero(processed < 128)

    # The 180 degree rotation is recorded as the second step.
    rotate_recorded = result.steps[1].name == "rotate" and result.steps[1].params["degrees_clockwise"] == 180
    assert rotate_recorded, f"The rotation must be recorded second, got {result.steps}"

    # The crop frames only the house, judged by its aspect ratio.
    house_aspect_ratio = 141 / 201
    aspect_ratio = result.image.width / result.image.height
    frames_only_house = abs(aspect_ratio - house_aspect_ratio) < 0.1
    assert frames_only_house, f"The crop must frame only the house, got aspect ratio {aspect_ratio:.2f}"

    # The roof is on top, so the drawing is upright.
    roof_columns = dark_columns[dark_rows == dark_rows.min()]
    floor_columns = dark_columns[dark_rows == dark_rows.max()]
    is_upright = abs(roof_columns.mean() - result.image.width / 2) < 30 and np.ptp(floor_columns) > result.image.width / 2
    assert is_upright, f"The roof must be on top, got top row columns {roof_columns.min()}-{roof_columns.max()}"

    # Few dark pixels remain, so the hand and tablecloth are gone.
    dark_share = len(dark_rows) / processed.size
    assert dark_share < 0.15, f"Only thin strokes may remain, not the hand, got {dark_share:.0%} dark pixels"


def draw_flat_sheet() -> np.ndarray:
    """
    Draw a light A4 sheet in landscape with a rectangle in the middle and a short stroke far from it in a corner.
    """
    sheet = np.full((SHEET_SIZE[1], SHEET_SIZE[0]), 215, dtype=np.uint8)
    cv2.rectangle(sheet, (250, 180), (450, 300), 50, 4)
    cv2.line(sheet, (90, 400), (150, 400), 50, 4)
    return sheet


def photograph_sheet(corners: list[tuple[int, int]]) -> bytes:
    """
    Warp the flat sheet onto the given corners of an 800 x 600 dark table photo and encode it as PNG.
    """
    width, height = SHEET_SIZE
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, np.array(corners, dtype=np.float32))
    photo = cv2.warpPerspective(draw_flat_sheet(), transform, (800, 600), borderValue=40)
    return cv2.imencode(".png", photo)[1].tobytes()


def decode_stroke_mask(payload: bytes) -> np.ndarray:
    """
    Decode a black-on-white strokes PNG to a mask that is True on the strokes.
    """
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) < 128


def decode_filled_mask(payload: bytes) -> np.ndarray:
    """
    Decode a canvas's filled mask PNG to a mask that is True where the canvas was filled in.
    """
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 0


def test_paper_canvas_keeps_shifted_captures_in_one_frame() -> None:
    """
    Two photos of a skewed sheet, moved a few pixels apart, give canvases of the configured size with aligned strokes.
    """
    first = make_paper_canvas(photograph_sheet([(150, 110), (660, 80), (690, 480), (120, 500)]), 0, *CANVAS_SIZE)
    second = make_paper_canvas(photograph_sheet([(153, 107), (657, 84), (693, 477), (118, 503)]), 0, *CANVAS_SIZE)

    # Both canvases are grayscale PNGs of exactly the configured size, whatever the sheet's size in the photo.
    for canvas in (first, second):
        pixels = cv2.imdecode(np.frombuffer(canvas.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert pixels.shape == (360, 512), f"Expected a 512x360 canvas, got {pixels.shape}"
        image = canvas.image
        assert (image.path, image.mode, image.width, image.height) == ("canvas.png", "L", 512, 360), f"Unexpected {image}"

    # The steps name the corner warp and the resize, without a crop, since all four corners were found.
    step_names = [step.name for step in first.steps]
    assert step_names == ["grayscale", "rotate", "isolate_paper", "resize_to_canvas"], f"Unexpected steps {step_names}"
    assert first.steps[2].params["corners_found"], f"The corners must be found, got {dict(first.steps[2].params)}"

    # Every stroke of the second capture lies within 3 px of a stroke of the first.
    first_strokes = decode_stroke_mask(extract_canvas_strokes(first.payload).payload)
    second_strokes = decode_stroke_mask(extract_canvas_strokes(second.payload).payload)
    distance_to_first = cv2.distanceTransform(np.where(first_strokes, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    worst = float(distance_to_first[second_strokes].max())
    assert second_strokes.any() and worst <= 3, f"Strokes must land within 3 px between captures, got {worst:.1f} px"

    # The sheet warped by its corners is seen everywhere except a thin filled rim at the canvas border.
    filled = decode_filled_mask(first.filled_payload)
    inner = np.zeros_like(filled)
    inner[20:-20, 20:-20] = True
    assert not (filled & inner).any() and filled[0, 0] and filled[-1, -1], "Only a thin rim at the border may be filled"


def test_paper_canvas_without_corners_crops_to_the_paper_and_fills_a_hand() -> None:
    """
    A sheet cut by the frame is cropped to the paper's box, and a hand reaching in from the edge is filled and marked.
    """
    photo = np.full((600, 800), 40, dtype=np.uint8)
    photo[80:520, 200:] = 215
    cv2.rectangle(photo, (300, 150), (500, 300), 50, 4)
    cv2.rectangle(photo, (600, 380), (800, 450), 70, -1)
    canvas = make_paper_canvas(cv2.imencode(".png", photo)[1].tobytes(), 0, *CANVAS_SIZE)
    pixels = cv2.imdecode(np.frombuffer(canvas.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    filled = decode_filled_mask(canvas.filled_payload)

    # The missing corners are recorded, and the crop to the paper's box comes before the resize.
    step_names = [step.name for step in canvas.steps]
    assert step_names == ["grayscale", "rotate", "isolate_paper", "crop_to_paper", "resize_to_canvas"], f"Got {step_names}"
    assert not canvas.steps[2].params["corners_found"], "The corners of a sheet cut by the frame must not be found"

    # The crop starts inside the paper, not at the table, and the canvas and its mask have the requested size.
    crop = canvas.steps[3].params
    assert crop["x"] >= 200 and crop["y"] >= 80, f"The crop must start inside the paper, got {dict(crop)}"
    assert pixels.shape == (360, 512), f"Expected a 512x360 canvas, got {pixels.shape}"
    image = canvas.filled
    assert (image.path, image.mode, image.width, image.height) == ("canvas-filled.png", "L", 512, 360), f"Unexpected {image}"

    # The hand, at the lower right of the paper's box, is filled, while the middle of the paper is seen.
    to_canvas = (CANVAS_SIZE[0] / crop["width"], CANVAS_SIZE[1] / crop["height"])
    hand = (round((700 - crop["x"]) * to_canvas[0]), round((415 - crop["y"]) * to_canvas[1]))
    middle = (round((400 - crop["x"]) * to_canvas[0]), round((225 - crop["y"]) * to_canvas[1]))
    assert filled[hand[1], hand[0]], f"The hand at {hand} must be marked as filled"
    assert not filled[middle[1], middle[0]], f"The paper's middle at {middle} must be seen"
    assert 0.03 < filled.mean() < 0.3, f"Expected the hand and the paper's rim filled, got share {filled.mean():.3f}"


def test_paper_canvas_without_paper_or_image() -> None:
    """
    A black frame gives a canvas of the requested size recording that no paper was found, and non-image bytes are refused.
    """
    canvas = make_paper_canvas(cv2.imencode(".png", np.zeros((480, 640), dtype=np.uint8))[1].tobytes(), 180, 512, 360)

    # The missing paper is recorded, and the size still matches.
    assert not canvas.steps[2].params["paper_found"], f"No paper must be recorded, got {dict(canvas.steps[2].params)}"
    assert (canvas.image.width, canvas.image.height) == (512, 360), f"Unexpected size {canvas.image}"

    # Bytes that are not an image raise InvalidSketchError for both the canvas and its strokes.
    with pytest.raises(InvalidSketchError):
        make_paper_canvas(b"not an image", 0, *CANVAS_SIZE)
    with pytest.raises(InvalidSketchError):
        extract_canvas_strokes(b"not an image")


def test_canvas_strokes_keep_strokes_far_from_the_drawing() -> None:
    """
    The canvas strokes keep a short stroke far from the main drawing, which the generation preprocessing drops.
    """
    photo = photograph_sheet([(150, 110), (660, 80), (690, 480), (120, 500)])
    strokes = extract_canvas_strokes(make_paper_canvas(photo, 0, *CANVAS_SIZE).payload)
    mask = decode_stroke_mask(strokes.payload)

    # The strokes image is a grayscale strokes.png of the canvas size, black on mostly white.
    image = strokes.image
    assert (image.path, image.mode, image.width, image.height) == ("strokes.png", "L", 512, 360), f"Unexpected {image}"
    assert 0 < mask.mean() < 0.1, f"Expected thin black strokes on white, got {mask.mean():.1%} dark"

    # The far stroke, near the sheet's lower left corner, survives, and the steps record it.
    far_corner = mask[int(360 * 0.7):, :int(512 * 0.3)]
    assert far_corner.any(), "The stroke far from the drawing must be kept"
    step_names = [step.name for step in strokes.steps]
    assert step_names == ["mark_strokes", "thicken_strokes"], f"Unexpected steps {step_names}"

    # The generation preprocessing drops that stroke as too far from the main drawing.
    dropped = preprocess_sketch(photo).steps[3].params["dropped_strokes"]
    assert dropped >= 1, f"Generation preprocessing must drop the far stroke, dropped {dropped}"


# A blank sheet's corners as photographed on a table, level and skewed.
BLANK_SHEET_CORNERS = {
    "untilted": [(100, 50), (700, 50), (700, 550), (100, 550)],
    "tilted": [(150, 110), (660, 80), (690, 480), (120, 500)],
}


def photograph_lit_sheet(corners: list[tuple[int, int]], draw_strokes: bool) -> bytes:
    """
    Photograph an A4 sheet, lit brighter on the left, on the given corners of a darker noisy table, encoded as JPEG.
    """
    width, height = SHEET_SIZE
    sheet = np.tile(np.linspace(225, 195, width), (height, 1)).astype(np.uint8)

    # Strokes in the middle, in the lower left corner, and on the dimmer right side.
    if draw_strokes:
        cv2.rectangle(sheet, (250, 180), (450, 300), 50, 4)
        cv2.line(sheet, (90, 400), (150, 400), 50, 4)
        cv2.line(sheet, (560, 120), (640, 120), 60, 4)

    # Warp the sheet onto the table, add camera noise, and compress it as a webcam would.
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, np.array(corners, dtype=np.float32))
    photo = cv2.warpPerspective(sheet, transform, (800, 600), borderValue=90).astype(np.float64)
    noisy = np.clip(photo + np.random.default_rng(0).normal(0, 2, photo.shape), 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


@pytest.mark.parametrize("corners", BLANK_SHEET_CORNERS.values(), ids=BLANK_SHEET_CORNERS.keys())
def test_lit_sheet_on_a_table_marks_only_its_strokes(corners: list[tuple[int, int]]) -> None:
    """
    A blank lit sheet on a darker table has no stroke pixels, and the same sheet with strokes keeps them and no border marks.
    """
    blank_canvas = make_paper_canvas(photograph_lit_sheet(corners, draw_strokes=False), 0, *CANVAS_SIZE)
    blank_strokes = extract_canvas_strokes(blank_canvas.payload)
    blank = decode_stroke_mask(blank_strokes.payload)
    drawn_canvas = make_paper_canvas(photograph_lit_sheet(corners, draw_strokes=True), 0, *CANVAS_SIZE)
    mask = decode_stroke_mask(extract_canvas_strokes(drawn_canvas.payload).payload)

    # No pixel of the blank sheet is marked, and the whole warped sheet was used with a paper-level fill outside it.
    assert not blank.any(), f"A blank sheet must have no strokes, got {int(blank.sum())} pixels"
    assert blank_strokes.steps[0].params["kept_strokes"] == 0, f"No stroke may be kept, got {blank_strokes.steps[0].params}"
    paper = blank_canvas.steps[2].params
    filled = paper["whole_warped_sheet"] and paper["outside_fill"] == "nearest_paper"
    assert filled, f"Expected the whole warped sheet with a paper-level fill, got {dict(paper)}"

    # Each drawn stroke, including the one on the dimmer side, has marked pixels around its place on the canvas.
    scale_x, scale_y = CANVAS_SIZE[0] / SHEET_SIZE[0], CANVAS_SIZE[1] / SHEET_SIZE[1]
    stroke_boxes = {"rectangle": (250, 180, 450, 300), "far line": (90, 390, 150, 410), "dim side line": (560, 110, 640, 130)}
    for name, (left, top, right, bottom) in stroke_boxes.items():
        region = mask[round(top * scale_y):round(bottom * scale_y), round(left * scale_x):round(right * scale_x)]
        assert region.any(), f"The {name} must be kept"

    # Nothing is marked within 10 px of the canvas border, where the table or a white fill would show.
    inner = np.zeros_like(mask)
    inner[10:-10, 10:-10] = True
    border_pixels = int((mask & ~inner).sum())
    assert border_pixels == 0, f"No strokes may be marked along the border, got {border_pixels} pixels"


def photograph_full_frame_sheet(with_hand: bool) -> bytes:
    """
    Photograph a sheet that fills the whole frame with a rectangle on it, optionally with a hand and arm from the lower right.
    """
    photo = np.full((600, 800), 215, dtype=np.uint8)
    cv2.rectangle(photo, (300, 150), (500, 300), 50, 4)
    if with_hand:
        cv2.ellipse(photo, (560, 450), (70, 45), 25, 0, 360, 110, -1)
        cv2.line(photo, (560, 450), (799, 599), 110, 48)

    return cv2.imencode(".png", photo)[1].tobytes()


def test_paper_canvas_ignores_a_hand_from_the_frame_edge_when_leveling() -> None:
    """
    A hand reaching in from the frame edge is not read as the sheet's edge, so the canvas is not turned and its strokes stay put.
    """
    plain = make_paper_canvas(photograph_full_frame_sheet(with_hand=False), 0, *CANVAS_SIZE)
    with_hand = make_paper_canvas(photograph_full_frame_sheet(with_hand=True), 0, *CANVAS_SIZE)

    # The sheet's outline is measured by its convex hull, which bridges the hand, so no tilt is removed.
    paper = with_hand.steps[2].params
    assert paper["tilt_degrees"] == 0.0, f"A hand must not tilt the canvas, got {paper['tilt_degrees']} degrees"
    assert paper["tilt_outline"] == "convex_hull", f"The canvas must record how its tilt was measured, got {dict(paper)}"

    # Every stroke of the capture with the hand lies within 3 px of a stroke of the capture without it.
    plain_strokes = decode_stroke_mask(extract_canvas_strokes(plain.payload).payload)
    hand_strokes = decode_stroke_mask(extract_canvas_strokes(with_hand.payload).payload)
    distance_to_plain = cv2.distanceTransform(np.where(plain_strokes, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    worst = float(distance_to_plain[hand_strokes].max())
    assert hand_strokes.any() and worst <= 3, f"Strokes must stay within 3 px with a hand in the frame, got {worst:.1f} px"

    # Generation preprocessing keeps its own leveling, which records no tilt outline, so generation sketches stay unchanged.
    generation_paper = preprocess_sketch(photograph_full_frame_sheet(with_hand=True)).steps[2].params
    assert "tilt_outline" not in generation_paper, f"Unexpected tilt outline in {dict(generation_paper)}"
