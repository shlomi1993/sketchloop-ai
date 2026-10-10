import cv2
import numpy as np

from sketchloop.preprocessing import preprocess_sketch


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
    A light page with one dark off-center rectangle is cropped around it, resized to 512 px, and its steps recorded.
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


def test_perspective_straightens_paper() -> None:
    """
    A skewed light sheet on a dark table is found and warped to an upright rectangle the size of its longer edges.
    """
    photo = np.full((600, 800), 40, dtype=np.uint8)
    corners = np.array([[150, 100], [650, 60], [700, 520], [120, 480]], dtype=np.int32)
    cv2.fillConvexPoly(photo, corners, 220)
    _, encoded = cv2.imencode(".png", photo)

    result = preprocess_sketch(encoded.tobytes())
    paper = result.steps[2].params

    # The paper step finds the sheet's corners.
    found_corners = result.steps[2].name == "isolate_paper" and paper["corners_found"]
    assert found_corners, f"The paper corners must be found, got {dict(paper)}"

    # The warped sheet spans its longer edges within 6 px.
    size_matches_edges = abs(paper["width"] - 581) <= 6 and abs(paper["height"] - 463) <= 6
    assert size_matches_edges, f"Output must span the paper's longer edges, got {dict(paper)}"


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
