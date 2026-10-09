import cv2
import numpy as np

from sketchloop.preprocessing import preprocess_sketch


def test_preprocessing_crops_to_strokes_resizes_and_records_steps() -> None:
    """
    A light page with one dark off-center rectangle is cropped around it, resized to 512 px, and its steps recorded.
    """
    page = np.full((600, 900), 210, dtype=np.uint8)
    cv2.rectangle(page, (500, 100), (700, 200), 30, 4)
    _, encoded = cv2.imencode(".png", page)

    result = preprocess_sketch(encoded.tobytes())
    processed = cv2.imdecode(np.frombuffer(result.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    step_names = [step.name for step in result.steps]
    expected_steps = ["grayscale", "correct_perspective", "crop_to_drawing", "normalize_contrast", "resize"]
    assert step_names == expected_steps, f"Unexpected steps {step_names}"

    crop = result.steps[2].params
    crop_hugs_strokes = crop["found"] and 480 <= crop["x"] < 500 and 80 <= crop["y"] < 100
    assert crop_hugs_strokes, f"Crop must hug the strokes, got {crop}"

    size_matches_record = processed.shape == (result.image.height, result.image.width) and max(processed.shape) == 512
    assert size_matches_record, f"Expected a 512 px grayscale image, got {processed.shape}"

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
    perspective = result.steps[1].params

    found_paper = result.steps[1].name == "correct_perspective" and perspective["found"]
    assert found_paper, f"The paper outline must be found, got {dict(perspective)}"

    size_matches_edges = abs(perspective["width"] - 581) <= 6 and abs(perspective["height"] - 463) <= 6
    assert size_matches_edges, f"Output must span the paper's longer edges, got {dict(perspective)}"
