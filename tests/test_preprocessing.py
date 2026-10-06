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
    assert step_names == ["grayscale", "crop_to_drawing", "normalize_contrast", "resize"], f"Unexpected steps {step_names}"

    crop = result.steps[1].params
    crop_hugs_strokes = crop["found"] and 480 <= crop["x"] < 500 and 80 <= crop["y"] < 100
    assert crop_hugs_strokes, f"Crop must hug the strokes, got {crop}"

    size_matches_record = processed.shape == (result.image.height, result.image.width) and max(processed.shape) == 512
    assert size_matches_record, f"Expected a 512 px grayscale image, got {processed.shape}"

    image_ref_is_grayscale_png = result.image.path == "sketch.png" and result.image.mode == "L"
    assert image_ref_is_grayscale_png, f"Unexpected image ref {result.image}"
