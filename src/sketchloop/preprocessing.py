import cv2
import hashlib
import numpy as np

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from sketchloop.capture import InvalidSketchError
from sketchloop.domain import ControlValue, ImageRef


# Strokes are pixels this much darker than the paper, and the crop keeps a margin around them.
STROKE_CONTRAST: Final = 64
CROP_MARGIN_RATIO: Final = 0.05
TARGET_LONG_SIDE_PIXELS: Final = 512

# A paper outline counts only if it covers this share of the image but does not fill it, which would mean no visible edge.
MIN_PAPER_AREA_RATIO: Final = 0.2
MAX_PAPER_AREA_RATIO: Final = 0.95


@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessingStep:
    """
    One applied preprocessing operation and the parameters it actually used.
    """
    name: str
    params: Mapping[str, ControlValue]

    def __post_init__(self) -> None:
        # Copy into a read-only view so the recorded parameters cannot change after the fact.
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessedSketch:
    """
    The processed sketch image, its PNG bytes, and the ordered steps that produced it.
    """
    image: ImageRef
    payload: bytes
    steps: tuple[PreprocessingStep, ...]


def preprocess_sketch(payload: bytes) -> PreprocessedSketch:
    """
    Convert a raw sketch to a cropped, contrast-normalized, 512 px grayscale PNG and record each step.

    Args:
        payload (bytes): Encoded raw sketch image.

    Returns:
        PreprocessedSketch: The processed image reference, PNG bytes, and applied steps.
    """
    # Decode straight to grayscale, since strokes carry the drawing and color would only add noise.
    gray = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise InvalidSketchError("Sketch image cannot be decoded for preprocessing. Save it as PNG or JPEG and retry.")

    steps = [PreprocessingStep(name="grayscale", params={})]

    # Straighten a photographed sheet of paper into a rectangle, keeping the image when no paper outline is found.
    flattened, perspective_params = correct_perspective(gray)
    steps.append(PreprocessingStep(name="correct_perspective", params=perspective_params))

    # Crop to the dark strokes plus a margin, keeping the full image when no strokes stand out from the paper.
    cropped, crop_params = crop_to_drawing(flattened)
    steps.append(PreprocessingStep(name="crop_to_drawing", params=crop_params))

    # Stretch the remaining intensities to the full 0-255 range.
    input_min, input_max = int(cropped.min()), int(cropped.max())
    normalized: cv2.typing.MatLike = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX)
    steps.append(PreprocessingStep(name="normalize_contrast", params={"input_min": input_min, "input_max": input_max}))

    # Scale so the longer side is the target size, keeping the aspect ratio.
    height, width = normalized.shape
    scale = TARGET_LONG_SIDE_PIXELS / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation, interpolation_name = (cv2.INTER_AREA, "area") if scale < 1 else (cv2.INTER_CUBIC, "cubic")
    resized = cv2.resize(normalized, size, interpolation=interpolation)
    resize_params = {"width": size[0], "height": size[1], "interpolation": interpolation_name}
    steps.append(PreprocessingStep(name="resize", params=resize_params))

    # Encode the result as PNG and describe it for the run folder.
    encoded, buffer = cv2.imencode(".png", resized)
    if not encoded:
        raise InvalidSketchError("Preprocessed sketch cannot be encoded as PNG. Retry with --raw.")

    processed = buffer.tobytes()
    image = ImageRef(
        path="sketch.png",
        width=size[0],
        height=size[1],
        mode="L",
        media_type="image/png",
        sha256=hashlib.sha256(processed).hexdigest()
    )
    return PreprocessedSketch(image=image, payload=processed, steps=tuple(steps))


def correct_perspective(gray: np.ndarray) -> tuple[np.ndarray, dict[str, ControlValue]]:
    """
    Warp the largest light four-cornered region (the paper) to a rectangle, or keep the image if none is found.

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        tuple[np.ndarray, dict[str, ControlValue]]: The straightened image and the perspective parameters used.
    """
    # Separate the light paper from a darker background and take the largest outer outline.
    height, width = gray.shape
    _, paper_mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper = max(contours, key=cv2.contourArea, default=None)
    not_found: dict[str, ControlValue] = {"found": False, "width": width, "height": height}
    area_ratio = 0.0 if paper is None else cv2.contourArea(paper) / (width * height)
    if not MIN_PAPER_AREA_RATIO <= area_ratio <= MAX_PAPER_AREA_RATIO:
        return gray, not_found

    # Simplify the outline and accept it only as a convex quadrilateral.
    corners = cv2.approxPolyDP(paper, 0.02 * cv2.arcLength(paper, True), True)
    if len(corners) != 4 or not cv2.isContourConvex(corners):
        return gray, not_found

    # Order corners as top-left, top-right, bottom-right, bottom-left using coordinate sums and differences.
    points = corners.reshape(4, 2).astype(np.float32)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
    top_left, bottom_right = points[np.argmin(sums)], points[np.argmax(sums)]
    top_right, bottom_left = points[np.argmin(differences)], points[np.argmax(differences)]
    source = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    # Size the output by the longer of each pair of opposite edges, then warp the paper onto it.
    out_width = round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    out_height = round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    target = np.array([[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]], dtype=np.float32)
    warped = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(source, target), (out_width, out_height))
    corner_text = " ".join(f"{round(float(x))},{round(float(y))}" for x, y in source)
    return warped, {"found": True, "corners": corner_text, "width": out_width, "height": out_height}


def crop_to_drawing(gray: np.ndarray) -> tuple[np.ndarray, dict[str, ControlValue]]:
    """
    Crop a grayscale image to its dark strokes plus a small margin, or keep it whole if none are found.

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        tuple[np.ndarray, dict[str, ControlValue]]: The cropped image and the crop parameters used.
    """
    # Smooth paper noise, then mark pixels clearly darker than the typical (median) paper level.
    blurred = cv2.medianBlur(gray, 5)
    threshold = int(np.median(blurred)) - STROKE_CONTRAST
    stroke_points = cv2.findNonZero((blurred < threshold).astype(np.uint8))
    height, width = gray.shape
    if stroke_points is None:
        return gray, {"found": False, "threshold": threshold, "x": 0, "y": 0, "width": width, "height": height}

    # Grow the strokes' bounding box by the margin, clamped to the image.
    x, y, box_width, box_height = cv2.boundingRect(stroke_points)
    margin = round(CROP_MARGIN_RATIO * max(box_width, box_height))
    left, top = max(0, x - margin), max(0, y - margin)
    right, bottom = min(width, x + box_width + margin), min(height, y + box_height + margin)
    params: dict[str, ControlValue] = {"found": True, "threshold": threshold, "x": left, "y": top,
                                       "width": right - left, "height": bottom - top, "margin_pixels": margin}
    return gray[top:bottom, left:right], params
