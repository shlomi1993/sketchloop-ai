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

    # Crop to the dark strokes plus a margin, keeping the full image when no strokes stand out from the paper.
    cropped, crop_params = crop_to_drawing(gray)
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
