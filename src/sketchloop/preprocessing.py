import cv2
import hashlib
import numpy as np

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from sketchloop.capture import InvalidSketchError
from sketchloop.domain import ControlValue, ImageRef


# Generation conditioning treats pixels this much darker than the paper as strokes, and the crop keeps a margin around them.
STROKE_CONTRAST: Final = 64
CROP_MARGIN_RATIO: Final = 0.05
TARGET_LONG_SIDE_PIXELS: Final = 512

# A paper outline counts only if it covers this share of the image but does not fill it, which would mean no visible edge.
MIN_PAPER_AREA_RATIO: Final = 0.2
MAX_PAPER_AREA_RATIO: Final = 0.95

# Pixel sizes tuned on a 640 px webcam frame, scaled with the image's long side.
REFERENCE_LONG_SIDE_PIXELS: Final = 640
OPENING_PIXELS: Final = 9
PAPER_ERODE_PIXELS: Final = 25
PAPER_BLUR_PIXELS: Final = 31
MIN_STROKE_AREA_PIXELS: Final = 20
STROKE_REACH_PIXELS: Final = 40
THICKEN_PIXELS: Final = 3

# Pencil strokes are pixels this much darker than the surrounding paper.
LOCAL_STROKE_CONTRAST: Final = 12

# OpenCV rotation codes by clockwise angle in degrees.
ROTATE_CODES: Final = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


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


def odd_kernel_size(reference_pixels: int, scale: float) -> int:
    """
    Scale a pixel size tuned on the reference frame and round it to an odd kernel size of at least 1.

    Args:
        reference_pixels (int): Size in pixels on a frame with the reference long side.
        scale (float): Ratio of the image's long side to the reference long side.

    Returns:
        int: The scaled, odd kernel size.
    """
    return max(1, round(reference_pixels * scale)) | 1


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
    not_found: dict[str, ControlValue] = {"corners_found": False, "width": width, "height": height}
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
    return warped, {"corners_found": True, "corners": corner_text, "width": out_width, "height": out_height}


def isolate_paper(gray: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, dict[str, ControlValue]]:
    """
    Straighten the paper when its four corners are visible, then whiten everything outside its slightly shrunken outline.

    Args:
        gray (np.ndarray): Grayscale image.
        scale (float): Ratio of the image's long side to the reference long side.

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, ControlValue]]: The paper-only image, the paper mask, and the parameters used.
    """
    flattened, params = correct_perspective(gray)

    # Take the largest bright region without specks, which finds the paper even when an edge is outside the frame.
    _, bright = cv2.threshold(cv2.GaussianBlur(flattened, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening_size = odd_kernel_size(OPENING_PIXELS, scale)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((opening_size, opening_size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    paper_mask = np.zeros_like(flattened)

    # Fill the region's outline so dark strokes on the paper are not holes in it.
    if count > 1:
        paper_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        contours, _ = cv2.findContours((labels == paper_label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(paper_mask, contours, -1, 255, cv2.FILLED)

    # Shrink the outline, also from the frame edge, so the paper's own edge is not mistaken for a stroke.
    erode_size = odd_kernel_size(PAPER_ERODE_PIXELS, scale)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    paper_mask = cv2.erode(paper_mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    paper_found = bool(paper_mask.any())
    if not paper_found:
        paper_mask = np.full_like(flattened, 255)

    # Whiten everything outside the paper.
    paper_only = np.where(paper_mask > 0, flattened, 255).astype(np.uint8)
    paper_share = round(np.count_nonzero(paper_mask) / paper_mask.size, 3)
    params |= {"paper_found": paper_found, "erode_pixels": erode_size, "paper_share": paper_share}
    return paper_only, paper_mask, params


def is_within_reach(box: np.ndarray, anchor: np.ndarray, reach: int) -> bool:
    """
    Check whether a box, given as left, top, width, and height, comes within a distance of an anchor box.

    Args:
        box (np.ndarray): Box to check.
        anchor (np.ndarray): Box to measure from.
        reach (int): Allowed gap in pixels.

    Returns:
        bool: True when the box overlaps the anchor grown by the reach.
    """
    left, top, width, height = (int(value) for value in box)
    anchor_left, anchor_top, anchor_width, anchor_height = (int(value) for value in anchor)
    overlaps_horizontally = left < anchor_left + anchor_width + reach and left + width > anchor_left - reach
    overlaps_vertically = top < anchor_top + anchor_height + reach and top + height > anchor_top - reach
    return overlaps_horizontally and overlaps_vertically


def extract_strokes(paper_only: np.ndarray, paper_mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, dict[str, ControlValue]]:  # noqa: E501
    """
    Draw the strokes near the main drawing as black on white, or keep the paper image if none are found.

    Args:
        paper_only (np.ndarray): Grayscale image that is white outside the paper.
        paper_mask (np.ndarray): Paper mask, nonzero inside the paper.
        scale (float): Ratio of the image's long side to the reference long side.

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, ControlValue]]: The drawing, the stroke mask, and the parameters used.
    """
    # Mark pixels darker than the local paper brightness, which tolerates uneven light and faint pencil.
    blur_size = odd_kernel_size(PAPER_BLUR_PIXELS, scale)
    background = cv2.medianBlur(paper_only, blur_size)
    darker = ((background.astype(np.int16) - paper_only.astype(np.int16)) > LOCAL_STROKE_CONTRAST) & (paper_mask > 0)

    # Drop specks smaller than the minimum stroke area.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(darker.astype(np.uint8))
    min_area = max(1, round(MIN_STROKE_AREA_PIXELS * scale * scale))
    reach = max(1, round(STROKE_REACH_PIXELS * scale))
    large = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= min_area]
    params: dict[str, ControlValue] = {"contrast": LOCAL_STROKE_CONTRAST, "blur_pixels": blur_size, "min_area_pixels": min_area,
                                       "reach_pixels": reach}
    if not large:
        return paper_only, np.zeros_like(paper_only), params | {"found": False}

    # Keep strokes near the largest one, which drops shadows and paper-edge arcs far from the drawing.
    main = max(large, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    near = [label for label in large if is_within_reach(stats[label, :4], stats[main, :4], reach)]
    strokes = np.isin(labels, near).astype(np.uint8) * 255
    drawing = np.where(strokes > 0, 0, 255).astype(np.uint8)
    return drawing, strokes, params | {"found": True, "kept_strokes": len(near), "dropped_strokes": len(large) - len(near)}


def crop_to_drawing(drawing: np.ndarray, strokes: np.ndarray) -> tuple[np.ndarray, dict[str, ControlValue]]:
    """
    Crop the drawing to its strokes plus a small margin, or keep it whole if there are none.

    Args:
        drawing (np.ndarray): Grayscale drawing.
        strokes (np.ndarray): Stroke mask, nonzero on strokes.

    Returns:
        tuple[np.ndarray, dict[str, ControlValue]]: The cropped drawing and the crop parameters used.
    """
    stroke_points = cv2.findNonZero(strokes)
    height, width = drawing.shape
    if stroke_points is None:
        return drawing, {"found": False, "x": 0, "y": 0, "width": width, "height": height}

    # Grow the strokes' bounding box by the margin, clamped to the image.
    x, y, box_width, box_height = cv2.boundingRect(stroke_points)
    margin = round(CROP_MARGIN_RATIO * max(box_width, box_height))
    left, top = max(0, x - margin), max(0, y - margin)
    right, bottom = min(width, x + box_width + margin), min(height, y + box_height + margin)
    params: dict[str, ControlValue] = {"found": True, "x": left, "y": top,
                                       "width": right - left, "height": bottom - top, "margin_pixels": margin}
    return drawing[top:bottom, left:right], params


def preprocess_sketch(payload: bytes, rotation: int = 0) -> PreprocessedSketch:
    """
    Turn a raw sketch photo into upright, thickened black strokes on white, cropped and resized to 512 px.

    Args:
        payload (bytes): Encoded raw sketch image.
        rotation (int, optional): Clockwise rotation in degrees, one of 0, 90, 180, or 270. Defaults to 0.

    Returns:
        PreprocessedSketch: The processed image reference, PNG bytes, and applied steps.
    """
    if rotation not in ROTATE_CODES:
        raise ValueError(f"Rotation {rotation} is not supported. Use 0, 90, 180, or 270 degrees clockwise.")

    # Decode straight to grayscale, since strokes carry the drawing and color would only add noise.
    gray = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise InvalidSketchError("Sketch image cannot be decoded for preprocessing. Save it as PNG or JPEG and retry.")

    steps = [PreprocessingStep(name="grayscale", params={})]

    # Turn the photo upright before anything depends on orientation.
    rotate_code = ROTATE_CODES[rotation]
    rotated = gray if rotate_code is None else cv2.rotate(gray, rotate_code)
    steps.append(PreprocessingStep(name="rotate", params={"degrees_clockwise": rotation}))

    # Keep only the paper, then only the strokes that belong to the drawing.
    pixel_scale = max(rotated.shape) / REFERENCE_LONG_SIDE_PIXELS
    paper_only, paper_mask, paper_params = isolate_paper(rotated, pixel_scale)
    steps.append(PreprocessingStep(name="isolate_paper", params=paper_params))
    drawing, strokes, stroke_params = extract_strokes(paper_only, paper_mask, pixel_scale)
    steps.append(PreprocessingStep(name="extract_strokes", params=stroke_params))

    # Crop to the strokes plus a margin, keeping the full image when none were found.
    cropped, crop_params = crop_to_drawing(drawing, strokes)
    steps.append(PreprocessingStep(name="crop_to_drawing", params=crop_params))

    # Thicken thin pencil lines by growing the dark pixels.
    thicken_size = odd_kernel_size(THICKEN_PIXELS, pixel_scale)
    thickened = cv2.erode(cropped, np.ones((thicken_size, thicken_size), np.uint8))
    steps.append(PreprocessingStep(name="thicken_strokes", params={"kernel_pixels": thicken_size}))

    # Scale so the longer side is the target size, keeping the aspect ratio.
    height, width = thickened.shape
    scale = TARGET_LONG_SIDE_PIXELS / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation, interpolation_name = (cv2.INTER_AREA, "area") if scale < 1 else (cv2.INTER_CUBIC, "cubic")
    resized = cv2.resize(thickened, size, interpolation=interpolation)
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
