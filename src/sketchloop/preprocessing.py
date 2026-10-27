"""
Sketch preprocessing that straightens, crops, normalizes, and resizes a photographed drawing.
"""

import cv2
import hashlib
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from sketchloop.capture import InvalidSketchError
from sketchloop.domain import ColorMode, ControlValue, ImageRef, Rotation


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

# A partly visible sheet is straightened only when it tilts more than this, so a level sheet is left untouched.
MIN_TILT_DEGREES: Final = 1.0

# Outline points this close to the image border are where the frame cuts a sheet, not the sheet's own corners or edges.
FRAME_EDGE_PIXELS: Final = 3

# OpenCV rotation codes by clockwise angle in degrees.
ROTATE_CODES: Final = {
    Rotation.DEG_000: None,
    Rotation.DEG_090: cv2.ROTATE_90_CLOCKWISE,
    Rotation.DEG_180: cv2.ROTATE_180,
    Rotation.DEG_270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


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
    """
    return max(1, round(reference_pixels * scale)) | 1


def threshold_bright(gray: np.ndarray) -> np.ndarray:
    """
    Separate light paper from a darker background with a blurred Otsu threshold, returning a mask that is 255 on bright pixels.
    """
    return cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def find_paper_corners(gray: np.ndarray) -> np.ndarray | None:
    """
    Find the four corners of the largest light region when it is a convex quadrilateral of a plausible size.

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray | None: The corners as a 4x2 float array, or None when the paper's corners are not all visible.
    """
    # Take the largest outer outline of the bright region and check that it is a plausible paper size.
    height, width = gray.shape
    contours = cv2.findContours(threshold_bright(gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    paper = max(contours, key=cv2.contourArea, default=None)
    area_ratio = 0.0 if paper is None else cv2.contourArea(paper) / (width * height)
    if not MIN_PAPER_AREA_RATIO <= area_ratio <= MAX_PAPER_AREA_RATIO:
        return None

    # Simplify the outline and accept it only as a convex quadrilateral.
    corners: np.ndarray = cv2.approxPolyDP(paper, 0.02 * cv2.arcLength(paper, True), True)
    if len(corners) != 4 or not cv2.isContourConvex(corners):
        return None

    # A corner on the frame edge is where the frame cuts the sheet, not a paper corner.
    corners = corners.reshape(4, 2).astype(np.float32)
    if is_on_frame_edge(corners, width, height).any():
        return None

    return corners


def is_on_frame_edge(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Mark which points lie within a few pixels of the image border, where the frame cuts a sheet.
    """
    x, y = points[:, 0], points[:, 1]
    return (x <= FRAME_EDGE_PIXELS) | (y <= FRAME_EDGE_PIXELS) | (x >= width - 1 - FRAME_EDGE_PIXELS) | (y >= height - 1 - FRAME_EDGE_PIXELS)  # noqa: E501


def list_outline_edges(outline: np.ndarray) -> np.ndarray:
    """
    Return the edge vectors of a closed outline, from each point to the next.
    """
    points = outline.reshape(-1, 2).astype(np.float64)
    return np.roll(points, -1, axis=0) - points


def measure_tilt_degrees(edges: np.ndarray) -> float:
    """
    Measure how far edges turn clockwise from level, between -45 and 45 degrees, weighting longer edges more.

    Args:
        edges (np.ndarray): Edge vectors as an Nx2 array, such as the sides of a sheet.

    Returns:
        float: The tilt in degrees, positive when the sheet is turned clockwise in the image.
    """
    # Quadruple each edge angle so edges 90 degrees apart agree, then average them as unit vectors.
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    angles = 4 * np.arctan2(edges[:, 1], edges[:, 0])
    return float(np.degrees(np.arctan2((lengths * np.sin(angles)).sum(), (lengths * np.cos(angles)).sum()) / 4))


def order_corners(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Order four corners as top-left, top-right, bottom-right, bottom-left, with the smallest turn from the camera's view.

    Args:
        points (np.ndarray): Four corner points as a 4x2 array.

    Returns:
        tuple[np.ndarray, float]: The ordered corners, and the sheet's tilt in degrees.
    """
    # Turn the corners level around their center, so sums and differences pick the corners even for a steep tilt.
    tilt = measure_tilt_degrees(list_outline_edges(points))
    radians = np.radians(tilt)
    offsets = points - points.mean(axis=0)
    level = np.column_stack([offsets[:, 0] * np.cos(radians) + offsets[:, 1] * np.sin(radians),
                             -offsets[:, 0] * np.sin(radians) + offsets[:, 1] * np.cos(radians)])

    # Pick each corner on the level points and return the original points in that order.
    sums, differences = level.sum(axis=1), np.diff(level, axis=1).ravel()
    order = [np.argmin(sums), np.argmin(differences), np.argmax(sums), np.argmax(differences)]
    return points[order].astype(np.float32), tilt


def correct_perspective(gray: np.ndarray) -> tuple[np.ndarray, dict[str, ControlValue]]:
    """
    Warp the largest light four-cornered region (the paper) to a rectangle, or keep the image if none is found.

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        tuple[np.ndarray, dict[str, ControlValue]]: The straightened image and the perspective parameters used.
    """
    height, width = gray.shape
    corners = find_paper_corners(gray)
    if corners is None:
        return gray, {"corners_found": False, "width": width, "height": height}

    # Size the output by the longer of each pair of opposite edges, then warp the paper onto it.
    source, tilt = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = source
    out_width = round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    out_height = round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    target = np.array([[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]], dtype=np.float32)
    warped = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(source, target), (out_width, out_height))
    corner_text = " ".join(f"{round(float(x))},{round(float(y))}" for x, y in source)
    params: dict[str, ControlValue] = {
        "corners_found": True,
        "corners": corner_text,
        "tilt_degrees": round(tilt, 1),
        "width": out_width,
        "height": out_height,
    }
    return warped, params


def find_paper_mask(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Fill the largest bright region without specks, which finds the paper even when an edge is outside the frame.

    Args:
        image (np.ndarray): Grayscale image.
        scale (float): Ratio of the image's long side to the reference long side.

    Returns:
        np.ndarray: Mask that is 255 inside the paper, or all zero when no bright region exists.
    """
    # Remove bright specks, then label the remaining bright regions.
    opening_size = odd_kernel_size(OPENING_PIXELS, scale)
    bright = cv2.morphologyEx(threshold_bright(image), cv2.MORPH_OPEN, np.ones((opening_size, opening_size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    paper_mask = np.zeros_like(image)

    # Fill the largest region's outline so dark strokes on the paper are not holes in it.
    if count > 1:
        paper_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        contours = cv2.findContours((labels == paper_label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(paper_mask, contours, -1, 255, cv2.FILLED)

    return paper_mask


def straighten_tilt(gray: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    """
    Turn the image so the largest bright region's edges are level, for a tilted sheet whose corners are not all visible.

    Args:
        gray (np.ndarray): Grayscale image.
        scale (float): Ratio of the image's long side to the reference long side.

    Returns:
        tuple[np.ndarray, float]: The straightened image, enlarged to keep every pixel, and the tilt it removed in degrees.
    """
    contours = cv2.findContours(find_paper_mask(gray, scale), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return gray, 0.0

    # Measure the tilt from the paper's own edges, skipping edges where the frame cuts the sheet, and leave a level sheet.
    height, width = gray.shape
    paper = max(contours, key=cv2.contourArea)
    outline = cv2.approxPolyDP(paper, 0.004 * cv2.arcLength(paper, True), True).reshape(-1, 2).astype(np.float64)
    on_frame = is_on_frame_edge(outline, width, height)
    inner_edges = list_outline_edges(outline)[~(on_frame & np.roll(on_frame, -1))]
    tilt = measure_tilt_degrees(inner_edges) if len(inner_edges) else 0.0
    if abs(tilt) < MIN_TILT_DEGREES:
        return gray, 0.0

    # Turn back by the tilt on a canvas large enough for the turned image, filling new corners dark like a table.
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), tilt, 1.0)
    cos, sin = abs(rotation[0, 0]), abs(rotation[0, 1])
    new_width, new_height = round(width * cos + height * sin), round(width * sin + height * cos)
    rotation[:, 2] += [(new_width - width) / 2, (new_height - height) / 2]
    straightened = cv2.warpAffine(gray, rotation, (new_width, new_height), flags=cv2.INTER_LINEAR, borderValue=0)
    return straightened, round(tilt, 1)


def isolate_paper(gray: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, dict[str, ControlValue]]:
    """
    Straighten the paper by its corners or, when some are hidden, by its edges, then whiten everything outside its outline.

    Args:
        gray (np.ndarray): Grayscale image.
        scale (float): Ratio of the image's long side to the reference long side.

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, ControlValue]]: The paper-only image, the paper mask, and the parameters used.
    """
    flattened, params = correct_perspective(gray)

    # Straighten a tilted sheet whose corners are not all visible, using its visible edges.
    if not params["corners_found"]:
        flattened, tilt = straighten_tilt(flattened, scale)
        params |= {"tilt_degrees": tilt, "width": flattened.shape[1], "height": flattened.shape[0]}

    # Shrink the outline, also from the frame edge, so the paper's own edge is not mistaken for a stroke.
    erode_size = odd_kernel_size(PAPER_ERODE_PIXELS, scale)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    paper_mask = cv2.erode(find_paper_mask(flattened, scale), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
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


def mark_darker_pixels(paper_only: np.ndarray, paper_mask: np.ndarray, blur_size: int) -> np.ndarray:
    """
    Mark paper pixels darker than the local paper brightness, which tolerates uneven light and faint pencil.
    """
    background = cv2.medianBlur(paper_only, blur_size)
    return ((background.astype(np.int16) - paper_only.astype(np.int16)) > LOCAL_STROKE_CONTRAST) & (paper_mask > 0)


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
    blur_size = odd_kernel_size(PAPER_BLUR_PIXELS, scale)
    darker = mark_darker_pixels(paper_only, paper_mask, blur_size)

    # Drop specks smaller than the minimum stroke area.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(darker.astype(np.uint8))
    min_area = max(1, round(MIN_STROKE_AREA_PIXELS * scale * scale))
    reach = max(1, round(STROKE_REACH_PIXELS * scale))
    large = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= min_area]
    params: dict[str, ControlValue] = {
        "contrast": LOCAL_STROKE_CONTRAST,
        "blur_pixels": blur_size,
        "min_area_pixels": min_area,
        "reach_pixels": reach
    }
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
    params: dict[str, ControlValue] = {
        "found": True,
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
        "margin_pixels": margin
    }
    return drawing[top:bottom, left:right], params


def decode_grayscale(payload: bytes) -> np.ndarray:
    """
    Decode an encoded image straight to grayscale, since strokes carry the drawing and color would only add noise.

    Args:
        payload (bytes): Encoded raw sketch image.

    Returns:
        np.ndarray: Grayscale image.
    """
    gray = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise InvalidSketchError("Sketch image cannot be decoded for preprocessing. Save it as PNG or JPEG and retry.")

    return gray


def rotate_image(gray: np.ndarray, rotation: int) -> np.ndarray:
    """
    Rotate an image clockwise by 0, 90, 180, or 270 degrees.
    """
    rotate_code = ROTATE_CODES[rotation]
    return gray if rotate_code is None else cv2.rotate(gray, rotate_code)


def thicken_strokes(drawing: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, ControlValue]]:
    """
    Thicken thin pencil lines by growing the dark pixels, returning the drawing and the kernel size used.
    """
    thicken_size = odd_kernel_size(THICKEN_PIXELS, scale)
    return cv2.erode(drawing, np.ones((thicken_size, thicken_size), np.uint8)), {"kernel_pixels": thicken_size}


def resize_long_side(image: np.ndarray) -> tuple[np.ndarray, dict[str, ControlValue]]:
    """
    Scale an image so its longer side is the target size, keeping the aspect ratio.

    Args:
        image (np.ndarray): Image to resize.

    Returns:
        tuple[np.ndarray, dict[str, ControlValue]]: The resized image and the size and interpolation used.
    """
    height, width = image.shape
    scale = TARGET_LONG_SIDE_PIXELS / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation, interpolation_name = (cv2.INTER_AREA, "area") if scale < 1 else (cv2.INTER_CUBIC, "cubic")
    resized = cv2.resize(image, size, interpolation=interpolation)
    return resized, {"width": size[0], "height": size[1], "interpolation": interpolation_name}


def encode_processed_sketch(image: np.ndarray) -> tuple[ImageRef, bytes]:
    """
    Encode the processed sketch as PNG and describe it as `sketch.png` in the run folder.

    Args:
        image (np.ndarray): Processed grayscale sketch.

    Returns:
        tuple[ImageRef, bytes]: The image reference and the PNG bytes.
    """
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise InvalidSketchError("Preprocessed sketch cannot be encoded as PNG. Retry with --raw.")

    payload = buffer.tobytes()
    height, width = image.shape
    image_ref = ImageRef(
        path="sketch.png",
        width=width,
        height=height,
        mode=ColorMode.L,
        media_type="image/png",
        sha256=hashlib.sha256(payload).hexdigest()
    )
    return image_ref, payload


def preprocess_sketch(payload: bytes, rotation: int = Rotation.DEG_000) -> PreprocessedSketch:
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

    # Decode to grayscale and turn the photo upright before anything depends on orientation.
    gray = decode_grayscale(payload)
    rotated = rotate_image(gray, rotation)
    steps = [
        PreprocessingStep(name="grayscale", params={}),
        PreprocessingStep(name="rotate", params={"degrees_clockwise": rotation}),
    ]

    # Keep only the paper, then only the strokes that belong to the drawing.
    pixel_scale = max(rotated.shape) / REFERENCE_LONG_SIDE_PIXELS
    paper_only, paper_mask, paper_params = isolate_paper(rotated, pixel_scale)
    steps.append(PreprocessingStep(name="isolate_paper", params=paper_params))
    drawing, strokes, stroke_params = extract_strokes(paper_only, paper_mask, pixel_scale)
    steps.append(PreprocessingStep(name="extract_strokes", params=stroke_params))

    # Crop to the strokes, thicken them, and scale to the target size.
    cropped, crop_params = crop_to_drawing(drawing, strokes)
    steps.append(PreprocessingStep(name="crop_to_drawing", params=crop_params))
    thickened, thicken_params = thicken_strokes(cropped, pixel_scale)
    steps.append(PreprocessingStep(name="thicken_strokes", params=thicken_params))
    resized, resize_params = resize_long_side(thickened)
    steps.append(PreprocessingStep(name="resize", params=resize_params))

    image, processed = encode_processed_sketch(resized)
    return PreprocessedSketch(image=image, payload=processed, steps=tuple(steps))


class Preprocessor(ABC):
    """
    Prepares a raw sketch for generation and records each applied step with the parameters it used.
    """

    @abstractmethod
    def preprocess(self, payload: bytes, rotation: int) -> PreprocessedSketch:
        """
        Turn a raw sketch into the generation sketch, raising when it cannot apply the rotation.

        Args:
            payload (bytes): Encoded raw sketch image.
            rotation (int): Clockwise rotation in degrees, one of 0, 90, 180, or 270.

        Returns:
            PreprocessedSketch: The processed image with a relative path, its bytes, and the ordered steps.
        """
        ...


class StrokePreprocessor(Preprocessor):
    """
    Default preprocessor that straightens the paper, isolates the strokes, and crops and resizes them.
    """

    def preprocess(self, payload: bytes, rotation: int) -> PreprocessedSketch:
        """
        Preprocess the sketch with preprocess_sketch.
        """
        return preprocess_sketch(payload, rotation)
