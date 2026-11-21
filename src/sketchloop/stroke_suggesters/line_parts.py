import cv2
import numpy as np

from dataclasses import asdict, dataclass

from sketchloop.capture import InvalidSketchError
from sketchloop.domain import ControlValue
from sketchloop.stroke_suggestion import StrokeSuggestionRequest


@dataclass(frozen=True, slots=True, kw_only=True)
class LinePartParams:
    line_contrast: int = 40  # A line pixel is at least this much darker than the local median
    median_pixels: int = 31  # Odd window of the local median that stands for the paper level
    dark_level: int = 90  # Pixels darker than this are lines whatever their surroundings
    existing_stroke_kernel_pixels: int = 13  # Diameter of the ellipse grown around existing strokes, whose lines are not new
    min_speck_pixels: int = 25  # Line specks with fewer pixels are dropped
    link_pixels: int = 10  # Strokes this close to each other join one part
    min_part_pixels: int = 150  # Parts with fewer pixels are dropped
    max_straight_width_ratio: float = 0.08  # A part no wider than this share of its length is nearly straight
    min_straight_span_share: float = 0.6  # A nearly straight part spanning this share of the canvas width or height is dropped
    long_line_share: float = 0.4  # A straight segment at least this share of the canvas's longer side is a long line
    long_line_vote_share: float = 0.5  # A long line needs line pixels along at least this share of that minimum length
    long_line_gap_share: float = 0.04  # Gaps up to this share of the canvas's longer side do not break a long line
    long_line_grow_pixels: int = 3  # Diameter the lines grow by before the search, so a line slanting a pixel stays whole
    long_line_max_near_share: float = 0.5  # A long line with more of its length near existing strokes continues them and stays
    long_line_band_pixels: int = 9  # Width of the band removed along each long line


def list_extraction_settings(params: LinePartParams) -> dict[str, ControlValue]:
    return {f"extract_{name}": value for name, value in asdict(params).items()}


def make_ellipse(diameter_pixels: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter_pixels, diameter_pixels))


def decode_canvas(request: StrokeSuggestionRequest, canvas_payload: bytes) -> np.ndarray:
    """
    Decode the canvas bytes to grayscale, refusing bytes that are not an image of the request's canvas size.

    Args:
        request (StrokeSuggestionRequest): Request whose canvas describes the expected size.
        canvas_payload (bytes): PNG bytes of the canvas.

    Returns:
        np.ndarray: Grayscale canvas.
    """
    # OpenCV refuses empty bytes with its own error, so treat them as undecodable here.
    gray = cv2.imdecode(np.frombuffer(canvas_payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if canvas_payload else None
    if gray is None:
        raise InvalidSketchError("Suggestion canvas cannot be decoded. Capture the paper again and retry.")

    # A canvas of another size would give masks that do not overlay the recorded canvas.
    actual = f"{gray.shape[1]}x{gray.shape[0]}"
    expected = f"{request.canvas.width}x{request.canvas.height}"
    if actual != expected:
        raise InvalidSketchError(f"Suggestion canvas is {actual}, not the requested {expected}. Capture again and retry.")

    return gray


def mark_long_lines(lines: np.ndarray, near_existing: np.ndarray, params: LinePartParams) -> np.ndarray:
    # Search the slightly grown lines for straight segments that are long relative to the canvas.
    longer_side = max(lines.shape)
    min_length = round(params.long_line_share * longer_side)
    grown = cv2.dilate(lines, make_ellipse(params.long_line_grow_pixels))
    found = cv2.HoughLinesP(grown, 1, np.pi / 180, threshold=round(params.long_line_vote_share * min_length),
                            minLineLength=min_length, maxLineGap=round(params.long_line_gap_share * longer_side))
    segments = [] if found is None else found.reshape(-1, 4).tolist()

    # Mark a band along each segment mostly away from the existing strokes, since one mostly along them continues them.
    band = np.zeros_like(lines)
    for start_x, start_y, end_x, end_y in segments:
        segment = np.zeros_like(lines)
        cv2.line(segment, (start_x, start_y), (end_x, end_y), 255, 1)
        on_segment = segment > 0
        if np.count_nonzero(near_existing & on_segment) <= params.long_line_max_near_share * np.count_nonzero(on_segment):
            cv2.line(band, (start_x, start_y), (end_x, end_y), 255, params.long_line_band_pixels)

    return band > 0


def extract_new_lines(line_drawing_gray: np.ndarray, canvas_strokes: np.ndarray, params: LinePartParams) -> np.ndarray:
    # Lines are pixels clearly darker than the local paper, or dark outright.
    median = cv2.medianBlur(line_drawing_gray, params.median_pixels)
    is_line = (median.astype(np.int16) - line_drawing_gray > params.line_contrast) | (line_drawing_gray < params.dark_level)

    # Lines on or near the existing strokes are the model redrawing them, so only the rest are new.
    near_existing = cv2.dilate(canvas_strokes, make_ellipse(params.existing_stroke_kernel_pixels)) > 0

    # Long straight lines across the whole drawing are ruled paper, floors, or horizons, so they are not new either.
    long_lines = mark_long_lines(np.where(is_line, 255, 0).astype(np.uint8), near_existing, params)
    return np.where(is_line & ~near_existing & ~long_lines, 255, 0).astype(np.uint8)


def drop_specks(lines: np.ndarray, min_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    kept = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= min_pixels]
    return np.where(np.isin(labels, kept), 255, 0).astype(np.uint8)


def is_frame_line(part: np.ndarray, params: LinePartParams) -> bool:
    # A nearly straight part is much longer than it is wide.
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(cv2.findNonZero(part))
    length, width = max(side_a, side_b), min(side_a, side_b)
    if width > params.max_straight_width_ratio * length:
        return False

    # It is a frame line when it also spans most of the canvas in either direction.
    _, _, box_width, box_height = cv2.boundingRect(part)
    canvas_height, canvas_width = part.shape
    share = params.min_straight_span_share
    return box_width >= share * canvas_width or box_height >= share * canvas_height


def group_parts(new_lines: np.ndarray, params: LinePartParams, max_parts: int) -> list[np.ndarray]:
    # Link strokes within the link distance by growing each by half of it, then label the linked groups.
    lines = drop_specks(new_lines, params.min_speck_pixels)
    linked = cv2.dilate(lines, make_ellipse(params.link_pixels + 1))
    count, labels = cv2.connectedComponents(linked, connectivity=8)

    # Each part is a group's own line pixels, kept when large enough and not a straight frame line.
    parts = []
    for label in range(1, count):
        part = np.where((labels == label) & (lines > 0), 255, 0).astype(np.uint8)
        if cv2.countNonZero(part) >= params.min_part_pixels and not is_frame_line(part, params):
            parts.append(part)

    # Masks already stop at the canvas edge, so parts touching it are kept, largest first.
    parts.sort(key=cv2.countNonZero, reverse=True)
    return parts[:max_parts]


def count_skeleton_pixels(mask: np.ndarray) -> int:
    # Peel the mask with repeated erosions, keeping what each opening removes, which is OpenCV's morphological skeleton.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    remaining = mask.copy()
    skeleton = np.zeros_like(mask)
    while cv2.countNonZero(remaining):
        eroded = cv2.erode(remaining, kernel)
        skeleton |= cv2.subtract(remaining, cv2.dilate(eroded, kernel))
        remaining = eroded

    # Any non-empty mask has a skeleton of at least one pixel.
    return max(1, cv2.countNonZero(skeleton))
