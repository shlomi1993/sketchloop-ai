"""
Tests of new-line extraction and part grouping on synthetic line drawings.
"""

import cv2
import numpy as np
import pytest

from sketchloop.stroke_suggesters.line_parts import (LinePartParams, count_skeleton_pixels, extract_new_lines, group_parts,
                                                     list_extraction_settings)


CANVAS_SHAPE = (360, 512)
RULE_ROWS = (20, 80, 140, 260, 320)


def make_blank() -> np.ndarray:
    """
    Return a white grayscale canvas of the default size.
    """
    return np.full(CANVAS_SHAPE, 255, dtype=np.uint8)


def draw_existing_strokes() -> np.ndarray:
    """
    Return a mask that is 255 on an existing rectangle, the person's strokes.
    """
    strokes = np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    cv2.rectangle(strokes, (150, 100), (350, 200), 255, 3)
    return strokes


def test_extract_new_lines_skips_lines_near_existing_strokes() -> None:
    """
    New lines are the drawing's dark lines away from the existing strokes, without light gray texture.
    """
    line_drawing = make_blank()
    cv2.rectangle(line_drawing, (150, 100), (350, 200), 0, 3)
    cv2.line(line_drawing, (150, 205), (350, 205), 0, 2)
    cv2.circle(line_drawing, (250, 290), 40, 30, 3)
    line_drawing[20:60, 400:500] = 225

    new_lines = extract_new_lines(line_drawing, draw_existing_strokes(), LinePartParams())

    # The redrawn rectangle and the line 5 px below it are not new.
    assert not new_lines[95:208, 145:355].any(), "Lines within reach of the existing strokes must be removed"

    # The circle below is new, and the light gray patch is not a line.
    assert new_lines[245:335, 205:295].any(), "The circle away from the strokes must be a new line"
    assert not new_lines[20:60, 400:500].any(), "Light gray texture must not count as a line"
    assert set(np.unique(new_lines)) <= {0, 255}, "The new-line mask must hold only 0 and 255"


def test_extract_new_lines_drops_ruling_and_keeps_parts() -> None:
    """
    Ruled lines across the whole drawing are removed, while new parts between them stay whole.
    """
    line_drawing = make_blank()
    for rule_y in RULE_ROWS:
        cv2.line(line_drawing, (0, rule_y), (511, rule_y), 60, 2)
    cv2.rectangle(line_drawing, (150, 100), (350, 200), 0, 3)
    cv2.circle(line_drawing, (70, 110), 20, 0, 3)
    cv2.polylines(line_drawing, [np.array([[420, 230], [440, 250], [460, 230], [480, 250]])], False, 0, 3)

    parts = group_parts(extract_new_lines(line_drawing, draw_existing_strokes(), LinePartParams()), LinePartParams(), 4)
    boxes = [cv2.boundingRect(part) for part in parts]

    # Only the circle and the zigzag remain, each whole.
    assert len(parts) == 2, f"Expected the circle and the zigzag, got boxes {boxes}"
    assert abs(boxes[0][0] - 49) <= 2 and abs(boxes[0][2] - 43) <= 2, f"The circle must be whole, got {boxes[0]}"
    assert boxes[1][0] <= 420 and boxes[1][0] + boxes[1][2] >= 480, f"The zigzag must be whole, got {boxes[1]}"

    # No part keeps any piece of a ruled line.
    covered = np.bitwise_or.reduce(np.stack(parts), axis=0)
    for rule_y in RULE_ROWS:
        assert not covered[rule_y - 2:rule_y + 3].any(), f"The ruled line at y={rule_y} must be dropped"


def test_extract_new_lines_drops_horizon_crossing_the_drawing() -> None:
    """
    A horizon line crossing the existing drawing is dropped on both sides of it, not cut into short new pieces.
    """
    line_drawing = make_blank()
    cv2.rectangle(line_drawing, (150, 100), (350, 200), 0, 3)
    cv2.line(line_drawing, (0, 150), (511, 152), 0, 2)

    new_lines = extract_new_lines(line_drawing, draw_existing_strokes(), LinePartParams())

    # Neither side of the horizon is a new line.
    assert not new_lines[140:165].any(), f"The horizon must be dropped, kept {cv2.countNonZero(new_lines[140:165])} pixels"


def test_extract_new_lines_keeps_legs_attached_to_the_drawing() -> None:
    """
    A leg continuing an existing post and a leg hanging from the seat survive, although the first is long and straight.
    """
    existing = np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    cv2.line(existing, (200, 30), (200, 230), 255, 3)
    cv2.line(existing, (200, 230), (330, 230), 255, 3)
    line_drawing = make_blank()
    cv2.line(line_drawing, (200, 30), (200, 340), 0, 3)
    cv2.line(line_drawing, (200, 230), (330, 230), 0, 3)
    cv2.line(line_drawing, (330, 230), (330, 340), 0, 3)

    parts = group_parts(extract_new_lines(line_drawing, existing, LinePartParams()), LinePartParams(), 4)
    boxes = sorted(cv2.boundingRect(part) for part in parts)

    # Both legs are parts that reach down to their ends.
    assert len(parts) == 2, f"Expected the two legs, got boxes {boxes}"
    assert abs(boxes[0][0] - 199) <= 2 and boxes[0][1] + boxes[0][3] >= 340, f"The post's leg must survive, got {boxes[0]}"
    assert abs(boxes[1][0] - 329) <= 2 and boxes[1][1] + boxes[1][3] >= 340, f"The seat's leg must survive, got {boxes[1]}"


def test_group_parts_orders_parts_and_drops_small_ones() -> None:
    """
    Separate shapes become parts, largest first, with near strokes linked and specks and small groups dropped.
    """
    circle, rectangle = np.zeros(CANVAS_SHAPE, dtype=np.uint8), np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    cv2.circle(circle, (100, 100), 50, 255, 3)
    cv2.rectangle(rectangle, (300, 50), (400, 110), 255, 2)
    new_lines = circle | rectangle
    cv2.line(new_lines, (300, 250), (360, 250), 255, 2)
    cv2.line(new_lines, (368, 250), (430, 250), 255, 2)
    cv2.rectangle(new_lines, (100, 300), (110, 310), 255, 1)
    new_lines[340:343, 480:483] = 255

    parts = group_parts(new_lines, LinePartParams(), 4)
    boxes = [cv2.boundingRect(part) for part in parts]

    # The circle, rectangle, and the two lines 8 px apart are three parts, largest first.
    assert len(parts) == 3, f"Expected 3 parts, got {len(parts)} with boxes {boxes}"
    sizes = [cv2.countNonZero(part) for part in parts]
    assert sizes == sorted(sizes, reverse=True), f"Parts must be largest first, got sizes {sizes}"
    assert boxes[0] == cv2.boundingRect(circle), f"The circle must come first, got {boxes[0]}"
    assert boxes[1] == cv2.boundingRect(rectangle), f"The rectangle must come second, got {boxes[1]}"
    assert boxes[2][0] <= 300 and boxes[2][0] + boxes[2][2] >= 430, f"The two close lines must form one part, got {boxes[2]}"

    # The 11 px square, under the minimum part size, and the speck are in no part.
    covered = np.bitwise_or.reduce(np.stack(parts), axis=0)
    assert not covered[300:311, 100:111].any(), "The small square must be dropped"
    assert not covered[340:343, 480:483].any(), "The speck must be dropped"


def test_group_parts_keeps_edge_parts_and_drops_frame_lines() -> None:
    """
    Parts cut by the canvas edge are kept, while a long straight line across the canvas is dropped as a frame line.
    """
    new_lines = np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    cv2.circle(new_lines, (500, 200), 40, 255, 3)
    cv2.line(new_lines, (10, 15), (500, 15), 255, 3)
    cv2.line(new_lines, (100, 60), (200, 60), 255, 3)
    cv2.line(new_lines, (300, 300), (300, 359), 255, 4)

    parts = group_parts(new_lines, LinePartParams(), 4)
    boxes = [cv2.boundingRect(part) for part in parts]

    # The circle at the right edge, the short line, and the leg at the bottom edge remain, largest first.
    assert len(parts) == 3, f"Expected the edge circle, the short line, and the leg, got boxes {boxes}"
    assert boxes[0][0] + boxes[0][2] == 512, f"The edge part must be clipped at the canvas edge, got {boxes[0]}"
    assert boxes[1][1] < 100, f"The kept line must be the short one, got {boxes[1]}"
    assert boxes[2][1] + boxes[2][3] == 360, f"The leg at the bottom edge must be kept, got {boxes[2]}"

    # The line across the top is dropped.
    covered = np.bitwise_or.reduce(np.stack(parts), axis=0)
    assert not covered[10:21, 10:500].any(), "The frame line must be dropped"


@pytest.mark.parametrize("max_parts", [1, 2, 4])
def test_group_parts_returns_at_most_max_parts(max_parts: int) -> None:
    """
    Of five separate parts, only the largest max_parts are returned.
    """
    new_lines = np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    for index in range(5):
        cv2.circle(new_lines, (50 + index * 100, 180), 20 + index * 4, 255, 2)

    parts = group_parts(new_lines, LinePartParams(), max_parts)

    # The returned parts are the largest circles, from the right.
    assert len(parts) == max_parts, f"Expected {max_parts} parts, got {len(parts)}"
    lefts = [cv2.boundingRect(part)[0] for part in parts]
    assert lefts == sorted(lefts, reverse=True), f"The largest circles are on the right, got lefts {lefts}"


def test_count_skeleton_pixels_measures_line_length() -> None:
    """
    A 5 px thick, 100 px long line has a skeleton of about its length, no more than its pixels.
    """
    mask = np.zeros((40, 140), dtype=np.uint8)
    cv2.line(mask, (20, 20), (119, 20), 255, 5)

    skeleton_pixels = count_skeleton_pixels(mask)

    # The skeleton runs along the line and is a subset of it.
    assert 80 <= skeleton_pixels <= 105, f"Expected about 100 skeleton pixels, got {skeleton_pixels}"
    assert skeleton_pixels <= cv2.countNonZero(mask), "The skeleton must not have more pixels than the mask"


def test_extraction_settings_are_prefixed() -> None:
    """
    Every extraction threshold is listed with an extract_ prefix for the effective settings.
    """
    settings = list_extraction_settings(LinePartParams())

    # Each field appears once with its default.
    assert settings["extract_line_contrast"] == 40 and settings["extract_min_part_pixels"] == 150, f"Got {settings}"
    assert settings["extract_long_line_share"] == 0.4, f"The long-line filter must be recorded, got {settings}"
    assert all(name.startswith("extract_") for name in settings), f"Every name must start with extract_, got {list(settings)}"
