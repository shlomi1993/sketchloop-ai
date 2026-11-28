"""
Tests of new-line extraction and part grouping on synthetic line drawings.
"""

import cv2
import numpy as np

from sketchloop.stroke_suggesters.line_parts import (LinePartParams, extract_new_lines, group_parts)


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
    New lines are the drawing's dark lines away from the existing strokes, without light gray texture or a crossing horizon.
    """
    line_drawing = make_blank()
    cv2.rectangle(line_drawing, (150, 100), (350, 200), 0, 3)
    cv2.line(line_drawing, (150, 205), (350, 205), 0, 2)
    cv2.circle(line_drawing, (250, 290), 40, 30, 3)
    line_drawing[20:60, 400:500] = 225
    horizon = line_drawing.copy()
    cv2.line(horizon, (0, 150), (511, 152), 0, 2)

    new_lines = extract_new_lines(line_drawing, draw_existing_strokes(), LinePartParams())
    with_horizon = extract_new_lines(horizon, draw_existing_strokes(), LinePartParams())

    # The redrawn rectangle and the line 5 px below it are not new.
    assert not new_lines[95:208, 145:355].any(), "Lines within reach of the existing strokes must be removed"

    # A horizon crossing the drawing is dropped on both sides of it, not cut into short new pieces.
    assert not with_horizon[140:165].any(), f"The horizon must be dropped, kept {cv2.countNonZero(with_horizon[140:165])} pixels"

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
    Separate shapes become parts, largest first up to the limit, with near strokes linked and specks and small groups dropped.
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
    largest_two = group_parts(new_lines, LinePartParams(), 2)

    # The circle, rectangle, and the two lines 8 px apart are three parts, largest first, and a limit of two keeps the first two.
    assert len(parts) == 3, f"Expected 3 parts, got {len(parts)} with boxes {boxes}"
    sizes = [cv2.countNonZero(part) for part in parts]
    assert sizes == sorted(sizes, reverse=True), f"Parts must be largest first, got sizes {sizes}"
    assert boxes[0] == cv2.boundingRect(circle), f"The circle must come first, got {boxes[0]}"
    assert boxes[1] == cv2.boundingRect(rectangle), f"The rectangle must come second, got {boxes[1]}"
    assert boxes[2][0] <= 300 and boxes[2][0] + boxes[2][2] >= 430, f"The two close lines must form one part, got {boxes[2]}"
    assert [cv2.boundingRect(part) for part in largest_two] == boxes[:2], f"Expected the two largest parts, got {largest_two}"

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


def test_extract_new_lines_drops_loose_lines_from_the_border() -> None:
    """
    Loose straight lines from the canvas border or along it are dropped as sheet edges, unlike legs and short lines.
    """
    line_drawing = make_blank()
    cv2.rectangle(line_drawing, (150, 100), (350, 200), 0, 3)
    cv2.line(line_drawing, (40, 0), (40, 175), 0, 3)
    cv2.line(line_drawing, (0, 300), (168, 300), 0, 3)
    cv2.line(line_drawing, (500, 359), (500, 300), 0, 3)
    cv2.line(line_drawing, (330, 203), (330, 359), 0, 3)
    cv2.line(line_drawing, (420, 40), (420, 215), 0, 3)
    cv2.line(line_drawing, (300, 1), (380, 1), 0, 3)

    parts = group_parts(extract_new_lines(line_drawing, draw_existing_strokes(), LinePartParams()), LinePartParams(), 4)
    boxes = sorted(cv2.boundingRect(part) for part in parts)

    # The vertical line from the top border, the horizontal one from the left border, and the one along the top are dropped.
    covered = np.bitwise_or.reduce(np.stack(parts), axis=0) if parts else np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    assert not covered[:180, 30:51].any(), f"The loose line from the top border must be dropped, got boxes {boxes}"
    assert not covered[290:311, :175].any(), f"The loose line from the left border must be dropped, got boxes {boxes}"
    assert not covered[:6, 295:385].any(), f"The short line along the top edge must be dropped, got boxes {boxes}"

    # The leg from the drawing to the bottom border, the short line at the border, and the free inner line remain whole.
    assert len(parts) == 3, f"Expected the leg, the short line, and the inner line, got boxes {boxes}"
    assert boxes[0][0] <= 330 and boxes[0][1] + boxes[0][3] == 360, f"The leg must reach the bottom border, got {boxes[0]}"
    assert boxes[1][1] <= 40 and boxes[1][1] + boxes[1][3] >= 215, f"The free inner line must be whole, got {boxes[1]}"
    assert boxes[2][0] >= 497 and boxes[2][1] <= 300, f"The short line at the border must stay, got {boxes[2]}"
