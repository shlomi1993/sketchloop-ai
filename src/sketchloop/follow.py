"""
Follow detection: align each later capture of a drawing to a suggestion's baseline canvas and score its new strokes per part.
"""

import cv2
import math
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from sketchloop.domain import ControlValue, Unavailable, freeze_controls, require, require_duration
from sketchloop.preprocessing import LOCAL_STROKE_CONTRAST, decode_grayscale, find_skeleton
from sketchloop.stroke_suggestion import is_count


ALIGNED_STROKES_METHOD: Final = ("aligned-strokes", "3")  # Detector name and version, bumped whenever its results change
DARKNESS_MEDIAN_PIXELS: Final = 31  # Window of the local paper level that darkness is measured against
DARKNESS_BLUR_SIGMA: Final = 2.0  # Blur of the darkness map, which smooths camera noise and widens ECC's reach
PAPER_LEVEL_DILATE_PIXELS: Final = 101  # Window of the local maximum taken as the paper level when finding occluders
PAPER_LEVEL_BLUR_SIGMA: Final = 20.0  # Blur of that paper level
OCCLUDER_CONTRAST: Final = 30  # Levels below the paper level that mark an occluder candidate, such as a hand
OCCLUDER_OPENING_PIXELS: Final = 25  # Opening that removes strokes, so only wide dark blobs stay occluders
MIN_OCCLUDER_SHARE: Final = 0.01  # Smallest occluder as a share of the canvas
OCCLUDER_GROW_PIXELS: Final = 10  # Margin added around each occluder, which covers its shadow and soft edge
ECC_ITERATIONS: Final = 100
ECC_EPSILON: Final = 1e-5
ECC_GAUSSIAN_PIXELS: Final = 5  # Gaussian filter size ECC applies to both darkness maps
ECC_STROKE_REACH_PIXELS: Final = 16  # ECC aligns only this near baseline strokes, so a new stroke elsewhere cannot drag it
ECC_NEW_STROKE_DISTANCE_PIXELS: Final = 12  # Capture strokes farther than this from every baseline stroke are new to ECC
ECC_NEW_STROKE_MARGIN_PIXELS: Final = 8  # New strokes and this margin around them are left out of ECC
MIN_ECC_STROKE_PIXELS: Final = 1000  # With fewer visible baseline stroke pixels, about a 90 px line, ECC uses the whole canvas
BASELINE_GROW_PIXELS: Final = 4  # Baseline strokes are grown by this much, so a slightly misaligned old stroke is not new
MIN_NEW_STROKE_SPECK_PIXELS: Final = 20  # New-stroke specks smaller than this are noise
MIN_ECC_CORRELATION: Final = 0.4  # A lower ECC correlation is not a usable alignment, though ECC converged
MAX_CORNER_SHIFT_PIXELS: Final = 32  # A warp that moves any canvas corner farther than this is not a plausible camera shift
TOLERANCES_PIXELS: Final = (4, 8, 16, 24)  # Distances at which coverage and precision are measured
PART_REACH_PIXELS: Final = 32  # A part's box is grown by this much to find the new strokes near it
CHAMFER_CAP_PIXELS: Final = 64  # Distances in the chamfer means are capped at this
SCORE_DECIMALS: Final = 4  # Shares and distances are rounded to this many decimals, which keeps drawing.json small
NOT_CONVERGED_REASON: Final = "ECC did not converge"
IMPLAUSIBLE_WARP_REASON: Final = "ECC converged to an implausible warp"
NO_NEW_STROKES_REASON: Final = "no new strokes"
NO_NEW_STROKES_NEAR_REASON: Final = "no new strokes near the part"
PART_OCCLUDED_REASON: Final = "the whole part is occluded"
FILLED_SHARE_NOT_RECORDED: Final = "not recorded in captures aligned before the filled share was"
ALIGNED_ON_NOT_RECORDED: Final = "not recorded in captures aligned before the region was"
NO_USABLE_CAPTURE_REASON: Final = "no usable capture"
NOTHING_NEW_REASON: Final = "nothing new drawn"
NO_PARTS_REASON: Final = "no parts suggested"


def is_finite_number(value: object) -> bool:
    """
    Tell whether a value is a finite int or float, not a bool.
    """
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def check_unit_share(value: object, label: str) -> None:
    """
    Refuse a share that is not a finite number from 0 to 1.
    """
    require(is_finite_number(value) and 0 <= value <= 1, f"{label} must be a number from 0 to 1, got {value!r}.")


def freeze_warp(warp: object) -> tuple[tuple[float, float, float], ...] | None:
    """
    Return a 3x3 warp as a tuple of float rows, or None when it is not 3 rows of 3 finite numbers.
    """
    if not isinstance(warp, Sequence) or len(warp) != 3:
        return None

    if not all(isinstance(row, Sequence) and len(row) == 3 and all(is_finite_number(value) for value in row) for row in warp):
        return None

    return tuple(tuple(float(value) for value in row) for row in warp)


class AlignmentRegion(StrEnum):
    """
    Which pixels of the baseline canvas ECC aligned on.
    """
    BASELINE_STROKES = "baseline_strokes"
    WHOLE_CANVAS = "whole_canvas"


@dataclass(frozen=True, slots=True, kw_only=True)
class Alignment:
    """
    How a capture's canvas was aligned to a suggestion's baseline canvas, and how much of it an occluder covered.
    """
    converged: bool  # Whether ECC converged
    correlation: float | Unavailable  # ECC correlation, unavailable with the reason when ECC did not converge
    warp: tuple[tuple[float, float, float], ...] | None  # 3x3 homography from capture to baseline pixels, None without one
    occluded_share: float  # Share of the capture's canvas masked as an occluder, such as a hand, or filled in rather than seen
    filled_share: float | Unavailable = Unavailable(FILLED_SHARE_NOT_RECORDED)  # Part of occluded_share filled in, as the rim is
    aligned_on: AlignmentRegion | Unavailable = Unavailable(ALIGNED_ON_NOT_RECORDED)  # Baseline pixels ECC aligned on

    def __post_init__(self) -> None:
        # A converged alignment has a finite correlation and a finite 3x3 warp, and a failed one has neither.
        require(isinstance(self.converged, bool), f"Alignment converged must be true or false, got {self.converged!r}.")
        if self.converged:
            correlation = self.correlation
            require(is_finite_number(correlation), f"A converged alignment needs a finite correlation, got {correlation}.")
            warp = freeze_warp(self.warp)
            require(warp is not None, f"A converged alignment needs a 3x3 warp of finite numbers, got {self.warp!r}.")
            object.__setattr__(self, "warp", warp)
        else:
            require(isinstance(self.correlation, Unavailable), "An alignment that did not converge has no correlation.")
            require(self.warp is None, "An alignment that did not converge has no warp.")
        check_unit_share(self.occluded_share, "Alignment occluded_share")

        # The filled area is part of the occluded area, so the difference is what an occluder such as a hand covered.
        if not isinstance(self.filled_share, Unavailable):
            check_unit_share(self.filled_share, "Alignment filled_share")
            filled, occluded = self.filled_share, self.occluded_share
            require(filled <= occluded, f"Alignment filled_share must be at most occluded_share, got {filled} > {occluded}.")

        # The region ECC aligned on is a known one, or unavailable in older records.
        if not isinstance(self.aligned_on, Unavailable):
            region, known = self.aligned_on, ", ".join(AlignmentRegion)
            require(region in list(AlignmentRegion), f"Alignment aligned_on must be one of {known}, got {region!r}.")
            object.__setattr__(self, "aligned_on", AlignmentRegion(region))


def freeze_tolerance_scores(scores: Mapping[int, float | Unavailable], label: str) -> Mapping[int, float | Unavailable]:
    require(isinstance(scores, Mapping) and bool(scores), f"{label} needs a score for at least one tolerance.")
    for tolerance, value in scores.items():
        require(is_count(tolerance, 1), f"{label} tolerance must be an int >= 1, got {tolerance!r}.")
        if not isinstance(value, Unavailable):
            check_unit_share(value, f"{label} at {tolerance} px")

    return MappingProxyType(dict(scores))


def is_distance(value: object) -> bool:
    return isinstance(value, Unavailable) or (is_finite_number(value) and value >= 0)


@dataclass(frozen=True, slots=True, kw_only=True)
class PartScores:
    part: int  # Part index
    occluded_share: float  # Share of the part's mask pixels under an occluder or a filled area
    new_pixels_near: int  # New-stroke pixels inside the part's box grown by PART_REACH_PIXELS
    coverage: Mapping[int, float | Unavailable]  # Share of the part's visible skeleton within t px of new strokes, by t
    precision: Mapping[int, float | Unavailable]  # Share of the new strokes near the part within t px of it, by t
    chamfer_part_to_new: float | Unavailable  # Mean distance from the visible skeleton to the new strokes, capped
    chamfer_new_to_part: float | Unavailable  # Mean distance from the new strokes near the part to it, capped

    def __post_init__(self) -> None:
        # The part, its occluded share, and the nearby new pixels are plain counts and shares.
        require(is_count(self.part, 0), f"Scored part must be an int >= 0, got {self.part!r}.")
        check_unit_share(self.occluded_share, f"Part {self.part} occluded_share")
        require(is_count(self.new_pixels_near, 0), f"Part {self.part} new_pixels_near must be an int >= 0.")

        # Coverage and precision share their tolerances, and precision exists exactly when new strokes are near.
        object.__setattr__(self, "coverage", freeze_tolerance_scores(self.coverage, f"Part {self.part} coverage"))
        object.__setattr__(self, "precision", freeze_tolerance_scores(self.precision, f"Part {self.part} precision"))
        same = set(self.coverage) == set(self.precision)
        require(same, f"Part {self.part} coverage and precision must have the same tolerances.")
        near = self.new_pixels_near > 0
        measured = [not isinstance(value, Unavailable) for value in (*self.precision.values(), self.chamfer_new_to_part)]
        require(all(flag == near for flag in measured), f"Part {self.part} has precision exactly when new strokes are near it.")
        distances = (self.chamfer_part_to_new, self.chamfer_new_to_part)
        require(all(is_distance(value) for value in distances), f"Part {self.part} chamfer distances must be >= 0 or unknown.")


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureScores:
    """
    A capture measured against the open suggestion's baseline canvas: its alignment and how much was drawn since.
    """
    suggestion: int  # Number of the suggestion whose window the capture fell in
    seconds_since_shown: float  # Seconds from when the suggestion was shown to the capture
    alignment: Alignment  # Alignment to the suggestion's baseline canvas
    new_stroke_pixels: int | None  # New-stroke pixels on the whole canvas, None when alignment failed
    parts: tuple[PartScores, ...] | None  # One per part in index order, empty when alignment failed, None in older records

    def __post_init__(self) -> None:
        # The suggestion and time place the capture in a window, and new strokes are counted only after an alignment.
        require(is_count(self.suggestion, 1), f"Scored suggestion must be an int >= 1, got {self.suggestion!r}.")
        require_duration(self.seconds_since_shown, "Capture seconds_since_shown")
        require(isinstance(self.alignment, Alignment), "Capture scores need an alignment.")
        pixels = self.new_stroke_pixels
        if self.alignment.converged:
            require(is_count(pixels, 0), f"An aligned capture needs new_stroke_pixels as an int >= 0, got {pixels!r}.")
        else:
            require(pixels is None, f"A capture that did not align has no new_stroke_pixels, got {pixels!r}.")

        # Parts are scored in index order, and only after an alignment.
        if self.parts is not None:
            object.__setattr__(self, "parts", tuple(self.parts))
            indexes = [part.part if isinstance(part, PartScores) else None for part in self.parts]
            require(indexes == list(range(len(indexes))), f"Part scores must be PartScores numbered from 0, got {indexes}.")
            require(self.alignment.converged or not self.parts, "A capture that did not align has no part scores.")


@dataclass(frozen=True, slots=True, kw_only=True)
class FollowMethod:
    """
    The follow detector a drawing used, recorded once per drawing so its scores can be interpreted and recomputed.
    """
    name: str  # Detector name, such as "aligned-strokes"
    version: str  # Detector version, bumped whenever its results change
    params: Mapping[str, ControlValue]  # Every threshold the detector uses, by snake_case name

    def __post_init__(self) -> None:
        # Name and version identify the method, and the parameters are plain values.
        require(isinstance(self.name, str) and bool(self.name.strip()), "Follow method needs a non-empty name.")
        require(isinstance(self.version, str) and bool(self.version.strip()), "Follow method needs a non-empty version.")
        object.__setattr__(self, "params", freeze_controls(self.params))


class FollowDetector(ABC):
    """
    Measures each capture of a drawing against the canvas a suggestion was made from.
    """

    @property
    @abstractmethod
    def method(self) -> tuple[str, str]:
        """
        Return the detector's name and version.
        """
        ...

    @abstractmethod
    def describe_params(self) -> Mapping[str, ControlValue]:
        """
        Return every threshold the detector uses, recorded once per drawing.
        """
        ...

    @abstractmethod
    def score(self, baseline_canvas: bytes, capture_canvas: bytes, part_masks: Sequence[bytes], *,
              baseline_filled: bytes | None = None,
              capture_filled: bytes | None = None) -> tuple[Alignment, int | None, tuple[PartScores, ...]]:
        ...

    def align(self, baseline_canvas: bytes, capture_canvas: bytes, *, baseline_filled: bytes | None = None,
              capture_filled: bytes | None = None) -> tuple[Alignment, int | None]:
        # Aligning is scoring without parts.
        alignment, new_stroke_pixels, _ = self.score(baseline_canvas, capture_canvas, (), baseline_filled=baseline_filled,
                                                     capture_filled=capture_filled)
        return alignment, new_stroke_pixels


def describe_follow_method(detector: FollowDetector) -> FollowMethod:
    """
    Record a detector's name, version, and parameters.
    """
    name, version = detector.method
    return FollowMethod(name=name, version=version, params=detector.describe_params())


def make_ellipse(radius_pixels: int) -> np.ndarray:
    """
    Return an elliptical structuring element that grows a mask by the radius.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_pixels + 1,) * 2)


def make_darkness_map(gray: np.ndarray) -> np.ndarray:
    """
    Measure how much darker each pixel is than its local paper, blurred, which does not change with the lighting.

    Args:
        gray (np.ndarray): Grayscale canvas.

    Returns:
        np.ndarray: float32 darkness map of the same size, 0 on bare paper.
    """
    local_paper = cv2.medianBlur(gray, DARKNESS_MEDIAN_PIXELS).astype(np.float32)
    darkness = np.clip(local_paper - gray.astype(np.float32), 0, 255)
    return cv2.GaussianBlur(darkness, (0, 0), DARKNESS_BLUR_SIGMA)


def find_occluders(gray: np.ndarray) -> np.ndarray:
    """
    Find wide dark blobs that cover the paper, such as a hand, grown by a margin.

    Args:
        gray (np.ndarray): Grayscale canvas.

    Returns:
        np.ndarray: Boolean mask of the occluders, true where the paper is covered.
    """
    # Estimate the paper level as a wide local maximum, which strokes cannot pull down the way a median is pulled.
    window = np.ones((PAPER_LEVEL_DILATE_PIXELS, PAPER_LEVEL_DILATE_PIXELS), dtype=np.uint8)
    paper_level = cv2.GaussianBlur(cv2.dilate(gray, window), (0, 0), PAPER_LEVEL_BLUR_SIGMA)
    dark = ((paper_level.astype(np.int16) - gray.astype(np.int16)) > OCCLUDER_CONTRAST).astype(np.uint8)

    # Open away thin strokes, keep blobs over the minimum share of the canvas, and grow them by the margin.
    opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, make_ellipse(OCCLUDER_OPENING_PIXELS // 2))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    min_area = MIN_OCCLUDER_SHARE * gray.size
    kept = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] > min_area]
    blobs = np.isin(labels, kept).astype(np.uint8)
    return cv2.dilate(blobs, make_ellipse(OCCLUDER_GROW_PIXELS)) > 0


def grow_mask(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    """
    Grow a boolean mask by the radius.
    """
    return cv2.dilate(mask.astype(np.uint8), make_ellipse(radius_pixels)) > 0


def choose_alignment_pixels(baseline_darkness: np.ndarray, capture_darkness: np.ndarray, baseline_visible: np.ndarray,
                            capture_visible: np.ndarray) -> tuple[AlignmentRegion, np.ndarray, np.ndarray]:
    """
    Choose the pixels ECC aligns on: near the baseline's visible strokes and away from the capture's new ones, if it has enough.

    Args:
        baseline_darkness (np.ndarray): Darkness map of the baseline canvas.
        capture_darkness (np.ndarray): Darkness map of the capture's canvas, before alignment.
        baseline_visible (np.ndarray): Boolean mask of baseline pixels nothing hides.
        capture_visible (np.ndarray): Boolean mask of capture pixels nothing hides.

    Returns:
        tuple[AlignmentRegion, np.ndarray, np.ndarray]: Which region was chosen, and the baseline and capture masks ECC uses.
    """
    # Too few baseline strokes cannot hold a homography, so the whole visible canvas is used instead.
    baseline_strokes = (baseline_darkness > LOCAL_STROKE_CONTRAST) & baseline_visible
    if baseline_strokes.sum() < MIN_ECC_STROKE_PIXELS:
        return AlignmentRegion.WHOLE_CANVAS, baseline_visible, capture_visible

    # New strokes appear away from the old ones, since the corners already align within a few px, so they are left out.
    baseline_region = grow_mask(baseline_strokes, ECC_STROKE_REACH_PIXELS) & baseline_visible
    capture_strokes = capture_darkness > LOCAL_STROKE_CONTRAST
    new_strokes = capture_strokes & ~grow_mask(baseline_strokes, ECC_NEW_STROKE_DISTANCE_PIXELS)
    capture_region = capture_visible & ~grow_mask(new_strokes, ECC_NEW_STROKE_MARGIN_PIXELS)
    return AlignmentRegion.BASELINE_STROKES, baseline_region, capture_region


def align_to_baseline(baseline_darkness: np.ndarray, capture_darkness: np.ndarray, baseline_valid: np.ndarray,
                      capture_valid: np.ndarray) -> tuple[float | Unavailable, np.ndarray | None]:
    """
    Find the homography that maps baseline pixels to capture pixels with ECC on the darkness maps, never raising.

    Args:
        baseline_darkness (np.ndarray): Darkness map of the baseline canvas.
        capture_darkness (np.ndarray): Darkness map of the capture's canvas.
        baseline_valid (np.ndarray): Boolean mask of baseline pixels to use, false on occluders.
        capture_valid (np.ndarray): Boolean mask of capture pixels to use, false on occluders.

    Returns:
        tuple[float | Unavailable, np.ndarray | None]: The ECC correlation and warp, or the reason and None when ECC failed.
    """
    # Start from the identity, since the paper corners already map every capture into nearly the same frame.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPSILON)
    masks = (baseline_valid.astype(np.uint8) * 255, capture_valid.astype(np.uint8) * 255)
    try:
        correlation, warp = cv2.findTransformECCWithMask(baseline_darkness, capture_darkness, *masks, np.eye(3, dtype=np.float32),
                                                         cv2.MOTION_HOMOGRAPHY, criteria, ECC_GAUSSIAN_PIXELS)
    except cv2.error as error:
        return Unavailable(f"{NOT_CONVERGED_REASON}: {error.err.strip() or 'no reason given'}"), None

    # A NaN correlation or warp is a failure that ECC did not raise.
    if not (math.isfinite(correlation) and np.isfinite(warp).all()):
        return Unavailable(f"{NOT_CONVERGED_REASON}: non-finite result"), None

    return float(correlation), warp


def invert_warp(baseline_to_capture: np.ndarray) -> tuple[tuple[float, float, float], ...] | None:
    """
    Turn ECC's baseline-to-capture homography into the capture-to-baseline one, scaled so its last entry is 1.

    Args:
        baseline_to_capture (np.ndarray): 3x3 homography from ECC.

    Returns:
        tuple[tuple[float, float, float], ...] | None: The inverse row by row, or None when it cannot be inverted.
    """
    # A singular or degenerate homography has no usable inverse.
    try:
        inverse = np.linalg.inv(baseline_to_capture.astype(np.float64))
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(inverse).all() or abs(inverse[2, 2]) < 1e-12:
        return None

    inverse /= inverse[2, 2]
    return tuple(tuple(float(value) for value in row) for row in inverse)


def find_new_strokes(baseline_darkness: np.ndarray, aligned_darkness: np.ndarray, occluders: np.ndarray) -> np.ndarray:
    """
    Mark strokes in the aligned capture that are not near a baseline stroke or an occluder, without specks.

    Args:
        baseline_darkness (np.ndarray): Darkness map of the baseline canvas.
        aligned_darkness (np.ndarray): Darkness map of the capture warped into the baseline frame.
        occluders (np.ndarray): Boolean mask of occluders in the baseline frame, from either canvas.

    Returns:
        np.ndarray: Boolean mask of the new strokes.
    """
    # Strokes are pixels darker than their local paper, and old strokes are grown so a small misalignment is not new.
    baseline_strokes = (baseline_darkness > LOCAL_STROKE_CONTRAST).astype(np.uint8)
    near_baseline = cv2.dilate(baseline_strokes, make_ellipse(BASELINE_GROW_PIXELS)) > 0
    candidates = (aligned_darkness > LOCAL_STROKE_CONTRAST) & ~near_baseline & ~occluders

    # Drop specks, which camera noise and JPEG blocks leave.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8))
    kept = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= MIN_NEW_STROKE_SPECK_PIXELS]
    return np.isin(labels, kept)


def decode_mask(payload: bytes | None, shape: tuple[int, ...], label: str) -> np.ndarray:
    # Read a PNG mask as true where it is nonzero, refusing one of another size, and treat no mask as all false.
    if payload is None:
        return np.zeros(shape, dtype=bool)

    mask = decode_grayscale(payload)
    if mask.shape != shape:
        raise ValueError(f"{label} is {mask.shape[1]}x{mask.shape[0]}, not the baseline's {shape[1]}x{shape[0]}.")

    return mask > 0


def find_implausible_warp(correlation: float, baseline_to_capture: np.ndarray, width: int, height: int) -> Unavailable | None:
    # Map the canvas corners, which must stay in front of the camera and move no farther than a camera shift.
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]], dtype=np.float64)
    mapped = corners @ baseline_to_capture.astype(np.float64).T
    if (mapped[:, 2] <= 0).any():
        return Unavailable(f"{IMPLAUSIBLE_WARP_REASON}: correlation {correlation:.3f}, the canvas folds over")

    shift = float(np.linalg.norm(mapped[:, :2] / mapped[:, 2:] - corners[:, :2], axis=1).max())
    if correlation < MIN_ECC_CORRELATION or shift > MAX_CORNER_SHIFT_PIXELS:
        return Unavailable(f"{IMPLAUSIBLE_WARP_REASON}: correlation {correlation:.3f} (at least {MIN_ECC_CORRELATION} needed), a corner moves {shift:.1f} px (at most {MAX_CORNER_SHIFT_PIXELS})")  # noqa: E501

    return None


def measure_distance_to(mask: np.ndarray) -> np.ndarray:
    # Euclidean distance from every pixel to the nearest true pixel of the mask, very large when the mask is empty.
    return cv2.distanceTransform(np.where(mask, 0, 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def grow_box(mask: np.ndarray, reach_pixels: int) -> np.ndarray:
    # Mark the mask's bounding box grown by the reach on every side, clipped to the canvas.
    left, top, width, height = cv2.boundingRect(mask.astype(np.uint8))
    box = np.zeros_like(mask)
    box[max(0, top - reach_pixels):top + height + reach_pixels, max(0, left - reach_pixels):left + width + reach_pixels] = True
    return box


def measure_shares(distances: np.ndarray) -> dict[int, float]:
    # Share of the distances within each tolerance.
    return {tolerance: round(float((distances <= tolerance).mean()), SCORE_DECIMALS) for tolerance in TOLERANCES_PIXELS}


def measure_chamfer(distances: np.ndarray) -> float:
    # Mean of the distances, each capped so one far pixel cannot dominate.
    return round(float(np.minimum(distances, CHAMFER_CAP_PIXELS).mean()), SCORE_DECIMALS)


def score_part(index: int, part: np.ndarray, new_strokes: np.ndarray, hidden: np.ndarray) -> PartScores:
    # Measure how much of the part is hidden, and which new strokes lie near it.
    occluded_share = round(float((part & hidden).sum() / max(1, part.sum())), SCORE_DECIMALS)
    near = new_strokes & grow_box(part, PART_REACH_PIXELS)
    new_pixels_near = int(near.sum())

    # Coverage asks how much of the visible part has new strokes nearby, unknown when all of it is hidden.
    visible_skeleton = (find_skeleton(part.astype(np.uint8)) > 0) & ~hidden
    if visible_skeleton.any():
        to_new = measure_distance_to(new_strokes)[visible_skeleton]
        coverage = measure_shares(to_new)
        part_to_new = measure_chamfer(to_new) if new_strokes.any() else Unavailable(NO_NEW_STROKES_REASON)
    else:
        coverage = {tolerance: Unavailable(PART_OCCLUDED_REASON) for tolerance in TOLERANCES_PIXELS}
        part_to_new = Unavailable(PART_OCCLUDED_REASON)

    # Precision asks how much of the new drawing near the part lies on it, unknown when nothing new is near.
    if new_pixels_near:
        to_part = measure_distance_to(part)[near]
        precision, new_to_part = measure_shares(to_part), measure_chamfer(to_part)
    else:
        precision = {tolerance: Unavailable(NO_NEW_STROKES_NEAR_REASON) for tolerance in TOLERANCES_PIXELS}
        new_to_part = Unavailable(NO_NEW_STROKES_NEAR_REASON)

    return PartScores(part=index, occluded_share=occluded_share, new_pixels_near=new_pixels_near, coverage=coverage,
                      precision=precision, chamfer_part_to_new=part_to_new, chamfer_new_to_part=new_to_part)


class AlignedStrokeFollowDetector(FollowDetector):
    """
    Follow detector that aligns captures to the baseline with ECC on darkness maps and finds new strokes, OpenCV only.
    """

    @property
    def method(self) -> tuple[str, str]:
        """
        Return ("aligned-strokes", version).
        """
        return ALIGNED_STROKES_METHOD

    def describe_params(self) -> Mapping[str, ControlValue]:
        """
        Return every threshold of the darkness maps, occluders, ECC, and new strokes.
        """
        return {
            "stroke_contrast": LOCAL_STROKE_CONTRAST,
            "darkness_median_pixels": DARKNESS_MEDIAN_PIXELS,
            "darkness_blur_sigma": DARKNESS_BLUR_SIGMA,
            "paper_level_dilate_pixels": PAPER_LEVEL_DILATE_PIXELS,
            "paper_level_blur_sigma": PAPER_LEVEL_BLUR_SIGMA,
            "occluder_contrast": OCCLUDER_CONTRAST,
            "occluder_opening_pixels": OCCLUDER_OPENING_PIXELS,
            "min_occluder_share": MIN_OCCLUDER_SHARE,
            "occluder_grow_pixels": OCCLUDER_GROW_PIXELS,
            "ecc_motion": "homography",
            "ecc_iterations": ECC_ITERATIONS,
            "ecc_epsilon": ECC_EPSILON,
            "ecc_gaussian_pixels": ECC_GAUSSIAN_PIXELS,
            "ecc_stroke_reach_pixels": ECC_STROKE_REACH_PIXELS,
            "ecc_new_stroke_distance_pixels": ECC_NEW_STROKE_DISTANCE_PIXELS,
            "ecc_new_stroke_margin_pixels": ECC_NEW_STROKE_MARGIN_PIXELS,
            "min_ecc_stroke_pixels": MIN_ECC_STROKE_PIXELS,
            "baseline_grow_pixels": BASELINE_GROW_PIXELS,
            "min_new_stroke_speck_pixels": MIN_NEW_STROKE_SPECK_PIXELS,
            "min_ecc_correlation": MIN_ECC_CORRELATION,
            "max_corner_shift_pixels": MAX_CORNER_SHIFT_PIXELS,
            "filled_is_occluded": True,
            "tolerances_pixels": " ".join(str(tolerance) for tolerance in TOLERANCES_PIXELS),
            "part_reach_pixels": PART_REACH_PIXELS,
            "chamfer_cap_pixels": CHAMFER_CAP_PIXELS,
            "skeleton": "morphological",
            "score_decimals": SCORE_DECIMALS,
        }

    def score(self, baseline_canvas: bytes, capture_canvas: bytes, part_masks: Sequence[bytes], *,
              baseline_filled: bytes | None = None,
              capture_filled: bytes | None = None) -> tuple[Alignment, int | None, tuple[PartScores, ...]]:
        baseline, capture = decode_grayscale(baseline_canvas), decode_grayscale(capture_canvas)
        if baseline.shape != capture.shape:
            raise ValueError(f"Capture canvas is {capture.shape[1]}x{capture.shape[0]}, not the baseline's {baseline.shape[1]}x{baseline.shape[0]}.")  # noqa: E501

        parts = [decode_mask(mask, baseline.shape, f"Part {index} mask") for index, mask in enumerate(part_masks)]
        capture_filled_mask = decode_mask(capture_filled, baseline.shape, "Capture filled mask")

        # Hide occluders and filled areas in both canvases, and align near baseline strokes so no new stroke steers ECC either.
        baseline_hidden = find_occluders(baseline) | decode_mask(baseline_filled, baseline.shape, "Baseline filled mask")
        capture_hidden = find_occluders(capture) | capture_filled_mask
        occluded_share, filled_share = float(capture_hidden.mean()), float(capture_filled_mask.mean())
        baseline_darkness, capture_darkness = make_darkness_map(baseline), make_darkness_map(capture)
        aligned_on, *regions = choose_alignment_pixels(baseline_darkness, capture_darkness, ~baseline_hidden, ~capture_hidden)
        correlation, baseline_to_capture = align_to_baseline(baseline_darkness, capture_darkness, *regions)

        # Refuse a failed, implausible, or singular warp, recording why.
        height, width = baseline.shape
        failure = correlation if isinstance(correlation, Unavailable) else None
        if failure is None:
            failure = find_implausible_warp(correlation, baseline_to_capture, width, height)
        warp = invert_warp(baseline_to_capture) if failure is None else None
        if failure is None and warp is None:
            failure = Unavailable(f"{NOT_CONVERGED_REASON}: singular warp")
        if failure is not None:
            alignment = Alignment(converged=False, correlation=failure, warp=None, occluded_share=occluded_share,
                                  filled_share=filled_share, aligned_on=aligned_on)
            return alignment, None, ()

        # Warp the capture into the baseline frame, repeating its border so the edge does not look like a stroke.
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        aligned = cv2.warpPerspective(capture, baseline_to_capture, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)

        # The capture's filled area moves with it, and pixels the capture did not cover at all are unseen too.
        filled_flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        aligned_filled = cv2.warpPerspective(capture_filled_mask.astype(np.uint8), baseline_to_capture, (width, height),
                                             flags=filled_flags, borderMode=cv2.BORDER_CONSTANT, borderValue=1) > 0
        hidden = baseline_hidden | find_occluders(aligned) | aligned_filled

        # Find the new strokes on the pixels nothing hides, then score each part against them.
        new_strokes = find_new_strokes(baseline_darkness, make_darkness_map(aligned), hidden)
        scores = tuple(score_part(index, part, new_strokes, hidden) for index, part in enumerate(parts))
        alignment = Alignment(converged=True, correlation=correlation, warp=warp, occluded_share=occluded_share,
                              filled_share=filled_share, aligned_on=aligned_on)
        return alignment, int(new_strokes.sum()), scores


class FollowResult(StrEnum):
    """
    Whether a suggestion, or one of its parts, was followed.
    """
    FOLLOWED = "followed"
    NOT_FOLLOWED = "not_followed"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True, kw_only=True)
class FollowRule:
    """
    Thresholds that turn a window's recorded scores into followed, not followed, or undetermined.
    """
    coverage_tolerance_pixels: int = 16  # Tolerance whose coverage is compared with min_coverage
    min_coverage: float = 0.35  # Smallest share of the part's skeleton near new strokes for followed
    precision_tolerance_pixels: int = 16  # Tolerance whose precision is compared with min_precision
    min_precision: float = 0.4  # Smallest share of the new strokes near the part that lie on it for followed
    min_correlation: float = 0.3  # A capture aligned with a lower ECC correlation is not usable
    max_part_occluded_share: float = 0.5  # A capture hiding more of the part than this is not usable for it
    min_new_stroke_pixels: int = 60  # Fewer new-stroke pixels on the whole canvas than this means nothing new was drawn

    def __post_init__(self) -> None:
        # Tolerances are pixel counts, the shares are from 0 to 1, and the pixel minimum is a count.
        for name in ("coverage_tolerance_pixels", "precision_tolerance_pixels"):
            require(is_count(getattr(self, name), 1), f"Follow rule {name} must be an int >= 1, got {getattr(self, name)!r}.")
        for name in ("min_coverage", "min_precision", "min_correlation", "max_part_occluded_share"):
            check_unit_share(getattr(self, name), f"Follow rule {name}")
        pixels = self.min_new_stroke_pixels
        require(is_count(pixels, 0), f"Follow rule min_new_stroke_pixels must be an int >= 0, got {pixels!r}.")

    def describe(self) -> str:
        coverage, precision = self.coverage_tolerance_pixels, self.precision_tolerance_pixels
        followed = f"coverage_{coverage} >= {self.min_coverage} and precision_{precision} >= {self.min_precision}"
        return (f"{followed}, undetermined below correlation {self.min_correlation}, over {self.max_part_occluded_share} of a "
                f"part occluded, or under {self.min_new_stroke_pixels} new pixels")


PROVISIONAL_RULE: Final = FollowRule()  # Rule in force until the pilot's hand labels tune it, changed only with an ADR


@dataclass(frozen=True, slots=True, kw_only=True)
class PartDecision:
    """
    Whether one part of a suggestion was followed, and which capture's scores decided it.
    """
    part: int  # Part index
    result: FollowResult  # The part's result
    deciding_capture: int | None  # Capture whose scores decided it, None when no capture was usable
    reason: str | None  # Why it is undetermined, such as "nothing new drawn", None when it was decided

    def __post_init__(self) -> None:
        # A decided part names its capture, and only an undetermined one has a reason.
        require(is_count(self.part, 0), f"Decided part must be an int >= 0, got {self.part!r}.")
        require(self.result in list(FollowResult), f"Part {self.part} result must be one of {', '.join(FollowResult)}.")
        object.__setattr__(self, "result", FollowResult(self.result))
        capture = self.deciding_capture
        require(capture is None or is_count(capture, 1), f"Part {self.part} deciding_capture must be an int >= 1 or null.")
        undetermined = self.result == FollowResult.UNDETERMINED
        require(undetermined or capture is not None, f"Part {self.part} is {self.result} without a deciding capture.")
        has_reason = isinstance(self.reason, str) and bool(self.reason.strip())
        require(has_reason == undetermined, f"Part {self.part} has a reason exactly when it is undetermined.")


def combine_part_results(results: Sequence[FollowResult]) -> FollowResult:
    """
    Followed when any part is followed, otherwise not followed when any part is not, otherwise undetermined.
    """
    for result in (FollowResult.FOLLOWED, FollowResult.NOT_FOLLOWED):
        if result in results:
            return result

    return FollowResult.UNDETERMINED


@dataclass(frozen=True, slots=True, kw_only=True)
class FollowDecision:
    """
    Whether a suggestion was followed, decided part by part from its window's scores, with the method and rule used.
    """
    method: str  # Detector name, such as "aligned-strokes"
    method_version: str  # Detector version whose scores were decided
    rule: FollowRule  # Rule that produced the results
    result: FollowResult  # The suggestion's result, combined from its parts
    parts: tuple[PartDecision, ...]  # One per part in index order
    reason: str | None = None  # Why a suggestion without parts is undetermined, None otherwise

    def __post_init__(self) -> None:
        # The method is named, the rule is a rule, and the parts are in index order.
        require(isinstance(self.method, str) and bool(self.method.strip()), "Follow decision needs a method name.")
        require(isinstance(self.method_version, str) and bool(self.method_version.strip()), "Follow decision needs a version.")
        require(isinstance(self.rule, FollowRule), f"Follow decision rule must be a FollowRule, got {self.rule!r}.")
        object.__setattr__(self, "parts", tuple(self.parts))
        indexes = [part.part if isinstance(part, PartDecision) else None for part in self.parts]
        require(indexes == list(range(len(indexes))), f"Part decisions must be PartDecision numbered from 0, got {indexes}.")

        # The result is the parts' combination, and only a suggestion without parts has a reason of its own.
        require(self.result in list(FollowResult), f"Follow decision result must be one of {', '.join(FollowResult)}.")
        object.__setattr__(self, "result", FollowResult(self.result))
        expected = combine_part_results([part.result for part in self.parts])
        require(self.result == expected, f"Follow decision result {self.result} must combine its parts into {expected}.")
        expected_reason = None if self.parts else NO_PARTS_REASON
        require(self.reason == expected_reason, f"Follow decision reason must be {expected_reason!r}, got {self.reason!r}.")


def read_tolerance(scores: Mapping[int, float | Unavailable], tolerance: int, label: str) -> float | Unavailable:
    """
    Return the score at a rule's tolerance, refusing a tolerance the scores were not measured at.
    """
    if tolerance not in scores:
        measured = ", ".join(str(value) for value in sorted(scores))
        raise ValueError(f"The rule's {label} tolerance of {tolerance} px was not measured. Use one of {measured} px.")

    return scores[tolerance]


def find_usable_part(scores: CaptureScores, part: int, rule: FollowRule) -> PartScores | None:
    """
    Return a capture's scores of one part when the capture aligned well enough and the part was visible enough, else None.

    Args:
        scores (CaptureScores): A capture's scores against the suggestion's baseline.
        part (int): Part index.
        rule (FollowRule): Rule whose correlation and occlusion limits apply.

    Returns:
        PartScores | None: The part's scores, or None when this capture cannot decide the part.
    """
    alignment = scores.alignment
    if not alignment.converged or alignment.correlation < rule.min_correlation:
        return None

    # Scores saved before per-part scoring, or of fewer parts, cannot decide this part.
    if scores.parts is None or part >= len(scores.parts):
        return None

    part_scores = scores.parts[part]
    coverage = read_tolerance(part_scores.coverage, rule.coverage_tolerance_pixels, "coverage")
    visible = part_scores.occluded_share <= rule.max_part_occluded_share and not isinstance(coverage, Unavailable)
    return part_scores if visible else None


def decide_part(part: int, window: Sequence[tuple[int, CaptureScores]], rule: FollowRule) -> PartDecision:
    """
    Decide one part from the latest usable capture of its window, since new strokes accumulate against the baseline.

    Args:
        part (int): Part index.
        window (Sequence[tuple[int, CaptureScores]]): Capture numbers and scores in the window, in capture order.
        rule (FollowRule): Rule to apply.

    Returns:
        PartDecision: The part's result, its deciding capture, and the reason when undetermined.
    """
    usable = [(number, scores, found) for number, scores in window if (found := find_usable_part(scores, part, rule))]
    if not usable:
        return PartDecision(part=part, result=FollowResult.UNDETERMINED, deciding_capture=None, reason=NO_USABLE_CAPTURE_REASON)

    # Too few new strokes anywhere means the person has not drawn yet, which says nothing about the part.
    number, scores, part_scores = usable[-1]
    if scores.new_stroke_pixels < rule.min_new_stroke_pixels:
        return PartDecision(part=part, result=FollowResult.UNDETERMINED, deciding_capture=number, reason=NOTHING_NEW_REASON)

    # Followed needs enough of the part drawn over and enough of the nearby new drawing on the part.
    coverage = read_tolerance(part_scores.coverage, rule.coverage_tolerance_pixels, "coverage")
    precision = read_tolerance(part_scores.precision, rule.precision_tolerance_pixels, "precision")
    precise = not isinstance(precision, Unavailable) and precision >= rule.min_precision
    followed = coverage >= rule.min_coverage and precise
    result = FollowResult.FOLLOWED if followed else FollowResult.NOT_FOLLOWED
    return PartDecision(part=part, result=result, deciding_capture=number, reason=None)


def decide_follow(window: Sequence[tuple[int, CaptureScores]], part_count: int, rule: FollowRule,
                  method: tuple[str, str]) -> FollowDecision:
    """
    Decide whether a suggestion was followed from the scores recorded in its window, without reading any image.

    Args:
        window (Sequence[tuple[int, CaptureScores]]): Number and scores of each scored capture in the window.
        part_count (int): Number of parts the suggestion had.
        rule (FollowRule): Rule to apply, the recorded one or any other for a reanalysis.
        method (tuple[str, str]): Name and version of the detector that recorded the scores.

    Returns:
        FollowDecision: Each part's result and the suggestion's combined result.
    """
    ordered = sorted(window, key=lambda entry: entry[0])
    parts = tuple(decide_part(part, ordered, rule) for part in range(part_count))
    result = combine_part_results([part.result for part in parts])
    name, version = method
    return FollowDecision(method=name, method_version=version, rule=rule, result=result, parts=parts,
                          reason=None if parts else NO_PARTS_REASON)
