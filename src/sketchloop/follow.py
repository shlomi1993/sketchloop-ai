"""
Follow detection: align each later capture of a drawing to a suggestion's baseline canvas and find the new strokes.
"""

import cv2
import math
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from sketchloop.domain import ControlValue, Unavailable, freeze_controls, require, require_duration
from sketchloop.preprocessing import LOCAL_STROKE_CONTRAST, decode_grayscale
from sketchloop.stroke_suggestion import is_count


ALIGNED_STROKES_METHOD: Final = ("aligned-strokes", "1")  # Detector name and version, bumped whenever its results change
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
BASELINE_GROW_PIXELS: Final = 4  # Baseline strokes are grown by this much, so a slightly misaligned old stroke is not new
MIN_NEW_STROKE_SPECK_PIXELS: Final = 20  # New-stroke specks smaller than this are noise
NOT_CONVERGED_REASON: Final = "ECC did not converge"


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Alignment:
    """
    How a capture's canvas was aligned to a suggestion's baseline canvas, and how much of it an occluder covered.
    """
    converged: bool  # Whether ECC converged
    correlation: float | Unavailable  # ECC correlation, unavailable with the reason when ECC did not converge
    warp: tuple[tuple[float, float, float], ...] | None  # 3x3 homography from capture to baseline pixels, None without one
    occluded_share: float  # Share of the capture's canvas masked as an occluder, such as a hand

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


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureScores:
    """
    A capture measured against the open suggestion's baseline canvas: its alignment and how much was drawn since.
    """
    suggestion: int  # Number of the suggestion whose window the capture fell in
    seconds_since_shown: float  # Seconds from when the suggestion was shown to the capture
    alignment: Alignment  # Alignment to the suggestion's baseline canvas
    new_stroke_pixels: int | None  # New-stroke pixels on the whole canvas, None when alignment failed

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
    def align(self, baseline_canvas: bytes, capture_canvas: bytes) -> tuple[Alignment, int | None]:
        """
        Align a capture's canvas to the baseline canvas and count the new-stroke pixels, never raising for a failed alignment.

        Args:
            baseline_canvas (bytes): PNG of the suggestion's baseline canvas.
            capture_canvas (bytes): PNG of the capture's canvas, the same size.

        Returns:
            tuple[Alignment, int | None]: The alignment, and the new-stroke pixels or None when alignment failed.
        """
        ...


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
            "baseline_grow_pixels": BASELINE_GROW_PIXELS,
            "min_new_stroke_speck_pixels": MIN_NEW_STROKE_SPECK_PIXELS,
        }

    def align(self, baseline_canvas: bytes, capture_canvas: bytes) -> tuple[Alignment, int | None]:
        """
        Align a capture's canvas to the baseline canvas with occluders masked, and count the new strokes.
        """
        baseline, capture = decode_grayscale(baseline_canvas), decode_grayscale(capture_canvas)
        if baseline.shape != capture.shape:
            raise ValueError(f"Capture canvas is {capture.shape[1]}x{capture.shape[0]}, not the baseline's {baseline.shape[1]}x{baseline.shape[0]}.")  # noqa: E501

        # Mask occluders in both canvases, so a hand neither steers the alignment nor counts as strokes.
        baseline_occluders, capture_occluders = find_occluders(baseline), find_occluders(capture)
        occluded_share = float(capture_occluders.mean())
        baseline_darkness = make_darkness_map(baseline)
        correlation, baseline_to_capture = align_to_baseline(baseline_darkness, make_darkness_map(capture), ~baseline_occluders,
                                                             ~capture_occluders)
        warp = invert_warp(baseline_to_capture) if baseline_to_capture is not None else None
        if warp is None:
            singular = Unavailable(f"{NOT_CONVERGED_REASON}: singular warp")
            reason = correlation if isinstance(correlation, Unavailable) else singular
            return Alignment(converged=False, correlation=reason, warp=None, occluded_share=occluded_share), None

        # Warp the capture into the baseline frame, repeating its border so the edge does not look like a stroke.
        height, width = baseline.shape
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        aligned = cv2.warpPerspective(capture, baseline_to_capture, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        occluders = baseline_occluders | find_occluders(aligned)
        new_strokes = find_new_strokes(baseline_darkness, make_darkness_map(aligned), occluders)
        alignment = Alignment(converged=True, correlation=correlation, warp=warp, occluded_share=occluded_share)
        return alignment, int(new_strokes.sum())
