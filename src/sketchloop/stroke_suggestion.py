"""
Next-stroke suggester interface and its request, part, suggestion, and output records.
"""

import hashlib

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from sketchloop.domain import (BackendIdentity, ColorMode, ControlValue, EffectiveSettings, ImageRef, SketchLoopError,
                               Unavailable, freeze_controls, require, require_duration)
from sketchloop.generation import ControlSpec


MAX_PARTS: Final = 4

# Overlay colors by part index, from the Okabe-Ito palette, which stays readable with color blindness.
PART_COLORS: Final = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
CANVAS_GRID_PIXELS: Final = 8  # Canvas sides are multiples of this, the latent grid of Stable Diffusion 1.5
LINE_DRAWING_FILE: Final = "line-drawing.png"  # Bare file name of a suggestion's line drawing, placed by the caller


class StrokeSuggestionError(SketchLoopError, RuntimeError):
    """
    The suggestion model cannot be loaded, or it fails while suggesting.
    """


def name_part_file(index: int) -> str:
    """
    Return the bare file name of a part's mask, numbered from 1, such as part-1.png.
    """
    return f"part-{index + 1}.png"


def is_count(value: object, minimum: int) -> bool:
    """
    Tell whether a value is an int, not a bool, of at least the minimum.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


@dataclass(frozen=True, slots=True, kw_only=True)
class StrokeSuggestionRequest:
    """
    Everything a stroke suggester needs to suggest new parts for a drawing in progress.
    """
    canvas: ImageRef  # Paper-frame strokes the suggester receives, black on white, mode L, sides divisible by 8
    prompt: str  # Prompt box text exactly as the participant typed it, possibly empty
    seed: int | None = None  # Seed, or None for a random one that the result records
    controls: Mapping[str, ControlValue] = field(default_factory=dict)  # Requested controls, empty for the defaults
    max_parts: int = MAX_PARTS  # Most parts to return, from 1 to MAX_PARTS

    def __post_init__(self) -> None:
        # The suggester draws on a grayscale canvas whose sides fit the model's latent grid.
        width, height = self.canvas.width, self.canvas.height
        require(self.canvas.mode == ColorMode.L, f"Suggestion canvas must be mode L, got {self.canvas.mode}.")
        fits_grid = width % CANVAS_GRID_PIXELS == 0 and height % CANVAS_GRID_PIXELS == 0
        require(fits_grid, f"Suggestion canvas sides must be multiples of {CANVAS_GRID_PIXELS}, got {width}x{height}.")

        # Keep the prompt exactly as typed, empty included, since the participant alone states what they draw.
        require(isinstance(self.prompt, str), "Suggestion prompt must be text, possibly empty.")
        require(self.seed is None or is_count(self.seed, 0), "seed must be None or an int >= 0.")
        valid_limit = is_count(self.max_parts, 1) and self.max_parts <= MAX_PARTS
        require(valid_limit, f"max_parts must be an int from 1 to {MAX_PARTS}, got {self.max_parts!r}.")
        object.__setattr__(self, "controls", freeze_controls(self.controls))


@dataclass(frozen=True, slots=True, kw_only=True)
class SuggestedPart:
    """
    One group of suggested strokes, with its mask, bounding box, and size.
    """
    index: int  # 0-based position, largest part first
    mask: ImageRef  # Mode L PNG at the canvas size, 255 on the part's strokes
    box: tuple[int, int, int, int]  # Left, top, width, and height in canvas pixels
    pixel_count: int  # Pixels in the mask, at least 1
    skeleton_pixels: int  # Pixels in the mask's one-pixel skeleton, so a size-scaled tolerance can be computed later
    label: str | None = None  # Part name from a labeler, always None in T09

    def __post_init__(self) -> None:
        require(is_count(self.index, 0), f"Part index must be an int >= 0, got {self.index!r}.")
        require(self.mask.mode == ColorMode.L, f"Part {self.index} mask must be mode L, got {self.mask.mode}.")

        # Hold the box as a tuple, since JSON loads it as a list, and keep it inside the mask.
        is_box = isinstance(self.box, tuple | list) and len(self.box) == 4 and all(is_count(value, 0) for value in self.box)
        require(is_box, f"Part {self.index} box must be four ints >= 0, got {self.box!r}.")
        object.__setattr__(self, "box", tuple(self.box))
        left, top, width, height = self.box
        inside = width >= 1 and height >= 1 and left + width <= self.mask.width and top + height <= self.mask.height
        require(inside, f"Part {self.index} box {self.box} must lie inside its {self.mask.width}x{self.mask.height} mask.")

        # A part has strokes, and its skeleton is a subset of them.
        area = self.mask.width * self.mask.height
        valid_pixels = is_count(self.pixel_count, 1) and self.pixel_count <= area
        require(valid_pixels, f"Part {self.index} pixel_count must be an int from 1 to {area}, got {self.pixel_count!r}.")
        skeleton = self.skeleton_pixels
        valid_skeleton = is_count(skeleton, 1) and skeleton <= self.pixel_count
        require(valid_skeleton, f"Part {self.index} skeleton_pixels must be an int from 1 to the pixel count, got {skeleton!r}.")
        has_label = self.label is None or (isinstance(self.label, str) and bool(self.label.strip()))
        require(has_label, f"Part {self.index} label must be None or non-empty text.")


@dataclass(frozen=True, slots=True, kw_only=True)
class StrokeSuggestion:
    """
    One suggestion's parts, the suggester that made them, and the settings, seed, and time it used.
    """
    suggester: BackendIdentity  # Adapter, version, execution kind, and model IDs with revisions
    effective: EffectiveSettings  # Full prompt, negative prompt, and every setting and extraction parameter used
    seed: int | Unavailable  # Seed actually used
    line_drawing: ImageRef | Unavailable  # The model's whole output, kept because it cannot be regenerated exactly elsewhere
    parts: tuple[SuggestedPart, ...]  # 0 to max_parts new parts, largest first
    timings_seconds: Mapping[str, float]  # generation and extraction, without any model load

    def __post_init__(self) -> None:
        # Parts are numbered in display order without gaps, and no suggestion has more than the overlay has colors.
        object.__setattr__(self, "parts", tuple(self.parts))
        indexes = [part.index for part in self.parts]
        require(len(self.parts) <= MAX_PARTS, f"A suggestion has at most {MAX_PARTS} parts, got {len(self.parts)}.")
        in_order = indexes == list(range(len(self.parts)))
        require(in_order, f"Part indexes must be 0..{len(self.parts) - 1} in order, got {indexes}.")

        # Every image has the canvas size, so masks and the line drawing overlay the canvas exactly.
        sizes = {(image.width, image.height) for image in self.images}
        require(len(sizes) <= 1, f"The line drawing and masks must share one size, got {sorted(sizes)}.")
        paths = [image.path for image in self.images]
        require(len(set(paths)) == len(paths), f"Suggestion image paths must be unique, got {paths}.")

        # Check the seed and timings, then freeze the timings so the record cannot change after the fact.
        valid_seed = isinstance(self.seed, Unavailable) or is_count(self.seed, 0)
        require(valid_seed, f"Suggestion seed must be an int >= 0 or Unavailable, got {self.seed!r}.")
        for stage, seconds in self.timings_seconds.items():
            require_duration(seconds, f"timings_seconds.{stage}")
        object.__setattr__(self, "timings_seconds", MappingProxyType(dict(self.timings_seconds)))

    @property
    def images(self) -> tuple[ImageRef, ...]:
        """
        Return the stored images, the line drawing when kept and then each part's mask.
        """
        line_drawing = () if isinstance(self.line_drawing, Unavailable) else (self.line_drawing,)
        return (*line_drawing, *(part.mask for part in self.parts))


@dataclass(frozen=True, slots=True, kw_only=True)
class StrokeSuggestionOutput:
    """
    A suggestion plus the bytes of its line drawing and masks, checked against their checksums, and any model load time.
    """
    suggestion: StrokeSuggestion  # The suggestion record
    payloads: Mapping[str, bytes]  # Bytes of the line drawing and each mask by path, checked against their checksums
    load_seconds: float | None = None  # Seconds this call spent loading models, None when it loaded none

    def __post_init__(self) -> None:
        # A reported load time is finite seconds of at least 0.
        if self.load_seconds is not None:
            require_duration(self.load_seconds, "load_seconds")

        # Freeze the payloads, then require exactly one payload per stored image.
        object.__setattr__(self, "payloads", MappingProxyType(dict(self.payloads)))
        images = {image.path: image for image in self.suggestion.images}
        require(set(self.payloads) == set(images), "Payload keys must equal the line drawing and mask paths.")

        # Catch a suggester whose bytes do not match the checksum it recorded.
        for path, payload in self.payloads.items():
            require(hashlib.sha256(payload).hexdigest() == images[path].sha256, f"Payload {path} fails its checksum.")


class StrokeSuggester(ABC):
    """
    Base class every next-stroke suggester subclasses, separate from generation, prompt suggestions, and follow detection.
    """

    @property
    @abstractmethod
    def controls(self) -> tuple[ControlSpec, ...]:
        """
        Return the controls the suggester accepts, with their bounds and defaults.
        """
        ...

    @abstractmethod
    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Validate the request and suggest new parts for the drawing on the canvas.

        Args:
            request (StrokeSuggestionRequest): Request to suggest for.
            canvas_payload (bytes): PNG bytes of the request's canvas.

        Returns:
            StrokeSuggestionOutput: The suggestion and the bytes of its images.
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """
        Say in one sentence which method, model, device, and precision suggest strokes, for the terminal and the page.
        """
        ...

    def load(self, width: int, height: int) -> None:
        """
        Load and warm up any model at the canvas size, which suggesters without a model skip.

        Args:
            width (int): Canvas width in pixels.
            height (int): Canvas height in pixels.
        """
