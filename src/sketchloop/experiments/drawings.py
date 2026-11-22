"""
Drawing records of guided sketching, with the checks that keep them valid and their JSON form in drawing.json.
"""

import math

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from sketchloop.domain import (ColorMode, ControlValue, ImageRef, Rotation, SketchSource, Unavailable, first_line, require,
                               require_duration)
from sketchloop.experiments.provenance import redact_paths
from sketchloop.experiments.serialization import (backend_from_dict, backend_to_dict, check_schema_version, decode_value,
                                                  effective_from_dict, effective_to_dict, encode_value, image_from_dict,
                                                  parse_time)
from sketchloop.follow import Alignment, CaptureScores, FollowMethod
from sketchloop.preprocessing import PreprocessingStep
from sketchloop.stroke_suggestion import (CANVAS_GRID_PIXELS, PART_COLORS, StrokeSuggestion, StrokeSuggestionRequest,
                                          SuggestedPart, is_count)


DRAWING_FILE: Final = "drawing.json"


class CaptureTrigger(StrEnum):
    """
    Why a capture was taken: the save interval, a suggestion press, or the end of the drawing.
    """
    INTERVAL = "interval"
    SUGGESTION = "suggestion"
    FINISH = "finish"


class DrawingEnd(StrEnum):
    """
    How a drawing ended: the person finished it, or a new session replaced it.
    """
    FINISHED = "finished"
    NEW_SESSION = "new_session"


class SuggestionStatus(StrEnum):
    """
    Whether a suggestion was shown or failed.
    """
    SHOWN = "shown"
    FAILED = "failed"


class SuggestionStage(StrEnum):
    """
    A timed stage of a capture or suggestion, recorded as a stage timing and as the stage a failure happened in.
    """
    CANVAS = "canvas"
    MODEL_LOAD = "model_load"
    GENERATION = "generation"
    EXTRACTION = "extraction"
    OVERLAY = "overlay"
    SCORING = "scoring"
    SAVING = "saving"


def is_aware(moment: object) -> bool:
    """
    Tell whether a value is a timezone-aware datetime.
    """
    return isinstance(moment, datetime) and moment.utcoffset() is not None


def freeze_timings(timings_seconds: Mapping[str, float], label: str) -> Mapping[str, float]:
    """
    Check that every stage time is finite seconds of at least 0, and return a read-only copy.
    """
    for stage, seconds in timings_seconds.items():
        require_duration(seconds, f"{label} {stage} time")

    return MappingProxyType(dict(timings_seconds))


@dataclass(frozen=True, slots=True, kw_only=True)
class StageFailure:
    """
    Where and how a capture or suggestion failed.
    """
    stage: SuggestionStage  # Stage that raised the error
    error: str  # Error class name
    message: str  # First line of the error message, with paths redacted

    def __post_init__(self) -> None:
        # Hold a known stage as the enum, so a stage loaded from JSON text matches a recorded one.
        stages = ", ".join(SuggestionStage)
        require(self.stage in list(SuggestionStage), f"Failure stage must be one of {stages}, got {self.stage!r}.")
        object.__setattr__(self, "stage", SuggestionStage(self.stage))


def describe_stage_failure(stage: SuggestionStage, error: BaseException) -> StageFailure:
    """
    Summarize an exception as a stage failure with its class name and the first line of its message, paths redacted.
    """
    return StageFailure(stage=stage, error=type(error).__name__, message=redact_paths(first_line(error)))


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureRecord:
    """
    One saved frame of a drawing: when and why it was taken, its raw image, and its paper-frame canvas.
    """
    number: int  # Capture number in the drawing, starting at 1
    trigger: CaptureTrigger  # interval, suggestion, or finish
    at: datetime  # Timezone-aware wall-clock time of the capture
    seconds_since_start: float  # Seconds from the drawing's start on a monotonic clock
    source: SketchSource  # Where the frame came from, such as webcam or upload
    frame_age_seconds: float | Unavailable  # Seconds from the webcam frame's arrival to the capture
    rotation: int  # Clockwise rotation in degrees applied before making the canvas
    raw: ImageRef  # Raw frame as received, under captures/
    canvas: ImageRef | None  # Paper-frame canvas at the drawing's canvas size, None when making it failed
    steps: tuple[PreprocessingStep, ...]  # Steps that made the canvas, empty when it failed
    timings_seconds: Mapping[str, float]  # Duration by stage, holding only stages that ran and were timed
    failure: StageFailure | None  # Why the canvas could not be made, None when it was
    follow: CaptureScores | None = None  # Alignment to the open suggestion's baseline, None outside a window or without a canvas

    def __post_init__(self) -> None:
        # Number, time, and trigger place the capture in the drawing.
        require(is_count(self.number, 1), f"Capture number must be an int >= 1, got {self.number!r}.")
        trigger = self.trigger
        require(trigger in list(CaptureTrigger), f"Capture trigger must be interval, suggestion, or finish, got {trigger!r}.")
        object.__setattr__(self, "trigger", CaptureTrigger(self.trigger))
        require(is_aware(self.at), f"Capture {self.number} time must be timezone-aware.")
        require_duration(self.seconds_since_start, f"Capture {self.number} seconds_since_start")

        # Only a streamed webcam frame has an age, and the rotation is one preprocessing can apply.
        source = self.source
        require(source in list(SketchSource), f"Capture source must be file, camera, webcam, or upload, got {source!r}.")
        object.__setattr__(self, "source", SketchSource(self.source))
        if not isinstance(self.frame_age_seconds, Unavailable):
            require_duration(self.frame_age_seconds, "Frame age")
            require(self.source == SketchSource.WEBCAM, f"Only a webcam capture has a frame age, got source {self.source!r}.")
        require(self.rotation in list(Rotation), f"Capture rotation must be 0, 90, 180, or 270, got {self.rotation!r}.")

        # A capture has a grayscale canvas exactly when making it did not fail.
        has_one = (self.canvas is None) == (self.failure is not None)
        require(has_one, f"Capture {self.number} needs a canvas or a failure, not both.")
        require(self.canvas is None or self.canvas.mode == ColorMode.L, f"Capture {self.number} canvas must be mode L.")
        require(self.follow is None or self.canvas is not None, f"Capture {self.number} has follow scores without a canvas.")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "timings_seconds", freeze_timings(self.timings_seconds, f"Capture {self.number}"))


@dataclass(frozen=True, slots=True, kw_only=True)
class SuggestionRecord:
    """
    One press of the suggest button: its baseline capture, request, result, overlay, timings, and any failure.
    """
    number: int  # Suggestion number in the drawing, starting at 1
    id: str  # Unique suggestion ID
    status: SuggestionStatus  # shown, or failed with a failure record
    requested_at: datetime  # Timezone-aware time of the press
    shown_at: datetime | None  # Timezone-aware time the overlay was ready, None when the suggestion failed
    baseline_capture: int  # Number of the capture taken at the press
    request: StrokeSuggestionRequest | None  # What was asked, None when the canvas failed
    result: StrokeSuggestion | None  # What the suggester returned, None when the suggestion failed
    part_colors: tuple[str, ...]  # Overlay color of each part, which is what the person saw
    overlay: ImageRef | None  # RGBA overlay at the canvas size, None when the suggestion failed
    timings_seconds: Mapping[str, float]  # Duration by stage, holding only stages that ran and were timed
    failure: StageFailure | None  # Why the suggestion failed, None when it was shown

    def __post_init__(self) -> None:
        # Number, ID, times, and the baseline capture identify the press.
        require(is_count(self.number, 1), f"Suggestion number must be an int >= 1, got {self.number!r}.")
        require(isinstance(self.id, str) and bool(self.id.strip()), f"Suggestion {self.number} needs a non-empty ID.")
        require(self.status in list(SuggestionStatus), f"Suggestion status must be shown or failed, got {self.status!r}.")
        object.__setattr__(self, "status", SuggestionStatus(self.status))
        require(is_aware(self.requested_at), f"Suggestion {self.number} requested_at must be timezone-aware.")
        require(is_count(self.baseline_capture, 1), f"Suggestion {self.number} baseline_capture must be an int >= 1.")

        # A shown suggestion has a request, result, overlay, and time shown, and a failed one has only its failure.
        shown = self.status == SuggestionStatus.SHOWN
        shown_fields = (self.request, self.result, self.overlay, self.shown_at)
        require(shown == (self.failure is None), f"Suggestion {self.number} must be shown exactly when it has no failure.")
        require(not shown or None not in shown_fields, f"Shown suggestion {self.number} needs a request, result, and overlay.")
        failed_fields = (self.result, self.overlay, self.shown_at)
        require(shown or failed_fields == (None, None, None), f"Failed suggestion {self.number} cannot have a result or overlay.")
        canvas_failed = self.failure is not None and self.failure.stage == SuggestionStage.CANVAS
        require(self.request is not None or canvas_failed, f"Suggestion {self.number} lacks a request without a canvas failure.")
        require(self.shown_at is None or (is_aware(self.shown_at) and self.shown_at >= self.requested_at),
                f"Suggestion {self.number} shown_at must be timezone-aware and not before requested_at.")

        # The result fits the request: no more parts than asked for, every image on the canvas, and one color per part.
        object.__setattr__(self, "part_colors", tuple(self.part_colors))
        expected_colors = PART_COLORS[:len(self.result.parts)] if self.result else ()
        colors = self.part_colors
        require(colors == expected_colors, f"Suggestion {self.number} part colors must be {expected_colors}, got {colors}.")
        if self.result is not None:
            check_result_fits_request(self.number, self.request, self.result, self.overlay)
        object.__setattr__(self, "timings_seconds", freeze_timings(self.timings_seconds, f"Suggestion {self.number}"))


def check_result_fits_request(number: int, request: StrokeSuggestionRequest, result: StrokeSuggestion, overlay: ImageRef) -> None:
    """
    Refuse a result with more parts than requested, or a mask, line drawing, or overlay that is not the canvas size.

    Args:
        number (int): Suggestion number, for messages.
        request (StrokeSuggestionRequest): Request the result answers.
        result (StrokeSuggestion): Suggester's result.
        overlay (ImageRef): Overlay drawn from the result.
    """
    parts, limit = len(result.parts), request.max_parts
    require(parts <= limit, f"Suggestion {number} has {parts} parts, more than the {limit} requested.")
    canvas_size = (request.canvas.width, request.canvas.height)
    for image in (*result.images, overlay):
        size = (image.width, image.height)
        require(size == canvas_size, f"Suggestion {number} image {image.path} is {size[0]}x{size[1]}, not the {canvas_size[0]}x{canvas_size[1]} canvas.")  # noqa: E501
    require(overlay.mode == ColorMode.RGBA, f"Suggestion {number} overlay must be mode RGBA, got {overlay.mode}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class DrawingRecord:
    """
    Contents of drawing.json: one drawing from its start to its end, with its captures and suggestions.
    """
    schema_version: int  # Record format version
    number: int  # Drawing number in the session, starting at 1
    drawing_id: str  # Unique drawing ID
    started_at: datetime  # Timezone-aware start time
    ended_at: datetime | None  # Timezone-aware end time, None while the drawing is open
    end: DrawingEnd | None  # finished or new_session, None while the drawing is open
    canvas_width: int  # Paper-frame canvas width in pixels, a multiple of 8
    canvas_height: int  # Paper-frame canvas height in pixels, a multiple of 8
    save_interval_seconds: float  # Seconds between interval captures while the drawing is open
    suggestions_enabled: bool  # Whether the drawing offers suggestions, false for a no-AI condition
    suggester: str | None  # Suggester class name, None when suggestions are off
    captures: tuple[CaptureRecord, ...]  # Captures in order
    suggestions: tuple[SuggestionRecord, ...]  # Suggestions in order, shown or failed
    final_capture: int | None  # Number of the capture taken at finish, None while open or when finished without a frame
    follow_detector: FollowMethod | None = None  # Follow method and parameters, None with suggestions off or in older drawings

    def __post_init__(self) -> None:
        # Identity, times, and the end are consistent.
        require(is_count(self.number, 1), f"Drawing number must be an int >= 1, got {self.number!r}.")
        require(isinstance(self.drawing_id, str) and bool(self.drawing_id.strip()), "Drawing ID must be non-empty.")
        require(is_aware(self.started_at), "Drawing started_at must be timezone-aware.")
        require((self.ended_at is None) == (self.end is None), "A drawing has an end time exactly when it has an end.")
        require(self.ended_at is None or is_aware(self.ended_at), "Drawing ended_at must be timezone-aware.")
        if self.end is not None:
            require(self.end in list(DrawingEnd), f"Drawing end must be one of {', '.join(DrawingEnd)}, got {self.end!r}.")
            object.__setattr__(self, "end", DrawingEnd(self.end))

        # The canvas fits the latent grid, and the interval is a positive number of seconds.
        sides = (self.canvas_width, self.canvas_height)
        on_grid = all(is_count(side, CANVAS_GRID_PIXELS) and side % CANVAS_GRID_PIXELS == 0 for side in sides)
        require(on_grid, f"Drawing canvas sides must be multiples of {CANVAS_GRID_PIXELS}, got {sides}.")
        interval = self.save_interval_seconds
        is_number = isinstance(interval, int | float) and not isinstance(interval, bool)
        valid_interval = is_number and math.isfinite(interval) and interval > 0
        require(valid_interval, f"Drawing save_interval_seconds must be finite seconds > 0, got {interval!r}.")

        # A drawing with suggestions off names no suggester and has no suggestions.
        require(isinstance(self.suggestions_enabled, bool), "Drawing suggestions_enabled must be true or false.")
        names_suggester = isinstance(self.suggester, str) and bool(self.suggester.strip())
        require(names_suggester == self.suggestions_enabled, "A drawing names a suggester exactly when suggestions are on.")
        object.__setattr__(self, "captures", tuple(self.captures))
        object.__setattr__(self, "suggestions", tuple(self.suggestions))
        require(self.suggestions_enabled or not self.suggestions, "A drawing with suggestions off cannot have suggestions.")
        self._check_captures()
        self._check_suggestions()
        self._check_follow()

    def _check_captures(self) -> None:
        # Captures are numbered without gaps, and every canvas has the drawing's canvas size.
        numbers = [capture.number for capture in self.captures]
        require(numbers == list(range(1, len(numbers) + 1)), f"Capture numbers must run from 1 without gaps, got {numbers}.")
        canvas_size = (self.canvas_width, self.canvas_height)
        for capture in self.captures:
            size = None if capture.canvas is None else (capture.canvas.width, capture.canvas.height)
            require(size in (None, canvas_size), f"Capture {capture.number} canvas is {size}, not the drawing's {canvas_size}.")

        # The final capture, when there is one, was taken at finish.
        if self.final_capture is not None:
            exists = is_count(self.final_capture, 1) and self.final_capture <= len(self.captures)
            final = self.final_capture
            require(exists and self.end is not None, f"Final capture {final!r} must be a capture of an ended drawing.")
            trigger = self.captures[final - 1].trigger
            require(trigger == CaptureTrigger.FINISH, f"Final capture {final} must have trigger finish, got {trigger}.")

    def _check_suggestions(self) -> None:
        # Suggestions are numbered without gaps, and each names a capture of this drawing.
        numbers = [suggestion.number for suggestion in self.suggestions]
        require(numbers == list(range(1, len(numbers) + 1)), f"Suggestion numbers must run from 1 without gaps, got {numbers}.")
        canvas_size = (self.canvas_width, self.canvas_height)
        for suggestion in self.suggestions:
            number = suggestion.number
            require(suggestion.baseline_capture <= len(self.captures), f"Suggestion {number} names a missing baseline capture.")
            if suggestion.request is not None:
                request_size = (suggestion.request.canvas.width, suggestion.request.canvas.height)
                require(request_size == canvas_size, f"Suggestion {number} canvas is {request_size}, not {canvas_size}.")

            # A shown suggestion's baseline was taken at the press, has a canvas, and came before it was shown.
            if suggestion.status == SuggestionStatus.SHOWN:
                baseline = self.captures[suggestion.baseline_capture - 1]
                valid_baseline = baseline.trigger == CaptureTrigger.SUGGESTION and baseline.canvas is not None
                require(valid_baseline and baseline.at <= suggestion.shown_at,
                        f"Suggestion {number} baseline must be a suggestion capture with a canvas, taken before it was shown.")

    def _check_follow(self) -> None:
        # Scores need the method that made them, and each names the latest suggestion shown before its capture.
        shown = [suggestion for suggestion in self.suggestions if suggestion.status == SuggestionStatus.SHOWN]
        for capture in self.captures:
            if capture.follow is None:
                continue

            number, scored = capture.number, capture.follow.suggestion
            require(self.follow_detector is not None, f"Capture {number} has follow scores but no follow method.")
            earlier = [suggestion.number for suggestion in shown if suggestion.baseline_capture < number]
            require(earlier and earlier[-1] == scored, f"Capture {number} is scored against suggestion {scored}, not the one open when it was taken.")  # noqa: E501


def plain_value(value: ControlValue) -> ControlValue:
    """
    Turn a number subclass, such as a NumPy float, into the plain Python type, so records hold plain JSON numbers.
    """
    if isinstance(value, bool | str):
        return value

    if isinstance(value, int):
        return int(value)

    return float(value) if isinstance(value, float) else value


def steps_to_list(steps: tuple[PreprocessingStep, ...]) -> list[dict[str, object]]:
    """
    Serialize preprocessing steps, with every parameter as a plain JSON value.
    """
    return [{"name": step.name, "params": {name: plain_value(value) for name, value in step.params.items()}} for step in steps]


def steps_from_list(data: list[Mapping[str, object]]) -> tuple[PreprocessingStep, ...]:
    """
    Rebuild preprocessing steps.
    """
    return tuple(PreprocessingStep(name=step["name"], params=step["params"]) for step in data)


def image_or_none(data: Mapping[str, object] | None) -> ImageRef | None:
    """
    Rebuild an image reference, or None when the record holds none.
    """
    return None if data is None else image_from_dict(data)


def alignment_to_dict(alignment: Alignment) -> dict[str, object]:
    """
    Serialize an alignment, with the warp as a list of rows and an unavailable correlation as its reason.
    """
    return {
        "converged": alignment.converged,
        "correlation": encode_value(alignment.correlation),
        "warp": [list(row) for row in alignment.warp] if alignment.warp is not None else None,
        "occluded_share": alignment.occluded_share,
    }


def capture_scores_to_dict(scores: CaptureScores) -> dict[str, object]:
    """
    Serialize a capture's scores against the open suggestion, in the fixed key order of drawing.json.
    """
    return {
        "suggestion": scores.suggestion,
        "seconds_since_shown": scores.seconds_since_shown,
        "alignment": alignment_to_dict(scores.alignment),
        "new_stroke_pixels": scores.new_stroke_pixels,
    }


def capture_scores_from_dict(data: Mapping[str, object]) -> CaptureScores:
    """
    Rebuild a capture's scores through their validating constructors.
    """
    alignment_data = data["alignment"]
    alignment = Alignment(converged=alignment_data["converged"], correlation=decode_value(alignment_data["correlation"]),
                          warp=alignment_data["warp"], occluded_share=alignment_data["occluded_share"])
    return CaptureScores(suggestion=data["suggestion"], seconds_since_shown=data["seconds_since_shown"], alignment=alignment,
                         new_stroke_pixels=data["new_stroke_pixels"])


def follow_method_to_dict(method: FollowMethod) -> dict[str, object]:
    """
    Serialize the drawing's follow method with plain parameter values.
    """
    return {"method": method.name, "version": method.version, "params": {name: plain_value(value) for name, value in method.params.items()}}  # noqa: E501


def follow_method_from_dict(data: Mapping[str, object] | None) -> FollowMethod | None:
    """
    Rebuild the drawing's follow method, or None when the record holds none.
    """
    return None if data is None else FollowMethod(name=data["method"], version=data["version"], params=data["params"])


def capture_to_dict(capture: CaptureRecord) -> dict[str, object]:
    """
    Serialize a capture in the fixed key order of drawing.json.

    Args:
        capture (CaptureRecord): Capture to serialize.

    Returns:
        dict[str, object]: JSON-ready capture.
    """
    return {
        "number": capture.number,
        "trigger": capture.trigger,
        "at": capture.at.isoformat(),
        "seconds_since_start": capture.seconds_since_start,
        "source": capture.source,
        "frame_age_seconds": encode_value(capture.frame_age_seconds),
        "rotation": capture.rotation,
        "raw": asdict(capture.raw),
        "canvas": asdict(capture.canvas) if capture.canvas else None,
        "preprocessing": steps_to_list(capture.steps),
        "timings_seconds": dict(capture.timings_seconds),
        "follow": capture_scores_to_dict(capture.follow) if capture.follow else None,
        "failure": asdict(capture.failure) if capture.failure else None,
    }


def capture_from_dict(data: Mapping[str, object]) -> CaptureRecord:
    """
    Rebuild a capture through its validating constructor.

    Args:
        data (Mapping[str, object]): Serialized capture.

    Returns:
        CaptureRecord: The validated capture.
    """
    failure = StageFailure(**data["failure"]) if data["failure"] is not None else None

    # Captures saved before follow detection have no follow key.
    follow = capture_scores_from_dict(data["follow"]) if data.get("follow") is not None else None
    return CaptureRecord(number=data["number"], trigger=data["trigger"], at=parse_time(data["at"]),
                         seconds_since_start=data["seconds_since_start"], source=data["source"],
                         frame_age_seconds=decode_value(data["frame_age_seconds"]), rotation=data["rotation"],
                         raw=image_from_dict(data["raw"]), canvas=image_or_none(data["canvas"]),
                         steps=steps_from_list(data["preprocessing"]), timings_seconds=data["timings_seconds"], failure=failure,
                         follow=follow)


def stroke_request_to_dict(request: StrokeSuggestionRequest) -> dict[str, object]:
    """
    Serialize a stroke suggestion request, with the prompt exactly as typed.
    """
    return {
        "canvas": asdict(request.canvas),
        "prompt": request.prompt,
        "seed": request.seed,
        "controls": dict(request.controls),
        "max_parts": request.max_parts,
    }


def stroke_request_from_dict(data: Mapping[str, object]) -> StrokeSuggestionRequest:
    """
    Rebuild a stroke suggestion request through its validating constructor.
    """
    return StrokeSuggestionRequest(canvas=image_from_dict(data["canvas"]), prompt=data["prompt"], seed=data["seed"],
                                   controls=data["controls"], max_parts=data["max_parts"])


def stroke_suggestion_to_dict(result: StrokeSuggestion, part_colors: tuple[str, ...]) -> dict[str, object]:
    """
    Serialize a suggester's result, writing each part's overlay color into the part.

    Args:
        result (StrokeSuggestion): Result to serialize.
        part_colors (tuple[str, ...]): Overlay color of each part, by index.

    Returns:
        dict[str, object]: JSON-ready result.
    """
    parts = [
        {
            "index": part.index,
            "mask": asdict(part.mask),
            "box": list(part.box),
            "pixel_count": part.pixel_count,
            "skeleton_pixels": part.skeleton_pixels,
            "color": color,
            "label": part.label,
        }
        for part, color in zip(result.parts, part_colors, strict=True)
    ]
    line_drawing = result.line_drawing
    return {
        "suggester": backend_to_dict(result.suggester),
        "effective": effective_to_dict(result.effective),
        "seed": encode_value(result.seed),
        "line_drawing": encode_value(line_drawing) if isinstance(line_drawing, Unavailable) else asdict(line_drawing),
        "parts": parts,
        "timings_seconds": dict(result.timings_seconds),
    }


def stroke_suggestion_from_dict(data: Mapping[str, object]) -> tuple[StrokeSuggestion, tuple[str, ...]]:
    """
    Rebuild a suggester's result and the overlay color of each part.

    Args:
        data (Mapping[str, object]): Serialized result.

    Returns:
        tuple[StrokeSuggestion, tuple[str, ...]]: The validated result and the part colors.
    """
    parts = tuple(SuggestedPart(index=item["index"], mask=image_from_dict(item["mask"]), box=item["box"],
                                pixel_count=item["pixel_count"], skeleton_pixels=item["skeleton_pixels"], label=item["label"])
                  for item in data["parts"])

    # Check for an unavailable line drawing first, since decode_value treats any dict as one.
    line_data = data["line_drawing"]
    line_drawing = decode_value(line_data) if "unavailable" in line_data else image_from_dict(line_data)
    result = StrokeSuggestion(suggester=backend_from_dict(data["suggester"]), effective=effective_from_dict(data["effective"]),
                              seed=decode_value(data["seed"]), line_drawing=line_drawing, parts=parts,
                              timings_seconds=data["timings_seconds"])
    return result, tuple(item["color"] for item in data["parts"])


def suggestion_to_dict(suggestion: SuggestionRecord) -> dict[str, object]:
    """
    Serialize a suggestion in the fixed key order of drawing.json.

    Args:
        suggestion (SuggestionRecord): Suggestion to serialize.

    Returns:
        dict[str, object]: JSON-ready suggestion.
    """
    request, result, overlay, failure = suggestion.request, suggestion.result, suggestion.overlay, suggestion.failure
    return {
        "number": suggestion.number,
        "id": suggestion.id,
        "status": suggestion.status,
        "requested_at": suggestion.requested_at.isoformat(),
        "shown_at": suggestion.shown_at.isoformat() if suggestion.shown_at else None,
        "baseline_capture": suggestion.baseline_capture,
        "request": stroke_request_to_dict(request) if request else None,
        "result": stroke_suggestion_to_dict(result, suggestion.part_colors) if result else None,
        "overlay": asdict(overlay) if overlay else None,
        "timings_seconds": dict(suggestion.timings_seconds),
        "failure": asdict(failure) if failure else None,
    }


def suggestion_from_dict(data: Mapping[str, object]) -> SuggestionRecord:
    """
    Rebuild a suggestion through its validating constructor.

    Args:
        data (Mapping[str, object]): Serialized suggestion.

    Returns:
        SuggestionRecord: The validated suggestion.
    """
    request = stroke_request_from_dict(data["request"]) if data["request"] is not None else None
    result, part_colors = stroke_suggestion_from_dict(data["result"]) if data["result"] is not None else (None, ())
    failure = StageFailure(**data["failure"]) if data["failure"] is not None else None
    shown_at = parse_time(data["shown_at"]) if data["shown_at"] is not None else None
    return SuggestionRecord(number=data["number"], id=data["id"], status=data["status"],
                            requested_at=parse_time(data["requested_at"]), shown_at=shown_at,
                            baseline_capture=data["baseline_capture"], request=request, result=result,
                            part_colors=part_colors, overlay=image_or_none(data["overlay"]),
                            timings_seconds=data["timings_seconds"], failure=failure)


def drawing_to_dict(record: DrawingRecord) -> dict[str, object]:
    """
    Serialize a drawing record in the fixed key order of drawing.json.

    Args:
        record (DrawingRecord): Drawing to serialize.

    Returns:
        dict[str, object]: JSON-ready drawing record.
    """
    return {
        "schema_version": record.schema_version,
        "drawing": record.number,
        "drawing_id": record.drawing_id,
        "started_at": record.started_at.isoformat(),
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
        "end": record.end,
        "canvas": {"width": record.canvas_width, "height": record.canvas_height},
        "save_interval_seconds": record.save_interval_seconds,
        "suggestions_enabled": record.suggestions_enabled,
        "suggester": record.suggester,
        "follow_detector": follow_method_to_dict(record.follow_detector) if record.follow_detector else None,
        "captures": [capture_to_dict(capture) for capture in record.captures],
        "suggestions": [suggestion_to_dict(suggestion) for suggestion in record.suggestions],
        "final_capture": record.final_capture,
    }


def drawing_from_dict(data: Mapping[str, object]) -> DrawingRecord:
    """
    Rebuild a drawing record through the validating constructors.

    Args:
        data (Mapping[str, object]): Contents of drawing.json.

    Returns:
        DrawingRecord: The validated drawing record.
    """
    check_schema_version(data)
    ended_at = parse_time(data["ended_at"]) if data["ended_at"] is not None else None
    canvas = data["canvas"]
    return DrawingRecord(schema_version=data["schema_version"], number=data["drawing"], drawing_id=data["drawing_id"],
                         started_at=parse_time(data["started_at"]), ended_at=ended_at, end=data["end"],
                         canvas_width=canvas["width"], canvas_height=canvas["height"],
                         save_interval_seconds=data["save_interval_seconds"], suggestions_enabled=data["suggestions_enabled"],
                         suggester=data["suggester"], captures=tuple(capture_from_dict(item) for item in data["captures"]),
                         suggestions=tuple(suggestion_from_dict(item) for item in data["suggestions"]),
                         final_capture=data["final_capture"],
                         follow_detector=follow_method_from_dict(data.get("follow_detector")))


def list_drawing_images(record: DrawingRecord) -> list[ImageRef]:
    """
    List every image a drawing recorded, relative to its folder, without duplicates.
    """
    images = [image for capture in record.captures for image in (capture.raw, capture.canvas) if image is not None]
    for suggestion in record.suggestions:
        images += [suggestion.request.canvas] if suggestion.request else []
        images += list(suggestion.result.images) if suggestion.result else []
        images += [suggestion.overlay] if suggestion.overlay else []

    return list({image.path: image for image in images}.values())
