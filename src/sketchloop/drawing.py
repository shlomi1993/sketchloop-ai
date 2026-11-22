"""
A guided drawing that records its captures and next-stroke suggestions from its start to its end, and draws their overlay.
"""

import cv2
import hashlib
import numpy as np
import threading
import time
import uuid

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import PurePosixPath
from typing import Final

from sketchloop.capture import CapturedSketch, InvalidSketchError, encode_frame_jpeg
from sketchloop.domain import SCHEMA_VERSION, ColorMode, ControlValue, ImageRef, Rotation, SketchLoopError, SketchSource
from sketchloop.experiments import (CaptureRecord, CaptureTrigger, DrawingEnd, DrawingRecord, ExperimentStore, StoreError,
                                    SuggestionRecord, SuggestionStage, SuggestionStatus, describe_stage_failure,
                                    resolve_frame_age)
from sketchloop.follow import AlignedStrokeFollowDetector, CaptureScores, FollowDetector, describe_follow_method
from sketchloop.preprocessing import PaperCanvas, PreprocessedSketch, decode_grayscale, extract_canvas_strokes, make_paper_canvas
from sketchloop.stroke_suggestion import (PART_COLORS, StrokeSuggester, StrokeSuggestion, StrokeSuggestionError,
                                          StrokeSuggestionOutput, StrokeSuggestionRequest)


NO_STROKES_MESSAGE: Final = "No strokes found on the paper. Draw something, then ask again."
CAPTURE_TIMING: Final = "capture"  # Timing key of the time taken to receive and encode the frame
OVERLAY_FILE: Final = "overlay.png"
OVERLAY_GROW_PIXELS: Final = 2  # Each part's mask is grown by this much, so thin suggested lines stay visible
OVERLAY_OPACITY: Final = 0.8  # Share of the overlay color in the preview where a part is drawn
STROKE_LEVEL: Final = 128  # Pixels of the suggester's input darker than this are strokes
INTERVAL_JPEG_QUALITY: Final = 90  # JPEG quality of interval captures' raw frames, while suggestion and finish frames stay PNG
JPEG_MEDIA_TYPE: Final = "image/jpeg"


def name_capture_files(number: int, raw_path: str) -> tuple[str, str, str]:
    """
    Name a capture's raw file, keeping its extension, its canvas file, and its filled mask, under captures/ in the drawing folder.
    """
    suffix = PurePosixPath(raw_path).suffix.lower() or ".png"
    stem = f"captures/capture-{number:04d}"
    return f"{stem}-raw{suffix}", f"{stem}.png", f"{stem}-filled.png"


def compress_interval_frame(captured: CapturedSketch) -> CapturedSketch:
    """
    Re-encode a raw frame as JPEG for an interval capture, keeping a JPEG as received and adding the time to the capture.

    Args:
        captured (CapturedSketch): Raw frame as received, PNG or JPEG.

    Returns:
        CapturedSketch: The frame as JPEG at INTERVAL_JPEG_QUALITY, or unchanged when it already was JPEG or cannot be decoded.
    """
    if captured.image.media_type == JPEG_MEDIA_TYPE:
        return captured

    # Keep every channel, so the JPEG has the frame's colors, and leave an undecodable frame for the canvas step to record.
    started = time.perf_counter()
    pixels = cv2.imdecode(np.frombuffer(captured.payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint8:
        return captured

    image, payload = encode_frame_jpeg(pixels, INTERVAL_JPEG_QUALITY)
    return CapturedSketch(image=image, payload=payload, capture_seconds=captured.capture_seconds + time.perf_counter() - started)


def place_suggestion(suggestion: StrokeSuggestion, payloads: Mapping[str, bytes],
                     folder: str) -> tuple[StrokeSuggestion, dict[str, bytes]]:
    """
    Move a suggestion's bare file names, and the keys of their bytes, under the suggestion's folder.

    Args:
        suggestion (StrokeSuggestion): Suggestion with bare file names, as the suggester returns it.
        payloads (Mapping[str, bytes]): Bytes by bare file name.
        folder (str): Folder relative to the drawing folder, ending in "/".

    Returns:
        tuple[StrokeSuggestion, dict[str, bytes]]: The placed suggestion and its bytes by placed path.
    """
    line_drawing = suggestion.line_drawing
    if isinstance(line_drawing, ImageRef):
        line_drawing = replace(line_drawing, path=folder + line_drawing.path)
    parts = tuple(replace(part, mask=replace(part.mask, path=folder + part.mask.path)) for part in suggestion.parts)
    placed = replace(suggestion, line_drawing=line_drawing, parts=parts)
    return placed, {folder + path: payload for path, payload in payloads.items()}


def check_suggestion_output(output: StrokeSuggestionOutput, request: StrokeSuggestionRequest) -> None:
    """
    Refuse more parts than requested, or an image that is not the canvas size, which the output alone cannot check.

    Args:
        output (StrokeSuggestionOutput): The suggester's output.
        request (StrokeSuggestionRequest): Request it answers.
    """
    parts, limit = len(output.suggestion.parts), request.max_parts
    if parts > limit:
        raise StrokeSuggestionError(f"The stroke suggester returned {parts} parts for at most {limit}. Fix the suggester.")

    # Every mask and the line drawing must overlay the canvas exactly.
    width, height = request.canvas.width, request.canvas.height
    for image in output.suggestion.images:
        if (image.width, image.height) != (width, height):
            raise StrokeSuggestionError(f"The stroke suggester returned {image.path} at {image.width}x{image.height}, not the {width}x{height} canvas. Fix the suggester.")  # noqa: E501


def parse_hex_color(color: str) -> tuple[int, int, int]:
    """
    Turn a "#RRGGBB" color into its red, green, and blue levels.
    """
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def draw_overlay(mask_payloads: Sequence[bytes], width: int, height: int) -> np.ndarray:
    """
    Paint each part's mask, grown slightly, in its part color on a transparent RGBA image, later parts on top.

    Args:
        mask_payloads (Sequence[bytes]): PNG bytes of each part's mask, by part index.
        width (int): Canvas width in pixels.
        height (int): Canvas height in pixels.

    Returns:
        np.ndarray: RGBA overlay in RGB channel order, opaque on the parts and transparent elsewhere.
    """
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OVERLAY_GROW_PIXELS + 1,) * 2)
    for index, payload in enumerate(mask_payloads):
        grown = cv2.dilate(decode_grayscale(payload), kernel)
        overlay[grown > 0] = (*parse_hex_color(PART_COLORS[index]), 255)

    return overlay


def encode_overlay(overlay: np.ndarray, path: str) -> tuple[ImageRef, bytes]:
    """
    Encode an RGBA overlay in RGB channel order as PNG, and describe it under the given path.

    Args:
        overlay (np.ndarray): RGBA overlay.
        path (str): Path relative to the drawing folder.

    Returns:
        tuple[ImageRef, bytes]: The image reference and the PNG bytes.
    """
    # OpenCV writes channels in BGRA order, so swap red and blue for a PNG that reads as RGBA.
    encoded, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
    if not encoded:
        raise SketchLoopError("The suggestion overlay cannot be encoded as PNG. Ask again.")

    payload = buffer.tobytes()
    height, width = overlay.shape[:2]
    image = ImageRef(path=path, width=width, height=height, mode=ColorMode.RGBA, media_type="image/png",
                     sha256=hashlib.sha256(payload).hexdigest())
    return image, payload


def blend_overlay(canvas: np.ndarray, overlay: np.ndarray | None) -> np.ndarray:
    """
    Return the grayscale canvas as RGB, with the overlay's colors blended at OVERLAY_OPACITY where it is opaque.
    """
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if overlay is not None:
        alpha = overlay[..., 3:].astype(np.float32) / 255 * OVERLAY_OPACITY
        rgb = rgb * (1 - alpha) + overlay[..., :3].astype(np.float32) * alpha

    return np.clip(rgb.round(), 0, 255).astype(np.uint8)


def has_strokes(strokes: PreprocessedSketch) -> bool:
    """
    Tell whether the suggester's input has any dark pixel.
    """
    return bool(np.any(decode_grayscale(strokes.payload) < STROKE_LEVEL))


class GuidedDrawing:
    """
    One drawing from its start to its end: captures, next-stroke suggestions with their overlays, and drawing.json.
    """

    def __init__(self, store: ExperimentStore, session_id: str, number: int, suggester: StrokeSuggester | None, *,
                 canvas_width: int, canvas_height: int, save_interval_seconds: float,
                 follow_detector: FollowDetector | None = None) -> None:
        self.store = store
        self.session_id = session_id
        self.suggester = suggester
        self.follow_detector = follow_detector or AlignedStrokeFollowDetector()
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._pending_files: dict[str, bytes] = {}
        self._overlay: np.ndarray | None = None
        self._open_suggestion: SuggestionRecord | None = None
        self._baseline: PaperCanvas | None = None
        self._part_masks: tuple[bytes, ...] = ()
        self._shown_since_start = 0.0
        self._loaded_suggester: StrokeSuggester | None = None
        self._record = DrawingRecord(schema_version=SCHEMA_VERSION, number=number, drawing_id=uuid.uuid4().hex,
                                     started_at=datetime.now(timezone.utc), ended_at=None, end=None, canvas_width=canvas_width,
                                     canvas_height=canvas_height, save_interval_seconds=save_interval_seconds,
                                     suggestions_enabled=suggester is not None,
                                     suggester=type(suggester).__name__ if suggester is not None else None, captures=(),
                                     suggestions=(), final_capture=None,
                                     follow_detector=describe_follow_method(self.follow_detector) if suggester else None,
                                     interval_jpeg_quality=INTERVAL_JPEG_QUALITY)

        # Save the empty record now, so the drawing folder exists and loads as not finished if the server stops.
        self._save({})

    @property
    def record(self) -> DrawingRecord:
        """
        Return the drawing's current record, which may hold timings not yet written to drawing.json.
        """
        return self._record

    @property
    def is_open(self) -> bool:
        """
        Tell whether the drawing has not ended yet.
        """
        return self._record.end is None

    @property
    def open_suggestion(self) -> SuggestionRecord | None:
        """
        Return the latest shown suggestion while the drawing is open, or None.
        """
        return self._open_suggestion

    @property
    def seconds_since_last_capture(self) -> float | None:
        """
        Return the seconds since the latest capture on the drawing's monotonic clock, or None before the first.
        """
        captures = self._record.captures
        if not captures:
            return None

        return time.monotonic() - self._started - captures[-1].seconds_since_start

    def _save(self, files: Mapping[str, bytes]) -> None:
        # Write the new files with any left from a failed save, then drawing.json, and forget them only once both are saved.
        self._pending_files |= files
        self.store.save_drawing(self.session_id, self._record, self._pending_files)
        self._pending_files = {}

    def _require_open(self) -> None:
        # A finished drawing is a closed record, so refuse anything that would change it.
        if not self.is_open:
            raise SketchLoopError(f"Drawing {self._record.number} is finished. Start a new drawing.")

    def _score_capture(self, canvas: PaperCanvas, seconds_since_start: float, timings: dict[str, float]) -> CaptureScores | None:
        # Score the canvas against the open suggestion's baseline and parts, with both canvases' filled areas hidden.
        # The detector records a failed alignment rather than raising.
        if self._open_suggestion is None or self._baseline is None:
            return None

        started = time.perf_counter()
        alignment, new_stroke_pixels, parts = self.follow_detector.score(
            self._baseline.payload, canvas.payload, self._part_masks, baseline_filled=self._baseline.filled_payload,
            capture_filled=canvas.filled_payload)
        timings[SuggestionStage.SCORING] = time.perf_counter() - started
        seconds_since_shown = seconds_since_start - self._shown_since_start
        return CaptureScores(suggestion=self._open_suggestion.number, seconds_since_shown=seconds_since_shown,
                             alignment=alignment, new_stroke_pixels=new_stroke_pixels, parts=parts)

    def _take_capture(self, captured: CapturedSketch, source: SketchSource, rotation: int, frame_age_seconds: float | None,
                      trigger: CaptureTrigger) -> tuple[CaptureRecord, PaperCanvas | None, Exception | None]:
        # Fix the time and number, and refuse values the record could not hold before doing any work.
        at, seconds_since_start = datetime.now(timezone.utc), time.monotonic() - self._started
        frame_age = resolve_frame_age(source, frame_age_seconds)
        if rotation not in list(Rotation):
            raise SketchLoopError(f"Rotation must be one of 0, 90, 180, or 270, got {rotation}.")

        # Make the paper-frame canvas, keeping a failure in the record rather than losing the frame.
        number = len(self._record.captures) + 1
        raw_path, canvas_path, filled_path = name_capture_files(number, captured.image.path)
        started = time.perf_counter()
        canvas, error = None, None
        try:
            canvas = make_paper_canvas(captured.payload, rotation, self._record.canvas_width, self._record.canvas_height)
        except Exception as caught:
            error = caught
        timings = {CAPTURE_TIMING: captured.capture_seconds, SuggestionStage.CANVAS: time.perf_counter() - started}
        follow = self._score_capture(canvas, seconds_since_start, timings) if canvas else None

        # Add the capture to the record, and its files to the next save.
        canvas_files = {canvas_path: canvas.payload, filled_path: canvas.filled_payload} if canvas else {}
        files = {raw_path: captured.payload} | canvas_files
        capture = CaptureRecord(number=number, trigger=trigger, at=at, seconds_since_start=seconds_since_start, source=source,
                                frame_age_seconds=frame_age, rotation=rotation, raw=replace(captured.image, path=raw_path),
                                canvas=replace(canvas.image, path=canvas_path) if canvas else None,
                                steps=canvas.steps if canvas else (), timings_seconds=timings,
                                failure=describe_stage_failure(SuggestionStage.CANVAS, error) if error else None,
                                follow=follow, filled=replace(canvas.filled, path=filled_path) if canvas else None)
        self._record = replace(self._record, captures=(*self._record.captures, capture))
        self._pending_files |= files
        return capture, canvas, error

    def _replace_suggestion(self, suggestion: SuggestionRecord) -> None:
        # Put the suggestion in the record, replacing an earlier version with the same number.
        kept = tuple(entry for entry in self._record.suggestions if entry.number != suggestion.number)
        self._record = replace(self._record, suggestions=(*kept, suggestion))

    def _record_failed_suggestion(self, failed: SuggestionRecord, error: Exception, unsaved_files: Mapping[str, bytes]) -> None:
        # Drop files only the failed suggestion would have named, keeping the capture's, then save the failure.
        for path in unsaved_files:
            self._pending_files.pop(path, None)

        self._replace_suggestion(failed)
        try:
            self._save({})
        except StoreError as store_error:
            error.add_note(f"The failed suggestion was not saved either: {store_error}")

    def _load_suggester(self, timings: dict[str, float]) -> None:
        # Load and warm up each suggester once per drawing, timing it only when the suggester has a model to load.
        if self._loaded_suggester is self.suggester:
            return

        started = time.perf_counter()
        self.suggester.load(self._record.canvas_width, self._record.canvas_height)
        if type(self.suggester).load is not StrokeSuggester.load:
            timings[SuggestionStage.MODEL_LOAD] = time.perf_counter() - started
        self._loaded_suggester = self.suggester

    def _run_suggester(self, request: StrokeSuggestionRequest, strokes_payload: bytes,
                       timings: dict[str, float]) -> StrokeSuggestionOutput:
        # Run the suggester, check what its output cannot check alone, and record its load and its own stage times.
        output = self.suggester.suggest(request, strokes_payload)
        check_suggestion_output(output, request)
        if output.load_seconds is not None:
            timings[SuggestionStage.MODEL_LOAD] = timings.get(SuggestionStage.MODEL_LOAD, 0.0) + output.load_seconds

        reported = output.suggestion.timings_seconds
        own_stages = (SuggestionStage.GENERATION, SuggestionStage.EXTRACTION)
        timings |= {stage: reported[stage] for stage in own_stages if stage in reported}
        return output

    def _show_suggestion(self, draft: Callable[..., SuggestionRecord], failed: Callable[..., SuggestionRecord],
                         request: StrokeSuggestionRequest, strokes: PreprocessedSketch, folder: str,
                         timings: dict[str, float]) -> tuple[SuggestionRecord, np.ndarray, tuple[bytes, ...]]:
        # Run each stage, tracking the active one so a failure record names it.
        stage = SuggestionStage.MODEL_LOAD
        files: dict[str, bytes] = {}
        try:
            self._load_suggester(timings)
            stage = SuggestionStage.GENERATION
            output = self._run_suggester(request, strokes.payload, timings)

            # Draw the overlay the person sees, and place the suggester's files under the suggestion's folder.
            stage = SuggestionStage.OVERLAY
            started = time.perf_counter()
            result, files = place_suggestion(output.suggestion, output.payloads, folder)
            mask_payloads = [files[part.mask.path] for part in result.parts]
            overlay = draw_overlay(mask_payloads, request.canvas.width, request.canvas.height)
            overlay_ref, files[folder + OVERLAY_FILE] = encode_overlay(overlay, folder + OVERLAY_FILE)
            files[request.canvas.path] = strokes.payload
            timings[SuggestionStage.OVERLAY] = time.perf_counter() - started
            shown = draft(status=SuggestionStatus.SHOWN, shown_at=datetime.now(timezone.utc), request=request, result=result,
                          part_colors=PART_COLORS[:len(result.parts)], overlay=overlay_ref, timings_seconds=timings, failure=None)

            # Save the files and drawing.json, with the baseline capture's files when they are not saved yet.
            stage = SuggestionStage.SAVING
            self._replace_suggestion(shown)
            started = time.perf_counter()
            self._save(files)
            timings[SuggestionStage.SAVING] = time.perf_counter() - started
        except Exception as error:
            # Keep a failed record in the drawing even when it cannot be saved, so the attempt stays visible.
            failure = describe_stage_failure(stage, error)
            self._record_failed_suggestion(failed(request=request, timings_seconds=timings, failure=failure), error, files)
            raise

        # The saving time is known only after saving, so it reaches drawing.json with the drawing's next save.
        shown = replace(shown, timings_seconds=timings)
        self._replace_suggestion(shown)
        return shown, overlay, tuple(mask_payloads)

    def suggest(self, captured: CapturedSketch, *, source: SketchSource, rotation: int, prompt: str, seed: int | None = None,
                controls: Mapping[str, ControlValue] | None = None, frame_age_seconds: float | None = None) -> SuggestionRecord:
        """
        Capture the frame, suggest next strokes for it, draw their overlay, and save the suggestion, or record why it failed.

        Args:
            captured (CapturedSketch): Frame at the press.
            source (SketchSource): Where the frame came from.
            rotation (int): Degrees to turn the frame clockwise before making the canvas.
            prompt (str): Prompt box text exactly as the participant typed it, possibly empty.
            seed (int | None, optional): Seed, or None for a random one that the result records. Defaults to None.
            controls (Mapping[str, ControlValue] | None, optional): Requested suggester controls, or None for the defaults.
                Defaults to None.
            frame_age_seconds (float | None, optional): Age of a streamed webcam frame, or None. Defaults to None.

        Returns:
            SuggestionRecord: The shown suggestion, possibly with no parts.
        """
        with self._lock:
            self._require_open()
            if self.suggester is None:
                raise SketchLoopError(f"Suggestions are off for drawing {self._record.number}.")

            # Capture the frame at the press, which becomes the suggestion's baseline.
            requested_at = datetime.now(timezone.utc)
            capture, canvas, canvas_error = self._take_capture(captured, source, rotation, frame_age_seconds,
                                                               CaptureTrigger.SUGGESTION)
            number = len(self._record.suggestions) + 1
            folder = f"suggestions/suggestion-{number}/"
            draft = partial(SuggestionRecord, number=number, id=uuid.uuid4().hex, requested_at=requested_at,
                            baseline_capture=capture.number)
            failed = partial(draft, status=SuggestionStatus.FAILED, shown_at=None, result=None, part_colors=(), overlay=None)
            timings: dict[str, float] = {SuggestionStage.CANVAS: capture.timings_seconds[SuggestionStage.CANVAS]}

            # Find the strokes on the canvas, recording a failed suggestion when there is no canvas.
            try:
                if canvas_error is not None:
                    raise canvas_error

                started = time.perf_counter()
                strokes = extract_canvas_strokes(canvas.payload)
                timings[SuggestionStage.CANVAS] += time.perf_counter() - started
            except Exception as error:
                failure = describe_stage_failure(SuggestionStage.CANVAS, error)
                self._record_failed_suggestion(failed(request=None, timings_seconds=timings, failure=failure), error, {})
                raise

            # Refuse a blank page before the model runs, keeping the capture but recording no suggestion.
            if not has_strokes(strokes):
                refusal = InvalidSketchError(NO_STROKES_MESSAGE)
                try:
                    self._save({})
                except StoreError as store_error:
                    refusal.add_note(f"The capture was not saved either: {store_error}")
                raise refusal

            request = StrokeSuggestionRequest(canvas=replace(strokes.image, path=folder + strokes.image.path), prompt=prompt,
                                              seed=seed, controls=controls or {})
            shown, overlay, self._part_masks = self._show_suggestion(draft, failed, request, strokes, folder, timings)
            self._overlay, self._open_suggestion, self._baseline = overlay, shown, canvas
            self._shown_since_start = time.monotonic() - self._started
            return shown

    def record_capture(self, captured: CapturedSketch, *, source: SketchSource, rotation: int,
                       frame_age_seconds: float | None = None) -> CaptureRecord | None:
        """
        Save an interval capture as a JPEG raw frame and its canvas, aligned to the open suggestion's baseline if any.

        Args:
            captured (CapturedSketch): Frame to keep, re-encoded as JPEG unless it already is.
            source (SketchSource): Where the frame came from.
            rotation (int): Degrees to turn the frame clockwise before making the canvas.
            frame_age_seconds (float | None, optional): Age of a streamed webcam frame, or None. Defaults to None.

        Returns:
            CaptureRecord | None: The saved capture, with any canvas failure kept in its record rather than raised, or None
                when a suggestion or another change is running.
        """
        # Skip rather than wait while a suggestion runs, since a frame saved after the wait would be stale.
        if not self._lock.acquire(blocking=False):
            return None

        try:
            self._require_open()

            # Compress the frame, since interval captures are most of a drawing's disk use, then capture and save it.
            capture = self._take_capture(compress_interval_frame(captured), source, rotation, frame_age_seconds,
                                         CaptureTrigger.INTERVAL)[0]
            started = time.perf_counter()
            self._save({})

            # The saving time is known only after saving, so it reaches drawing.json with the drawing's next save.
            timings = dict(capture.timings_seconds) | {SuggestionStage.SAVING: time.perf_counter() - started}
            capture = replace(capture, timings_seconds=timings)
            self._record = replace(self._record, captures=(*self._record.captures[:-1], capture))
            return capture
        finally:
            self._lock.release()

    def finish(self, captured: CapturedSketch | None, *, source: SketchSource | None = None, rotation: int = Rotation.DEG_000,
               frame_age_seconds: float | None = None, end: DrawingEnd = DrawingEnd.FINISHED) -> DrawingRecord:
        """
        End the drawing, after a final capture when a frame is given, and save it, so it refuses any further change.

        Args:
            captured (CapturedSketch | None): Final frame, or None to end without a final capture.
            source (SketchSource | None, optional): Where the final frame came from, needed with a frame. Defaults to None.
            rotation (int, optional): Degrees to turn the final frame clockwise. Defaults to 0.
            frame_age_seconds (float | None, optional): Age of a streamed webcam frame, or None. Defaults to None.
            end (DrawingEnd, optional): Why the drawing ends. Defaults to finished.

        Returns:
            DrawingRecord: The finished drawing's record.
        """
        with self._lock:
            self._require_open()

            # Take the final capture, whose canvas failure is kept in its record rather than stopping the finish.
            final_capture = None
            if captured is not None:
                if source is None:
                    raise SketchLoopError("A final frame needs its source, such as webcam or upload.")

                final_capture = self._take_capture(captured, source, rotation, frame_age_seconds, CaptureTrigger.FINISH)[0].number

            # Save the end, and undo it on a failed save, so the person can press finish again.
            ended_at = datetime.now(timezone.utc)
            self._record = replace(self._record, ended_at=ended_at, end=DrawingEnd(end), final_capture=final_capture)
            try:
                self._save({})
            except StoreError:
                self._record = replace(self._record, ended_at=None, end=None, final_capture=None)
                raise

            self._overlay, self._open_suggestion, self._baseline, self._part_masks = None, None, None, ()
            return self._record

    def compose_preview(self, canvas: PaperCanvas) -> bytes:
        """
        Return an RGB PNG of the canvas with the open suggestion's overlay blended over it, or the canvas alone without one.

        Args:
            canvas (PaperCanvas): Current frame's paper-frame canvas at the drawing's canvas size.

        Returns:
            bytes: PNG bytes of the preview.
        """
        # Read the overlay without the lock, so a preview never waits for a suggestion that is still running.
        gray = decode_grayscale(canvas.payload)
        size = (self._record.canvas_width, self._record.canvas_height)
        if (gray.shape[1], gray.shape[0]) != size:
            raise ValueError(f"Preview canvas is {gray.shape[1]}x{gray.shape[0]}, not the drawing's {size[0]}x{size[1]}.")

        preview = blend_overlay(gray, self._overlay)
        encoded, buffer = cv2.imencode(".png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        if not encoded:
            raise SketchLoopError("The guided preview cannot be encoded as PNG.")

        return buffer.tobytes()
