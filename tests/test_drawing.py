"""
Tests of guided drawings: captures, suggestions with their overlay, failures, finishing, and the composed preview.
"""

import cv2
import json
import numpy as np
import pytest

from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from sketchloop.capture import CapturedSketch, InvalidSketchError
from sketchloop.domain import ColorMode, SketchLoopError, SketchSource
from sketchloop.drawing import NO_STROKES_MESSAGE, GuidedDrawing, check_suggestion_output, parse_hex_color
from sketchloop.experiments import (CaptureTrigger, DrawingEnd, DrawingRecord, MemoryStore, StoreError, SuggestionStage,
                                    SuggestionStatus, drawing_from_dict, drawing_to_dict)
from sketchloop.preprocessing import make_paper_canvas
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester, encode_png
from sketchloop.stroke_suggestion import (LINE_DRAWING_FILE, PART_COLORS, StrokeSuggester, StrokeSuggestionError,
                                          StrokeSuggestionOutput, StrokeSuggestionRequest)


SESSION_ID: Final = "session"
CANVAS_SIZE: Final = (512, 360)
OUT_OF_MEMORY: Final = "Next-stroke suggestion failed: out of memory. Free memory and retry."


class FailingSuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester whose suggestion fails as if the device ran out of memory.
    """

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Raise the error a real suggester raises when generation fails.
        """
        raise StrokeSuggestionError(OUT_OF_MEMORY)


class MissingWeightsSuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester whose model cannot load, as when the weights were never downloaded.
    """

    def load(self, width: int, height: int) -> None:
        """
        Raise the error a real suggester raises for a missing model folder.
        """
        raise StrokeSuggestionError("Model folder models/x is missing. Run python scripts/download_models.py and retry.")


class EmptySuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester that finds nothing new to suggest, as for a nearly finished drawing.
    """

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Return the fake's suggestion with its parts removed, keeping only the line drawing.
        """
        output = super().suggest(request, canvas_payload)
        suggestion = replace(output.suggestion, parts=())
        return StrokeSuggestionOutput(suggestion=suggestion, payloads={LINE_DRAWING_FILE: output.payloads[LINE_DRAWING_FILE]})


class SmallMaskSuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester that draws its parts on a 64x64 canvas instead of the requested one.
    """

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Suggest for a blank 64x64 canvas, ignoring the requested canvas.
        """
        small_ref, small_payload = encode_png(np.full((64, 64), 255, dtype=np.uint8), "strokes.png")
        return super().suggest(replace(request, canvas=small_ref), small_payload)


class FlakyDrawingStore(MemoryStore):
    """
    Test-only memory store whose next drawing saves fail, before keeping anything, as a full disk would.
    """

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 0

    def save_drawing(self, session_id: str, record: DrawingRecord, files: Mapping[str, bytes]) -> None:
        """
        Fail while failures are left, then keep the files and record.
        """
        if self.failures_left:
            self.failures_left -= 1
            raise StoreError("Cannot save session/drawing-1/drawing.json: No space left on device. Free disk space, then retry.")

        super().save_drawing(session_id, record, files)


def start_drawing(store: MemoryStore, suggester: StrokeSuggester | None) -> GuidedDrawing:
    """
    Start drawing 1 of a test session at the default canvas size.
    """
    return GuidedDrawing(store, SESSION_ID, 1, suggester, canvas_width=CANVAS_SIZE[0], canvas_height=CANVAS_SIZE[1],
                         save_interval_seconds=2.0)


def suggest(drawing: GuidedDrawing, captured: CapturedSketch, prompt: str = "a car") -> None:
    """
    Press suggest with a webcam frame of age 0.2 s at no rotation.
    """
    drawing.suggest(captured, source=SketchSource.WEBCAM, rotation=0, prompt=prompt, frame_age_seconds=0.2)


def test_drawing_records_suggestions_and_final_capture(sheet_capture: CapturedSketch) -> None:
    """
    Two suggestions and a finish give consecutive captures, suggestion baselines, an overlay, and a record that round-trips.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    first = drawing.suggest(sheet_capture, source=SketchSource.WEBCAM, rotation=0, prompt="", frame_age_seconds=0.2)
    suggest(drawing, sheet_capture, prompt="  a red sports car ")
    record = drawing.finish(sheet_capture, source=SketchSource.UPLOAD)
    saved = store.drawings[(SESSION_ID, 1)]
    text = json.dumps(drawing_to_dict(record), allow_nan=False)

    # Captures are numbered in order, each suggestion's baseline is its press, and the final capture was taken at finish.
    triggers = [capture.trigger for capture in record.captures]
    assert triggers == [CaptureTrigger.SUGGESTION, CaptureTrigger.SUGGESTION, CaptureTrigger.FINISH], f"Bad triggers {triggers}"
    baselines = [suggestion.baseline_capture for suggestion in record.suggestions]
    assert baselines == [1, 2] and record.final_capture == 3, f"Expected baselines 1, 2 and final 3, got {baselines}"
    assert record.end == DrawingEnd.FINISHED and record.ended_at is not None, f"Expected a finished drawing, got {record.end}"

    # The prompt is kept exactly as typed, and the overlay has the canvas size in RGBA with one color per part.
    prompts = [suggestion.request.prompt for suggestion in record.suggestions]
    assert prompts == ["", "  a red sports car "], f"Prompts must be kept exactly as typed, got {prompts}"
    overlay = first.overlay
    assert (overlay.width, overlay.height, overlay.mode) == (*CANVAS_SIZE, ColorMode.RGBA), f"Bad overlay {overlay}"
    assert first.part_colors == PART_COLORS[:2] and first.status == SuggestionStatus.SHOWN, f"Bad suggestion {first}"

    # Every file the record names was saved under the drawing, in the place the design gives.
    paths = {path for (session, number, path) in store.drawing_files if (session, number) == (SESSION_ID, 1)}
    expected = {"captures/capture-0001-raw.png", "captures/capture-0001.png", "suggestions/suggestion-1/strokes.png",
                "suggestions/suggestion-1/line-drawing.png", "suggestions/suggestion-1/part-1.png",
                "suggestions/suggestion-1/overlay.png", "captures/capture-0003-raw.png"}
    assert expected <= paths, f"Missing drawing files {sorted(expected - paths)}"

    # The saved record is the final one, and it round-trips through JSON unchanged with plain numbers only.
    assert saved == record, "The store must hold the finished record"
    assert drawing_from_dict(json.loads(text)) == record, "drawing.json must load back unchanged"
    paper_share = drawing_to_dict(record)["captures"][0]["preprocessing"][2]["params"]["paper_share"]
    assert type(paper_share) is float, f"Canvas steps must hold plain floats, got {type(paper_share)}"

    # A finished drawing refuses further presses.
    with pytest.raises(SketchLoopError, match="is finished"):
        suggest(drawing, sheet_capture)


def test_drawing_refuses_blank_canvas_but_keeps_capture(blank_capture: CapturedSketch) -> None:
    """
    A canvas with no strokes is refused before the model runs, keeping its capture but recording no suggestion.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    with pytest.raises(InvalidSketchError, match=NO_STROKES_MESSAGE):
        suggest(drawing, blank_capture)

    # The capture is saved and no suggestion exists.
    saved = store.drawings[(SESSION_ID, 1)]
    assert len(saved.captures) == 1 and saved.suggestions == (), f"Expected one capture and no suggestion, got {saved}"
    assert (SESSION_ID, 1, "captures/capture-0001-raw.png") in store.drawing_files, "The refused press's frame must be saved"


@pytest.mark.parametrize(("suggester", "stage", "message"), [
    (FailingSuggester(), SuggestionStage.GENERATION, OUT_OF_MEMORY),
    (MissingWeightsSuggester(), SuggestionStage.MODEL_LOAD, "Run python scripts/download_models.py"),
    (SmallMaskSuggester(), SuggestionStage.GENERATION, "not the 512x360 canvas"),
])
def test_drawing_records_failed_suggestion_and_keeps_previous(sheet_frame: np.ndarray, sheet_capture: CapturedSketch,
                                                              suggester: StrokeSuggester, stage: SuggestionStage,
                                                              message: str) -> None:
    """
    A failing second press records its stage and error, re-raises, and leaves the first suggestion's overlay open.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.suggester = suggester
    with pytest.raises(StrokeSuggestionError, match=message):
        suggest(drawing, sheet_capture)
    failed = store.drawings[(SESSION_ID, 1)].suggestions[1]

    # The failure is saved with its stage, request, and baseline capture, and without a result.
    assert failed.status == SuggestionStatus.FAILED and failed.failure.stage == stage, f"Unexpected failure {failed.failure}"
    assert failed.request is not None and failed.result is None, f"A failed press keeps its request only, got {failed}"
    assert failed.baseline_capture == 2 and failed.shown_at is None, f"Unexpected baseline or time {failed}"

    # The first suggestion stays open, and the preview still shows its overlay.
    _, payload = cv2.imencode(".png", sheet_frame)
    canvas = make_paper_canvas(payload.tobytes(), 0, *CANVAS_SIZE)
    plain = cv2.imdecode(np.frombuffer(canvas.payload, np.uint8), cv2.IMREAD_COLOR)
    preview = cv2.imdecode(np.frombuffer(drawing.compose_preview(canvas), np.uint8), cv2.IMREAD_COLOR)
    assert drawing.open_suggestion.number == 1, f"Suggestion 1 must stay open, got {drawing.open_suggestion}"
    assert not np.array_equal(preview, plain), "The preview must still show suggestion 1's overlay"


def test_drawing_shows_zero_part_suggestion(sheet_capture: CapturedSketch) -> None:
    """
    A suggestion with no parts is shown and recorded with an empty overlay and no colors.
    """
    drawing = start_drawing(MemoryStore(), EmptySuggester())
    suggest(drawing, sheet_capture)
    suggestion = drawing.record.suggestions[0]

    # The attempt is a shown suggestion with no parts and a transparent overlay.
    assert suggestion.status == SuggestionStatus.SHOWN and suggestion.result.parts == (), f"Expected no parts, got {suggestion}"
    assert suggestion.part_colors == () and suggestion.overlay is not None, f"Expected an overlay and no colors, got {suggestion}"


def test_drawing_resaves_capture_after_failed_save(sheet_capture: CapturedSketch) -> None:
    """
    When saving a suggestion fails, the press is recorded as failed at saving, and the next save writes the capture's files.
    """
    store = FlakyDrawingStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    store.failures_left = 1
    with pytest.raises(StoreError, match="No space left"):
        suggest(drawing, sheet_capture)
    saved = store.drawings[(SESSION_ID, 1)]

    # The retry saved the capture with its files and the failed suggestion, without the suggestion's own files.
    files = {path for (_, _, path) in store.drawing_files}
    assert len(saved.captures) == 1 and saved.suggestions[0].failure.stage == SuggestionStage.SAVING, f"Unexpected record {saved}"
    assert "captures/capture-0001-raw.png" in files, f"The capture's frame must be saved on the next save, got {files}"
    assert not any(path.startswith("suggestions/") for path in files), f"A failed suggestion keeps no files, got {files}"


def test_drawing_finish_without_frame_and_preview_without_suggestion(sheet_frame: np.ndarray) -> None:
    """
    A preview without a suggestion is the canvas alone, and finishing without a frame ends the drawing with no final capture.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    _, payload = cv2.imencode(".png", sheet_frame)
    canvas = make_paper_canvas(payload.tobytes(), 0, *CANVAS_SIZE)
    preview = cv2.imdecode(np.frombuffer(drawing.compose_preview(canvas), np.uint8), cv2.IMREAD_COLOR)
    plain = cv2.imdecode(np.frombuffer(canvas.payload, np.uint8), cv2.IMREAD_COLOR)
    record = drawing.finish(None, end=DrawingEnd.NEW_SESSION)

    # The preview equals the canvas, and the drawing ended for a new session without a final capture.
    assert np.array_equal(preview, plain), "Without a suggestion the preview must be the canvas"
    assert (record.end, record.final_capture, record.captures) == (DrawingEnd.NEW_SESSION, None, ()), f"Bad end {record}"


def test_drawing_with_suggestions_off_refuses_presses(sheet_capture: CapturedSketch) -> None:
    """
    A drawing started without a suggester records that suggestions are off and refuses a press.
    """
    drawing = start_drawing(MemoryStore(), None)

    # The record marks suggestions off, and a press is refused.
    assert not drawing.record.suggestions_enabled and drawing.record.suggester is None, f"Bad record {drawing.record}"
    with pytest.raises(SketchLoopError, match="Suggestions are off"):
        suggest(drawing, sheet_capture)


def test_check_suggestion_output_refuses_more_parts_than_requested(sheet_capture: CapturedSketch) -> None:
    """
    An output with more parts than the request allowed is refused, which the output record alone cannot check.
    """
    image, payload = encode_png(np.full((360, 512), 255, dtype=np.uint8), "strokes.png")
    output = FakeStrokeSuggester().suggest(StrokeSuggestionRequest(canvas=image, prompt=""), payload)

    # The fake gives two parts, which a request for one part refuses.
    with pytest.raises(StrokeSuggestionError, match="2 parts for at most 1"):
        check_suggestion_output(output, StrokeSuggestionRequest(canvas=image, prompt="", max_parts=1))


def test_parse_hex_color() -> None:
    """
    A "#RRGGBB" color gives its red, green, and blue levels.
    """
    # The first part color is Okabe-Ito blue.
    assert parse_hex_color(PART_COLORS[0]) == (0, 114, 178), f"Unexpected levels {parse_hex_color(PART_COLORS[0])}"
