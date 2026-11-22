"""
Tests of guided drawings: captures, suggestions with their overlay, failures, finishing, and the composed preview.
"""

import cv2
import json
import numpy as np
import pytest
import threading

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Final

from sketchloop.capture import CapturedSketch, InvalidSketchError, encode_frame
from sketchloop.domain import ColorMode, InvalidRecordError, SketchLoopError, SketchSource, Unavailable
from sketchloop.drawing import (INTERVAL_JPEG_QUALITY, NO_STROKES_MESSAGE, GuidedDrawing, check_suggestion_output,
                                compress_interval_frame, parse_hex_color)
from sketchloop.experiments import (CaptureTrigger, DrawingEnd, DrawingRecord, MemoryStore, StoreError, SuggestionStage,
                                    SuggestionStatus, WindowEnd, decide_suggestion, drawing_from_dict, drawing_to_dict)
from sketchloop.follow import NO_USABLE_CAPTURE_REASON, PROVISIONAL_RULE, FollowResult, FollowRule
from sketchloop.preprocessing import make_paper_canvas
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester, encode_png
from sketchloop.stroke_suggestion import (LINE_DRAWING_FILE, PART_COLORS, StrokeSuggester, StrokeSuggestionError,
                                          StrokeSuggestionOutput, StrokeSuggestionRequest)


SESSION_ID: Final = "session"
CANVAS_SIZE: Final = (512, 360)
PartDrawer = Callable[[np.ndarray, bytes], np.ndarray]  # Draws a part's mask on a sheet frame
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


class BlockingSuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester that waits inside suggest until released, as a model run takes several seconds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Signal that the suggestion started, wait for the release, then suggest as the fake does.
        """
        self.started.set()
        self.release.wait(10)
        return super().suggest(request, canvas_payload)


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


def encode_capture(frame: np.ndarray) -> CapturedSketch:
    """
    Encode a frame as the PNG raw sketch the browser controller would capture, with a fixed 0.01 s capture time.
    """
    image, payload = encode_frame(frame)
    return CapturedSketch(image=image, payload=payload, capture_seconds=0.01)


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

    # The failure is saved with its stage, request, and baseline capture, and without a result, and its input is saved.
    assert failed.status == SuggestionStatus.FAILED and failed.failure.stage == stage, f"Unexpected failure {failed.failure}"
    assert (SESSION_ID, 1, failed.request.canvas.path) in store.drawing_files, "The failed press's strokes must be saved"
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
    suggestion_files = {path for path in files if path.startswith("suggestions/")}
    assert suggestion_files == {"suggestions/suggestion-1/strokes.png"}, f"A failed suggestion keeps only its input, got {files}"


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


def test_drawing_interval_captures_are_jpeg_and_aligned(sheet_frame: np.ndarray, sheet_capture: CapturedSketch) -> None:
    """
    Interval frames are kept as JPEG and press frames as PNG, and captures after a suggestion record their alignment.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    before = drawing.record_capture(sheet_capture, source=SketchSource.WEBCAM, rotation=0, frame_age_seconds=0.3)
    suggest(drawing, sheet_capture)
    shifted = encode_capture(np.roll(sheet_frame, (2, 3), axis=(0, 1)))
    after = drawing.record_capture(shifted, source=SketchSource.UPLOAD, rotation=0)
    record = drawing.finish(sheet_capture, source=SketchSource.UPLOAD)

    # Interval frames are JPEG, while the press and finish frames stay lossless PNG.
    media = [(capture.trigger, capture.raw.media_type, capture.raw.path) for capture in record.captures]
    expected = [(CaptureTrigger.INTERVAL, "image/jpeg", "captures/capture-0001-raw.jpg"),
                (CaptureTrigger.SUGGESTION, "image/png", "captures/capture-0002-raw.png"),
                (CaptureTrigger.INTERVAL, "image/jpeg", "captures/capture-0003-raw.jpg"),
                (CaptureTrigger.FINISH, "image/png", "captures/capture-0004-raw.png")]
    assert media == expected, f"Unexpected capture formats {media}"
    jpeg = store.drawing_files[(SESSION_ID, 1, "captures/capture-0001-raw.jpg")]
    assert jpeg.startswith(b"\xff\xd8\xff"), "The interval frame must be saved as JPEG bytes"

    # The JPEG quality is recorded, and every canvas has its filled mask saved beside it.
    assert record.interval_jpeg_quality == INTERVAL_JPEG_QUALITY == 90, f"Unexpected quality {record.interval_jpeg_quality}"
    filled_paths = [capture.filled.path for capture in record.captures]
    assert filled_paths[0] == "captures/capture-0001-filled.png", f"Unexpected filled mask paths {filled_paths}"
    assert all((SESSION_ID, 1, path) in store.drawing_files for path in filled_paths), "Every filled mask must be saved"

    # Only captures after the press are aligned, to suggestion 1, and the shifted frame aligns with its time recorded.
    assert before.follow is None and record.captures[1].follow is None, "Captures before a suggestion have no alignment"
    follow = after.follow
    assert follow.suggestion == 1 and follow.alignment.converged, f"Expected an alignment to suggestion 1, got {follow}"
    assert follow.new_stroke_pixels is not None and follow.seconds_since_shown >= 0, f"Unexpected scores {follow}"

    # Every suggested part is scored, in index order, at the four tolerances.
    part_count = len(record.suggestions[0].result.parts)
    indexes = [part.part for part in follow.parts]
    assert part_count > 0 and indexes == list(range(part_count)), f"Expected {part_count} scored parts, got {indexes}"
    assert set(follow.parts[0].coverage) == {4, 8, 16, 24}, f"Unexpected tolerances {follow.parts[0].coverage}"
    assert record.captures[3].follow.suggestion == 1, "The finish capture is aligned to the open suggestion too"
    stages = set(after.timings_seconds)
    assert {"capture", "canvas", "scoring", "saving"} <= stages, f"Unexpected capture timings {stages}"

    # The follow method is recorded once, and the whole record round-trips through JSON unchanged.
    assert record.follow_detector.name == "aligned-strokes", f"Unexpected follow method {record.follow_detector}"
    text = json.dumps(drawing_to_dict(record), allow_nan=False)
    assert drawing_from_dict(json.loads(text)) == record, "drawing.json with captures must load back unchanged"


def test_drawing_without_part_scores_loads(sheet_capture: CapturedSketch) -> None:
    """
    A drawing.json saved before part scores, filled masks and shares, and the JPEG quality loads them as not recorded.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    data = drawing_to_dict(drawing.record)
    del data["interval_jpeg_quality"]
    for capture in data["captures"]:
        del capture["filled"]
    del data["captures"][1]["follow"]["parts"]
    del data["captures"][1]["follow"]["alignment"]["filled_share"]
    loaded = drawing_from_dict(data)

    # The quality is unavailable with a reason, and the masks and part scores are absent.
    assert isinstance(loaded.interval_jpeg_quality, Unavailable), f"Expected no quality, got {loaded.interval_jpeg_quality}"
    assert all(capture.filled is None for capture in loaded.captures), "Older captures have no filled mask"
    follow = loaded.captures[1].follow
    assert follow.parts is None and follow.alignment.converged, f"Expected an alignment without part scores, got {follow}"
    assert isinstance(follow.alignment.filled_share, Unavailable), f"Expected no filled share, got {follow.alignment}"


def test_drawing_part_scores_must_match_the_suggestion(sheet_capture: CapturedSketch) -> None:
    """
    A capture whose part scores do not cover every part of the open suggestion is refused when loaded.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    data = drawing_to_dict(drawing.record)
    data["captures"][1]["follow"]["parts"] = []

    # The record names the mismatch.
    with pytest.raises(InvalidRecordError, match="scores 0 parts, not suggestion 1's"):
        drawing_from_dict(data)


def test_drawing_skips_interval_capture_during_suggestion(sheet_capture: CapturedSketch) -> None:
    """
    An interval capture that arrives while a suggestion runs is skipped at once, rather than saving a stale frame later.
    """
    suggester = BlockingSuggester()
    drawing = start_drawing(MemoryStore(), suggester)
    pressing = threading.Thread(target=suggest, args=(drawing, sheet_capture))
    pressing.start()
    try:
        assert suggester.started.wait(10), "The suggestion must start"
        skipped = drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    finally:
        suggester.release.set()
        pressing.join(10)

    # The skipped call saved nothing, and the next one after the suggestion saves an aligned capture.
    assert skipped is None and len(drawing.record.captures) == 1, f"Expected only the press capture, got {skipped}"
    after = drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    assert after.number == 2 and after.follow.suggestion == 1, f"Expected capture 2 aligned to suggestion 1, got {after}"


def test_drawing_without_alignment_fields_loads(sheet_capture: CapturedSketch) -> None:
    """
    A drawing.json written before follow detection, without follow fields, loads with no alignment and no follow method.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    data = drawing_to_dict(drawing.record)
    del data["follow_detector"], data["rule"]
    for capture in data["captures"]:
        del capture["follow"]
    loaded = drawing_from_dict(data)

    # The older record loads with no follow method and no capture alignment.
    assert loaded.follow_detector is None and loaded.captures[0].follow is None, f"Unexpected older drawing {loaded}"


def test_drawing_keeps_interval_capture_after_failed_save(sheet_capture: CapturedSketch) -> None:
    """
    A failed save of an interval capture raises, keeps the capture, and writes its files with the next save.
    """
    store = FlakyDrawingStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    store.failures_left = 1
    with pytest.raises(StoreError, match="No space left"):
        drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)

    # Both captures and the first one's files reached the store.
    files = {path for (_, _, path) in store.drawing_files}
    assert len(store.drawings[(SESSION_ID, 1)].captures) == 2, "Both interval captures must be saved"
    assert "captures/capture-0001-raw.jpg" in files, f"The first capture's frame must be saved on the retry, got {files}"


def test_compress_interval_frame_keeps_jpeg(sheet_frame: np.ndarray) -> None:
    """
    A JPEG frame is kept as received, while a PNG frame is re-encoded as JPEG with the same size.
    """
    jpeg_payload = cv2.imencode(".jpg", sheet_frame)[1].tobytes()
    png = encode_capture(sheet_frame)
    jpeg = CapturedSketch(image=replace(png.image, path="sketch-raw.jpg", media_type="image/jpeg"), payload=jpeg_payload,
                          capture_seconds=0.01)
    compressed = compress_interval_frame(png)

    # The JPEG is unchanged, and the PNG became a JPEG of the same size with its re-encoding timed.
    assert compress_interval_frame(jpeg) is jpeg, "A JPEG frame must be kept as received"
    size = (compressed.image.width, compressed.image.height, compressed.image.media_type)
    assert size == (640, 480, "image/jpeg") and compressed.capture_seconds >= 0.01, f"Unexpected compressed frame {size}"



def test_drawing_windows_end_at_the_next_suggestion_and_at_finish(sheet_frame: np.ndarray, sheet_capture: CapturedSketch,
                                                                   draw_part_on_sheet: PartDrawer) -> None:
    """
    A shown suggestion ends the previous window at its press, finish ends the last one, and each gets its decision.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    first = drawing.suggest(sheet_capture, source=SketchSource.WEBCAM, rotation=0, prompt="", frame_age_seconds=0.2)
    part_mask = store.drawing_files[(SESSION_ID, 1, first.result.parts[0].mask.path)]
    with_part = encode_capture(draw_part_on_sheet(sheet_frame, part_mask))
    drawing.record_capture(with_part, source=SketchSource.UPLOAD, rotation=0)
    suggest(drawing, with_part)

    # Suggestion 1's window ended at suggestion 2's press, capture 3, with the drawn part followed and the other not.
    ended = drawing.record.suggestions[0]
    assert (ended.window.end, ended.window.last_capture) == (WindowEnd.NEXT_SUGGESTION, 3), f"Unexpected window {ended.window}"
    parts = [(part.result, part.deciding_capture) for part in ended.decision.parts]
    assert parts == [(FollowResult.FOLLOWED, 3), (FollowResult.NOT_FOLLOWED, 3)], f"Unexpected part decisions {parts}"
    assert ended.decision.result == FollowResult.FOLLOWED and ended.decision.rule == PROVISIONAL_RULE, f"Got {ended.decision}"

    # Suggestion 2 stays open until finish, which ends it at the final capture, where nothing new was drawn.
    assert drawing.record.suggestions[1].window is None, "The latest suggestion's window must stay open"
    record = drawing.finish(with_part, source=SketchSource.UPLOAD)
    last = record.suggestions[1]
    assert (last.window.end, last.window.last_capture) == (WindowEnd.FINISHED, 4), f"Unexpected window {last.window}"
    assert last.decision.result == FollowResult.UNDETERMINED, f"Nothing new was drawn, got {last.decision}"

    # The rule is recorded once for the drawing, and the record with windows round-trips and was saved.
    assert record.rule == PROVISIONAL_RULE and store.drawings[(SESSION_ID, 1)] == record, "The decided record must be saved"
    assert drawing_from_dict(json.loads(json.dumps(drawing_to_dict(record)))) == record, "Windows must load back unchanged"


def test_drawing_failed_press_keeps_the_window_open(sheet_capture: CapturedSketch) -> None:
    """
    A failed second press neither ends nor decides the first window, which then holds the failed press's capture.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.suggester = FailingSuggester()
    with pytest.raises(StrokeSuggestionError):
        suggest(drawing, sheet_capture)

    # The saved record keeps suggestion 1 open after the failure.
    saved = store.drawings[(SESSION_ID, 1)]
    assert saved.suggestions[0].window is None and saved.suggestions[0].decision is None, "A failed press must not end it"

    # Ending the drawing for a new session ends suggestion 1, whose window holds the failed press's capture.
    record = drawing.finish(None, end=DrawingEnd.NEW_SESSION)
    window = record.suggestions[0].window
    assert (window.end, window.last_capture) == (WindowEnd.NEW_SESSION, 2), f"Unexpected window {window}"
    assert record.suggestions[1].window is None, "A failed suggestion has no window"


def test_drawing_empty_window_is_undetermined(sheet_capture: CapturedSketch) -> None:
    """
    A window that ends before any capture is taken holds no capture and decides every part as undetermined.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    record = drawing.finish(None, end=DrawingEnd.NEW_SESSION)
    suggestion = record.suggestions[0]

    # The window has no last capture, and each part has no usable capture.
    assert suggestion.window.last_capture is None, f"Expected an empty window, got {suggestion.window}"
    reasons = {(part.result, part.reason) for part in suggestion.decision.parts}
    assert reasons == {(FollowResult.UNDETERMINED, NO_USABLE_CAPTURE_REASON)}, f"Unexpected part decisions {reasons}"


def test_drawing_failed_finish_reopens_the_window(sheet_capture: CapturedSketch) -> None:
    """
    A finish whose save fails leaves the drawing and its last window open, so finishing again ends and decides it.
    """
    store = FlakyDrawingStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    store.failures_left = 1
    with pytest.raises(StoreError, match="No space left"):
        drawing.finish(None)

    # The window is open again, and the next finish ends it.
    assert drawing.is_open and drawing.record.suggestions[0].window is None, "A failed finish must reopen the window"
    record = drawing.finish(None)
    assert record.suggestions[0].window.end == WindowEnd.FINISHED, f"Unexpected window {record.suggestions[0].window}"


def test_decide_suggestion_recomputes_a_stored_drawing(sheet_frame: np.ndarray, sheet_capture: CapturedSketch,
                                                       draw_part_on_sheet: PartDrawer) -> None:
    """
    A loaded drawing's suggestion can be decided again from its stored scores with another rule.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    first = drawing.suggest(sheet_capture, source=SketchSource.WEBCAM, rotation=0, prompt="", frame_age_seconds=0.2)
    part_mask = store.drawing_files[(SESSION_ID, 1, first.result.parts[0].mask.path)]
    drawing.record_capture(encode_capture(draw_part_on_sheet(sheet_frame, part_mask)), source=SketchSource.UPLOAD, rotation=0)
    loaded = drawing_from_dict(json.loads(json.dumps(drawing_to_dict(drawing.finish(None)))))
    suggestion = loaded.suggestions[0]

    # The recorded rule gives the recorded decision, and a rule needing far more new strokes leaves it undetermined.
    assert decide_suggestion(loaded, suggestion, loaded.rule) == suggestion.decision, "The recorded decision must recompute"
    assert suggestion.decision.result == FollowResult.FOLLOWED, f"The drawn part must be followed, got {suggestion.decision}"
    stricter = decide_suggestion(loaded, suggestion, FollowRule(min_new_stroke_pixels=100_000))
    assert stricter.result == FollowResult.UNDETERMINED, f"A stricter rule must change the result, got {stricter}"


def break_window_record(data: dict[str, object], case: str) -> None:
    """
    Change a saved drawing with two suggestions and a finish so its windows contradict each other in the named way.
    """
    first, second = data["suggestions"]
    if case == "decision-without-window":
        first["window"] = None
    elif case == "two-open-windows":
        first["window"], first["decision"] = None, None
    elif case == "wrong-last-capture":
        first["window"]["last_capture"] = 4
    elif case == "next-without-a-next":
        second["window"]["end"] = "next_suggestion"
    elif case == "deciding-outside":
        first["decision"]["parts"][0] |= {"deciding_capture": 4, "result": "not_followed", "reason": None}
        first["decision"]["result"] = "not_followed"
    elif case == "windows-without-rule":
        data["rule"] = None


@pytest.mark.parametrize(("case", "message"), [
    ("decision-without-window", "decision exactly when its window ended"),
    ("two-open-windows", "window must be ended"),
    ("wrong-last-capture", "window ends at capture 3"),
    ("next-without-a-next", "ends by next_suggestion exactly when one followed"),
    ("deciding-outside", "decided by captures"),
    ("windows-without-rule", "without a rule has no suggestion windows"),
])
def test_drawing_with_contradicting_windows_is_refused(sheet_capture: CapturedSketch, case: str, message: str) -> None:
    """
    A drawing.json whose windows overlap, end in the wrong place, or decide from outside them is refused on load.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    suggest(drawing, sheet_capture)
    data = json.loads(json.dumps(drawing_to_dict(drawing.finish(sheet_capture, source=SketchSource.UPLOAD))))
    break_window_record(data, case)

    # The record names the contradiction.
    with pytest.raises(InvalidRecordError, match=message):
        drawing_from_dict(data)


def test_drawing_saved_before_decisions_loads(sheet_capture: CapturedSketch) -> None:
    """
    A finished drawing saved before windows and decisions, without a rule, loads with its suggestions undecided.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    data = drawing_to_dict(drawing.finish(sheet_capture, source=SketchSource.UPLOAD))
    del data["rule"]
    for suggestion in data["suggestions"]:
        del suggestion["window"], suggestion["decision"]
    loaded = drawing_from_dict(data)

    # No rule and no window or decision, while the scores stay for a later decision.
    suggestion = loaded.suggestions[0]
    assert loaded.rule is None and suggestion.window is None and suggestion.decision is None, f"Unexpected {suggestion}"
    assert decide_suggestion(loaded, suggestion, PROVISIONAL_RULE).parts, "An older drawing can still be decided from its scores"
