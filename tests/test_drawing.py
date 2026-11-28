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
from sketchloop.drawing import INTERVAL_JPEG_QUALITY, NO_STROKES_MESSAGE, GuidedDrawing
from sketchloop.experiments import (FINAL_PROMPT_NOT_RECORDED, ORIGIN_NOT_RECORDED, OUTSIDE_DRAWING_REASON, CaptureTrigger,
                                    DrawingEnd, DrawingRecord, DrawingStudy, MemoryStore, PromptOrigin, PromptOriginKind,
                                    PromptSuggestionRecord, StoreError, StudyCondition, SuggestionStage, SuggestionStatus,
                                    WindowEnd, decide_suggestion, drawing_from_dict, drawing_to_dict)
from sketchloop.follow import NO_USABLE_CAPTURE_REASON, PROVISIONAL_RULE, FollowResult, FollowRule
from sketchloop.preprocessing import make_paper_canvas
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester, encode_png
from sketchloop.stroke_suggestion import (LINE_DRAWING_FILE, PART_COLORS, StrokeSuggester, StrokeSuggestionError,
                                          StrokeSuggestionOutput, StrokeSuggestionRequest)


SESSION_ID: Final = "session"
CANVAS_SIZE: Final = (512, 360)
PartDrawer = Callable[[np.ndarray, bytes], np.ndarray]  # Draws a part's mask on a sheet frame
SUGGESTED_PROMPT: Final = "the sketched object, product design render, studio lighting, white background"
PromptMaker = Callable[..., PromptSuggestionRecord]  # Builds a shown prompt suggestion from its text and ID


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


def test_guided_drawing_from_start_to_finish(sheet_frame: np.ndarray, sheet_capture: CapturedSketch,
                                             draw_part_on_sheet: PartDrawer) -> None:
    """
    A drawing with an interval capture, two suggestions, a drawn part, and a finish records, decides, saves, and reloads.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    before = drawing.record_capture(sheet_capture, source=SketchSource.WEBCAM, rotation=0, frame_age_seconds=0.3)
    first = drawing.suggest(sheet_capture, source=SketchSource.WEBCAM, rotation=0, prompt="", frame_age_seconds=0.2)
    part_mask = store.drawing_files[(SESSION_ID, 1, first.result.parts[0].mask.path)]
    with_part = encode_capture(draw_part_on_sheet(sheet_frame, part_mask))
    after = drawing.record_capture(with_part, source=SketchSource.UPLOAD, rotation=0)
    suggest(drawing, with_part, prompt="  a red sports car ")
    second_open = drawing.record.suggestions[1].window is None
    record = drawing.finish(with_part, source=SketchSource.UPLOAD)
    text = json.dumps(drawing_to_dict(record), allow_nan=False)

    # Interval frames are JPEG and press and finish frames lossless PNG, numbered in order, with the quality recorded.
    media = [(capture.trigger, capture.raw.media_type) for capture in record.captures]
    expected_media = [(CaptureTrigger.INTERVAL, "image/jpeg"), (CaptureTrigger.SUGGESTION, "image/png"),
                      (CaptureTrigger.INTERVAL, "image/jpeg"), (CaptureTrigger.SUGGESTION, "image/png"),
                      (CaptureTrigger.FINISH, "image/png")]
    assert media == expected_media, f"Unexpected capture triggers and formats {media}"
    jpeg = store.drawing_files[(SESSION_ID, 1, "captures/capture-0001-raw.jpg")]
    assert jpeg.startswith(b"\xff\xd8\xff"), "The interval frame must be saved as JPEG bytes"
    assert record.interval_jpeg_quality == INTERVAL_JPEG_QUALITY == 90, f"Unexpected quality {record.interval_jpeg_quality}"

    # Each suggestion's baseline is its press, the final capture was taken at finish, and prompts are kept exactly as typed.
    baselines = [suggestion.baseline_capture for suggestion in record.suggestions]
    assert baselines == [2, 4] and record.final_capture == 5, f"Expected baselines 2, 4 and final 5, got {baselines}"
    assert record.end == DrawingEnd.FINISHED and record.ended_at is not None, f"Expected a finished drawing, got {record.end}"
    prompts = [suggestion.request.prompt for suggestion in record.suggestions]
    assert prompts == ["", "  a red sports car "], f"Prompts must be kept exactly as typed, got {prompts}"

    # The overlay has the canvas size in RGBA with one color per part.
    overlay = first.overlay
    assert (overlay.width, overlay.height, overlay.mode) == (*CANVAS_SIZE, ColorMode.RGBA), f"Bad overlay {overlay}"
    assert first.part_colors == PART_COLORS[:2] and first.status == SuggestionStatus.SHOWN, f"Bad suggestion {first}"

    # Every file the record names was saved in the place the design gives, filled masks included.
    paths = {path for (session, number, path) in store.drawing_files if (session, number) == (SESSION_ID, 1)}
    expected = {"captures/capture-0002-raw.png", "captures/capture-0002.png", "suggestions/suggestion-1/strokes.png",
                "suggestions/suggestion-1/line-drawing.png", "suggestions/suggestion-1/part-1.png",
                "suggestions/suggestion-1/overlay.png", "captures/capture-0005-raw.png", "captures/capture-0001-filled.png"}
    assert expected <= paths, f"Missing drawing files {sorted(expected - paths)}"
    assert all((SESSION_ID, 1, capture.filled.path) in store.drawing_files for capture in record.captures), "Bad filled masks"

    # Only captures after a press are aligned to it, and every suggested part is scored at the four tolerances.
    assert before.follow is None and record.captures[1].follow is None, "Captures up to a suggestion's press have no alignment"
    follow = after.follow
    assert follow.suggestion == 1 and follow.alignment.converged, f"Expected an alignment to suggestion 1, got {follow}"
    assert [part.part for part in follow.parts] == [0, 1], f"Expected both parts scored, got {follow.parts}"
    assert set(follow.parts[0].coverage) == {4, 8, 16, 24}, f"Unexpected tolerances {follow.parts[0].coverage}"
    assert {"capture", "canvas", "scoring", "saving"} <= set(after.timings_seconds), f"Bad timings {after.timings_seconds}"

    # Suggestion 2's press ended window 1, where the drawn part was followed and the other not.
    ended = record.suggestions[0]
    assert (ended.window.end, ended.window.last_capture) == (WindowEnd.NEXT_SUGGESTION, 4), f"Unexpected window {ended.window}"
    parts = [(part.result, part.deciding_capture) for part in ended.decision.parts]
    assert parts == [(FollowResult.FOLLOWED, 4), (FollowResult.NOT_FOLLOWED, 4)], f"Unexpected part decisions {parts}"
    assert ended.decision.result == FollowResult.FOLLOWED and ended.decision.rule == PROVISIONAL_RULE, f"Got {ended.decision}"

    # Suggestion 2 stayed open until finish, which ended it where nothing new was drawn.
    last = record.suggestions[1]
    assert second_open and (last.window.end, last.window.last_capture) == (WindowEnd.FINISHED, 5), f"Bad window {last.window}"
    assert last.decision.result == FollowResult.UNDETERMINED, f"Nothing new was drawn, got {last.decision}"

    # The finished record was saved and round-trips through JSON with plain numbers, naming its method and rule once.
    loaded = drawing_from_dict(json.loads(text))
    assert store.drawings[(SESSION_ID, 1)] == record and loaded == record, "drawing.json must hold and reload the record"
    assert (record.follow_detector.name, record.rule) == ("aligned-strokes", PROVISIONAL_RULE), f"Bad method {record}"
    paper_share = drawing_to_dict(record)["captures"][0]["preprocessing"][2]["params"]["paper_share"]
    assert type(paper_share) is float, f"Canvas steps must hold plain floats, got {type(paper_share)}"

    # The stored scores recompute the recorded decision, and a rule needing far more new strokes leaves it undetermined.
    suggestion = loaded.suggestions[0]
    assert decide_suggestion(loaded, suggestion, loaded.rule) == suggestion.decision, "The recorded decision must recompute"
    stricter = decide_suggestion(loaded, suggestion, FollowRule(min_new_stroke_pixels=100_000))
    assert stricter.result == FollowResult.UNDETERMINED, f"A stricter rule must change the result, got {stricter}"

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
    (MissingWeightsSuggester(), SuggestionStage.MODEL_LOAD, "Run python scripts/download_models.py"),
    (SmallMaskSuggester(), SuggestionStage.GENERATION, "not the 512x360 canvas"),
], ids=["model-load", "wrong-mask-size"])
def test_failed_suggestion_keeps_the_previous_one_open(sheet_frame: np.ndarray, sheet_capture: CapturedSketch,
                                                       suggester: StrokeSuggester, stage: SuggestionStage, message: str) -> None:
    """
    A failing second press records its stage, re-raises, and leaves the first suggestion's overlay and window open.
    """
    store = MemoryStore()
    drawing = start_drawing(store, FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.suggester = suggester
    with pytest.raises(StrokeSuggestionError, match=message):
        suggest(drawing, sheet_capture)
    saved = store.drawings[(SESSION_ID, 1)]
    failed = saved.suggestions[1]

    # The failure is saved with its stage, request, and baseline capture, and without a result, and its input is saved.
    assert failed.status == SuggestionStatus.FAILED and failed.failure.stage == stage, f"Unexpected failure {failed.failure}"
    assert (SESSION_ID, 1, failed.request.canvas.path) in store.drawing_files, "The failed press's strokes must be saved"
    assert failed.result is None and failed.baseline_capture == 2 and failed.shown_at is None, f"Unexpected {failed}"

    # The first suggestion stays open and undecided, and the preview still shows its overlay.
    _, payload = cv2.imencode(".png", sheet_frame)
    canvas = make_paper_canvas(payload.tobytes(), 0, *CANVAS_SIZE)
    plain = cv2.imdecode(np.frombuffer(canvas.payload, np.uint8), cv2.IMREAD_COLOR)
    preview = cv2.imdecode(np.frombuffer(drawing.compose_preview(canvas), np.uint8), cv2.IMREAD_COLOR)
    assert drawing.open_suggestion.number == 1, f"Suggestion 1 must stay open, got {drawing.open_suggestion}"
    assert saved.suggestions[0].window is None and saved.suggestions[0].decision is None, "A failed press must not end it"
    assert not np.array_equal(preview, plain), "The preview must still show suggestion 1's overlay"

    # Ending the drawing for a new session ends suggestion 1, whose window holds the failed press's capture.
    record = drawing.finish(None, end=DrawingEnd.NEW_SESSION)
    window = record.suggestions[0].window
    assert (window.end, window.last_capture) == (WindowEnd.NEW_SESSION, 2), f"Unexpected window {window}"
    assert record.suggestions[1].window is None, "A failed suggestion has no window"


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


def test_drawing_ended_before_any_later_capture(sheet_frame: np.ndarray, sheet_capture: CapturedSketch) -> None:
    """
    A preview before any suggestion is the canvas alone, and a window ended before any later capture is undetermined.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    _, payload = cv2.imencode(".png", sheet_frame)
    canvas = make_paper_canvas(payload.tobytes(), 0, *CANVAS_SIZE)
    preview = cv2.imdecode(np.frombuffer(drawing.compose_preview(canvas), np.uint8), cv2.IMREAD_COLOR)
    plain = cv2.imdecode(np.frombuffer(canvas.payload, np.uint8), cv2.IMREAD_COLOR)
    suggest(drawing, sheet_capture)
    record = drawing.finish(None, end=DrawingEnd.NEW_SESSION)
    suggestion = record.suggestions[0]

    # The preview equals the canvas, and the drawing ended for a new session without a final capture.
    assert np.array_equal(preview, plain), "Without a suggestion the preview must be the canvas"
    assert (record.end, record.final_capture) == (DrawingEnd.NEW_SESSION, None), f"Unexpected end {record}"

    # The window has no last capture, and each part has no usable capture.
    assert suggestion.window.last_capture is None, f"Expected an empty window, got {suggestion.window}"
    reasons = {(part.result, part.reason) for part in suggestion.decision.parts}
    assert reasons == {(FollowResult.UNDETERMINED, NO_USABLE_CAPTURE_REASON)}, f"Unexpected part decisions {reasons}"


def test_drawing_survives_failed_saves(sheet_capture: CapturedSketch, make_prompt_suggestion: PromptMaker) -> None:
    """
    Failed saves of a capture, a suggestion, a prompt suggestion, and the finish raise, lose nothing, and can be retried.
    """
    store = FlakyDrawingStore()
    drawing = start_drawing(store, FakeStrokeSuggester())

    # A failed interval capture save keeps the capture, and the next save writes its files.
    store.failures_left = 1
    with pytest.raises(StoreError, match="No space left"):
        drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    files = {path for (_, _, path) in store.drawing_files}
    assert len(store.drawings[(SESSION_ID, 1)].captures) == 2, "Both interval captures must be saved"
    assert "captures/capture-0001-raw.jpg" in files, f"The first capture's frame must be saved on the retry, got {files}"

    # A failed suggestion save records the press as failed at saving, keeping its capture and input but not its output.
    store.failures_left = 1
    with pytest.raises(StoreError, match="No space left"):
        suggest(drawing, sheet_capture)
    saved = store.drawings[(SESSION_ID, 1)]
    files = {path for (_, _, path) in store.drawing_files}
    assert saved.suggestions[0].failure.stage == SuggestionStage.SAVING, f"Unexpected record {saved.suggestions[0]}"
    assert "captures/capture-0003-raw.png" in files, f"The press's frame must be saved on the next save, got {files}"
    suggestion_files = {path for path in files if path.startswith("suggestions/")}
    assert suggestion_files == {"suggestions/suggestion-1/strokes.png"}, f"A failed suggestion keeps only its input, got {files}"

    # A failed prompt suggestion save leaves no entry the page never showed, in memory or in the store.
    store.failures_left = 1
    with pytest.raises(StoreError, match="No space left"):
        drawing.record_prompt_suggestion(make_prompt_suggestion(SUGGESTED_PROMPT))
    assert drawing.record.prompt_suggestions == (), f"The entry must be removed, got {drawing.record.prompt_suggestions}"
    assert store.drawings[(SESSION_ID, 1)].prompt_suggestions == (), "The store must not hold the unsaved entry"

    # A failed finish leaves the drawing and its last window open, so finishing again ends and decides it.
    suggest(drawing, sheet_capture)
    store.failures_left = 1
    with pytest.raises(StoreError, match="No space left"):
        drawing.finish(None)
    assert drawing.is_open and drawing.record.suggestions[1].window is None, "A failed finish must reopen the window"
    record = drawing.finish(None)
    assert record.suggestions[1].window.end == WindowEnd.FINISHED, f"Unexpected window {record.suggestions[1].window}"


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


def test_drawing_saved_by_older_versions_loads(sheet_capture: CapturedSketch) -> None:
    """
    A drawing.json without decisions, study fields, or masks loads them as unrecorded, and one without alignment loads too.
    """
    drawing = start_drawing(MemoryStore(), FakeStrokeSuggester())
    suggest(drawing, sheet_capture)
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    data = drawing_to_dict(drawing.finish(None))
    unaligned = json.loads(json.dumps(data))

    # Delete what T09 slices and T10 added, standing in for a drawing saved before them.
    for key in ("rule", "interval_jpeg_quality", "study", "prompt_suggestions", "final_prompt", "final_prompt_origin"):
        del data[key]
    for capture in data["captures"]:
        del capture["filled"]
    del data["captures"][1]["follow"]["alignment"]["filled_share"], data["captures"][1]["follow"]["alignment"]["aligned_on"]
    del data["suggestions"][0]["window"], data["suggestions"][0]["decision"], data["suggestions"][0]["prompt_origin"]
    loaded = drawing_from_dict(data)
    suggestion, alignment = loaded.suggestions[0], loaded.captures[1].follow.alignment

    # The older drawing has no rule, window, decision, masks, or study fields, and its scores can still be decided.
    assert loaded.rule is None and suggestion.window is None and suggestion.decision is None, f"Unexpected {suggestion}"
    assert decide_suggestion(loaded, suggestion, PROVISIONAL_RULE).parts, "An older drawing can still be decided from its scores"
    assert all(capture.filled is None for capture in loaded.captures), "Older captures have no filled mask"
    unrecorded = (loaded.interval_jpeg_quality, alignment.filled_share, alignment.aligned_on)
    assert all(isinstance(value, Unavailable) for value in unrecorded), f"Expected unrecorded values, got {unrecorded}"
    assert loaded.study is None and loaded.prompt_suggestions is None, f"Expected no study fields, got {loaded}"
    assert loaded.final_prompt == Unavailable(FINAL_PROMPT_NOT_RECORDED), f"Unexpected final prompt {loaded.final_prompt}"
    origins = (loaded.final_prompt_origin, suggestion.prompt_origin)
    assert origins == (Unavailable(ORIGIN_NOT_RECORDED),) * 2, f"Expected origins not recorded, got {origins}"

    # A drawing saved before follow detection loads with no follow method and no capture alignment.
    del unaligned["follow_detector"], unaligned["rule"]
    for capture in unaligned["captures"]:
        del capture["follow"]
    for suggestion_data in unaligned["suggestions"]:
        del suggestion_data["window"], suggestion_data["decision"]
    older = drawing_from_dict(unaligned)
    assert older.follow_detector is None and older.captures[1].follow is None, f"Unexpected older drawing {older}"


def break_window_record(data: dict[str, object], case: str) -> None:
    """
    Change a saved drawing with two suggestions and a finish so its windows contradict each other in the named way.
    """
    first = data["suggestions"][0]
    if case == "decision-without-window":
        first["window"] = None
    elif case == "wrong-last-capture":
        first["window"]["last_capture"] = 4
    elif case == "deciding-outside":
        first["decision"]["parts"][0] |= {"deciding_capture": 4, "result": "not_followed", "reason": None}
        first["decision"]["result"] = "not_followed"


@pytest.mark.parametrize(("case", "message"), [
    ("decision-without-window", "decision exactly when its window ended"),
    ("wrong-last-capture", "window ends at capture 3"),
    ("deciding-outside", "decided by captures"),
])
def test_drawing_with_contradicting_windows_is_refused(sheet_capture: CapturedSketch, case: str, message: str) -> None:
    """
    A drawing.json whose windows end in the wrong place or decide from outside them is refused on load.
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


def start_study_drawing(store: MemoryStore, condition: StudyCondition) -> GuidedDrawing:
    """
    Start a study drawing with the fake suggester in the AI condition and none without AI.
    """
    suggester = FakeStrokeSuggester() if condition == StudyCondition.AI else None
    return GuidedDrawing(store, SESSION_ID, 1, suggester, canvas_width=CANVAS_SIZE[0], canvas_height=CANVAS_SIZE[1],
                         save_interval_seconds=2.0, study=DrawingStudy(condition=condition))


def test_no_ai_drawing_refuses_all_ai_help(sheet_capture: CapturedSketch, make_prompt_suggestion: PromptMaker) -> None:
    """
    A drawing without AI records its condition with suggestions off and refuses both kinds of suggestion.
    """
    store = MemoryStore()
    drawing = start_study_drawing(store, StudyCondition.NO_AI)
    saved = store.drawings[(SESSION_ID, 1)]

    # The saved record has the condition, suggestions off, no suggester, and an empty prompt suggestion list.
    assert saved.study == DrawingStudy(condition=StudyCondition.NO_AI), f"Expected the no-AI condition, got {saved.study}"
    assert not saved.suggestions_enabled and saved.suggester is None, f"Expected no AI help, got {saved}"
    assert saved.prompt_suggestions == () and saved.final_prompt is None, f"Expected an empty open drawing, got {saved}"

    # Both kinds of suggestion are refused, and nothing is recorded.
    with pytest.raises(SketchLoopError, match="Suggestions are off"):
        suggest(drawing, sheet_capture)
    with pytest.raises(SketchLoopError, match="Suggestions are off in this drawing."):
        drawing.record_prompt_suggestion(make_prompt_suggestion(SUGGESTED_PROMPT))
    assert store.drawings[(SESSION_ID, 1)].prompt_suggestions == (), "A refused prompt suggestion must not be recorded"


def test_prompt_suggestions_link_to_stroke_suggestions(sheet_capture: CapturedSketch, make_prompt_suggestion: PromptMaker) -> None:  # noqa: E501
    """
    Each stroke suggestion records whether its prompt was suggested, edited, typed, or named an unknown prompt suggestion.
    """
    store = MemoryStore()
    drawing = start_study_drawing(store, StudyCondition.AI)
    first = drawing.record_prompt_suggestion(make_prompt_suggestion(SUGGESTED_PROMPT, "prompt-a", number=9))
    drawing.record_capture(sheet_capture, source=SketchSource.UPLOAD, rotation=0)
    second = drawing.record_prompt_suggestion(make_prompt_suggestion(SUGGESTED_PROMPT, "prompt-b"))
    same = drawing.suggest(sheet_capture, source=SketchSource.UPLOAD, rotation=0, prompt=f" {SUGGESTED_PROMPT}",
                           prompt_suggestion_id="prompt-a")
    edited = drawing.suggest(sheet_capture, source=SketchSource.UPLOAD, rotation=0, prompt="a lamp, product design render",
                             prompt_suggestion_id="prompt-b")
    stale = drawing.suggest(sheet_capture, source=SketchSource.UPLOAD, rotation=0, prompt="a lamp",
                            prompt_suggestion_id="from-another-drawing")
    typed = drawing.suggest(sheet_capture, source=SketchSource.UPLOAD, rotation=0, prompt="a lamp")

    # The drawing numbers each prompt suggestion and records the latest capture when it was shown.
    assert (first.number, first.after_capture) == (1, None), f"Expected number 1 before any capture, got {first}"
    assert (second.number, second.after_capture) == (2, 1), f"Expected number 2 after capture 1, got {second}"
    assert drawing.find_prompt_suggestion("prompt-b") == second and drawing.find_prompt_suggestion("x") is None, "Bad lookup"

    # The same text is suggested, a changed text edited, an unknown ID unavailable, and no ID typed, and every press is shown.
    origins = [suggestion.prompt_origin for suggestion in (same, edited, stale, typed)]
    expected = [PromptOrigin(kind=PromptOriginKind.SUGGESTED, prompt_suggestion="prompt-a"),
                PromptOrigin(kind=PromptOriginKind.EDITED, prompt_suggestion="prompt-b"), Unavailable(OUTSIDE_DRAWING_REASON),
                PromptOrigin(kind=PromptOriginKind.TYPED, prompt_suggestion=None)]
    assert origins == expected, f"Unexpected prompt origins {origins}"
    assert stale.status == SuggestionStatus.SHOWN, "An unknown prompt suggestion ID must not block the press"

    # The saved record holds both prompt suggestions and loads back unchanged from JSON.
    record = store.drawings[(SESSION_ID, 1)]
    assert record.prompt_suggestions == (first, second), f"Expected both prompt suggestions, got {record.prompt_suggestions}"
    data = json.loads(json.dumps(drawing_to_dict(record), allow_nan=False))
    assert drawing_from_dict(data) == record, "drawing.json with prompt suggestions must load back unchanged"
    assert data["suggestions"][0]["prompt_origin"] == {"kind": "suggested", "prompt_suggestion": "prompt-a"}, "Bad JSON origin"


@pytest.mark.parametrize(("prompt", "suggestion_id", "end", "expected_prompt", "expected_origin"), [
    ("a sports car, low and wide", "prompt-a", DrawingEnd.FINISHED, "a sports car, low and wide",
     PromptOrigin(kind=PromptOriginKind.EDITED, prompt_suggestion="prompt-a")),
    ("a car", "prompt-a", DrawingEnd.NEW_SESSION, None, None),
], ids=["edited", "new-session"])
def test_drawing_records_final_prompt(make_prompt_suggestion: PromptMaker, prompt: str | None, suggestion_id: str | None,
                                      end: DrawingEnd, expected_prompt: str | None,
                                      expected_origin: PromptOrigin | None) -> None:
    """
    Finishing records the final prompt and its origin, and a new session records none.
    """
    store = MemoryStore()
    drawing = start_study_drawing(store, StudyCondition.AI)
    drawing.record_prompt_suggestion(make_prompt_suggestion(SUGGESTED_PROMPT, "prompt-a"))
    record = drawing.finish(None, end=end, prompt=prompt, prompt_suggestion_id=suggestion_id)

    # The final prompt and its origin are recorded as given at finish, and none when a new session ended the drawing.
    assert record.final_prompt == expected_prompt, f"Expected final prompt {expected_prompt!r}, got {record.final_prompt!r}"
    assert record.final_prompt_origin == expected_origin, f"Expected origin {expected_origin}, got {record.final_prompt_origin}"
    assert drawing_from_dict(json.loads(json.dumps(drawing_to_dict(record)))) == record, "The finished drawing must load back"


def break_study_record(data: dict[str, object], case: str) -> None:
    """
    Change a saved AI drawing with two prompt suggestions, a linked stroke suggestion, and a final prompt in the named way.
    """
    if case == "condition-disagrees":
        data["study"] = {"condition": "no_ai"}
    elif case == "kind-disagrees":
        data["suggestions"][0]["prompt_origin"]["kind"] = "edited"


@pytest.mark.parametrize(("case", "message"), [
    ("condition-disagrees", "condition no_ai has suggestions off"),
    ("kind-disagrees", "recorded as edited, but its prompt and suggestion give suggested"),
])
def test_drawing_with_inconsistent_study_fields_is_refused(sheet_capture: CapturedSketch, make_prompt_suggestion: PromptMaker,
                                                           case: str, message: str) -> None:
    """
    A drawing.json whose condition, prompt origins, or final prompt contradict each other is refused on load.
    """
    drawing = start_study_drawing(MemoryStore(), StudyCondition.AI)
    drawing.record_prompt_suggestion(make_prompt_suggestion(SUGGESTED_PROMPT, "prompt-a"))
    drawing.suggest(sheet_capture, source=SketchSource.UPLOAD, rotation=0, prompt=SUGGESTED_PROMPT,
                    prompt_suggestion_id="prompt-a")
    drawing.record_prompt_suggestion(make_prompt_suggestion("a lamp, concept art", "prompt-b"))
    record = drawing.finish(None, prompt="a lamp, concept art", prompt_suggestion_id="prompt-b")
    data = json.loads(json.dumps(drawing_to_dict(record)))
    break_study_record(data, case)

    # The record names the broken rule.
    with pytest.raises(InvalidRecordError, match=message):
        drawing_from_dict(data)
