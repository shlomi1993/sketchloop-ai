"""
Tests of the ControlNet stroke suggester with injected pipelines, plus one opt-in real-model test.
"""

import cv2
import hashlib
import numpy as np
import pytest
import sys
import time

from pathlib import Path
from typing import Any

from sketchloop.backends.diffusers import PipelineRunner
from sketchloop.capture import InvalidSketchError
from sketchloop.domain import ColorMode, ImageRef, Unavailable, UnsupportedConfigurationError
from sketchloop.preprocessing import extract_canvas_strokes, make_paper_canvas
from sketchloop.stroke_suggesters.controlnet import (NEGATIVE_PROMPT, NEUTRAL_PROMPT, PROMPT_TOKEN_LIMIT,
                                                     ControlNetStrokeSuggester)
from sketchloop.stroke_suggestion import StrokeSuggestionError, StrokeSuggestionRequest


ROOT = Path(__file__).resolve().parents[2]
CANVAS_WIDTH, CANVAS_HEIGHT = 512, 360


class AddingRunner(PipelineRunner):
    """
    Test-only stand-in for the img2img pipeline that redraws the start image and adds a circle below the middle.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, seed: int, **pipeline_kwargs: Any) -> np.ndarray:
        """
        Record the call, then return the start image with a thin black circle added, as RGB in [0, 1].
        """
        self.calls.append({"seed": seed, **pipeline_kwargs})
        image = (pipeline_kwargs["image"][0] * 255).astype(np.uint8).copy()
        cv2.circle(image, (256, 290), 40, (0, 0, 0), 3)
        return image.astype(np.float32) / 255


class FailingRunner(PipelineRunner):
    """
    Test-only pipeline stand-in that fails every call, like sampling that runs out of memory.
    """

    def __call__(self, seed: int, **pipeline_kwargs: Any) -> np.ndarray:
        """
        Raise as a failed sampling step would.
        """
        raise RuntimeError("synthetic sampling failure\nwith a second line")


def count_words(text: str) -> int:
    """
    Count words plus the start and end tokens, a test stand-in for the CLIP tokenizer.
    """
    return len(text.split()) + 2


def make_canvas() -> tuple[StrokeSuggestionRequest, bytes]:
    """
    Build a white 512x360 canvas with a black rectangle and a request for it, with seed 3 and no prompt.
    """
    pixels = np.full((CANVAS_HEIGHT, CANVAS_WIDTH), 255, dtype=np.uint8)
    cv2.rectangle(pixels, (156, 100), (356, 200), 0, 3)
    payload = cv2.imencode(".png", pixels)[1].tobytes()
    canvas = ImageRef(path="strokes.png", width=CANVAS_WIDTH, height=CANVAS_HEIGHT, mode=ColorMode.L,
                      media_type="image/png", sha256=hashlib.sha256(payload).hexdigest())
    return StrokeSuggestionRequest(canvas=canvas, prompt="", seed=3), payload


def make_suggester(runner: PipelineRunner) -> ControlNetStrokeSuggester:
    """
    Create a CPU suggester with an injected runner and the word-counting tokenizer.
    """
    return ControlNetStrokeSuggester(device="cpu", runner=runner, count_tokens=count_words)


@pytest.mark.parametrize(("prompt", "expected_prompt", "template", "truncated"), [
    ("  a ship  ", "a simple black line drawing of a ship, clean thin black outlines on white paper, coloring book page, minimal", "with_prompt", False),  # noqa: E501
    ("   ", NEUTRAL_PROMPT, "neutral", False),
    (" ".join(["word"] * 100), None, "with_prompt", True),
], ids=["padded", "spaces", "long"])
def test_prompt_template_follows_the_typed_prompt(prompt: str, expected_prompt: str | None, template: str,
                                                  truncated: bool) -> None:
    """
    A typed prompt fills the line-drawing template, a blank one gives the neutral prompt, and a long one is marked truncated.
    """
    request, payload = make_canvas()
    request = StrokeSuggestionRequest(canvas=request.canvas, prompt=prompt, seed=3)
    runner = AddingRunner()
    suggestion = make_suggester(runner).suggest(request, payload).suggestion
    sent = runner.calls[0]["prompt"]

    # The pipeline gets and the record keeps the exact text, with the fixed negative prompt, and the request keeps the typed one.
    assert expected_prompt in (None, sent), f"Unexpected pipeline prompt {sent!r}"
    assert suggestion.effective.prompt == sent, f"Unexpected recorded prompt {suggestion.effective.prompt!r}"
    assert suggestion.effective.negative_prompt == NEGATIVE_PROMPT, "The negative prompt must be recorded"
    assert request.prompt == prompt, "The request must keep the prompt exactly as typed"

    # The template, the token count, and whether the prompt was cut at the token limit are recorded.
    controls = suggestion.effective.controls
    assert controls["prompt_template"] == template, f"Expected template {template}, got {controls['prompt_template']}"
    assert controls["prompt_tokens"] == count_words(sent), f"Unexpected token count {controls['prompt_tokens']}"
    assert (controls["prompt_tokens"] > PROMPT_TOKEN_LIMIT) == truncated, f"Unexpected token count {controls}"
    assert controls["prompt_truncated"] is truncated, f"Expected truncated {truncated}, got {controls['prompt_truncated']}"


def test_suggestion_gets_the_canvas_and_research_settings_and_finds_new_parts() -> None:
    """
    The pipeline gets the canvas, its inverse as control, and the research defaults, and only the added circle is a part.
    """
    request, payload = make_canvas()
    runner = AddingRunner()
    suggester = make_suggester(runner)
    output = suggester.suggest(request, payload)
    suggestion, call = output.suggestion, runner.calls[0]

    # The defaults from the research note reach the pipeline with the canvas size and the seed.
    settings = {name: call[name] for name in ("seed", "strength", "num_inference_steps", "guidance_scale",
                                              "controlnet_conditioning_scale", "width", "height")}
    expected = {"seed": 3, "strength": 0.85, "num_inference_steps": 5, "guidance_scale": 1.5,
                "controlnet_conditioning_scale": 0.8, "width": 512, "height": 360}
    assert settings == expected, f"Unexpected pipeline settings {settings}"
    assert call["negative_prompt"] == NEGATIVE_PROMPT, "The negative prompt must reach the pipeline"

    # The start image has black strokes on white, and the control image is its inverse.
    start, control = call["image"], call["control_image"]
    assert start.shape == control.shape == (1, 360, 512, 3), f"Unexpected image shapes {start.shape}, {control.shape}"
    assert start[0, 100, 200].tolist() == [0, 0, 0] and start[0, 10, 10].tolist() == [1, 1, 1], "Start must be black on white"
    assert np.array_equal(control, 1 - start), "The control image must be the inverted canvas"

    # The record names the real suggester and every effective setting, and the description names the run conditions.
    controls = suggestion.effective.controls
    recorded = {name: controls[name] for name in ("strength", "steps", "executed_steps", "guidance_scale", "conditioning_scale",
                                                  "device", "dtype", "scheduler", "lcm_lora_fused", "width", "height")}
    expected_record = {"strength": 0.85, "steps": 5, "executed_steps": 4, "guidance_scale": 1.5, "conditioning_scale": 0.8,
                       "device": "cpu", "dtype": "float32", "scheduler": "LCMScheduler", "lcm_lora_fused": True,
                       "width": 512, "height": 360}
    assert recorded == expected_record, f"Unexpected effective settings {recorded}"
    assert controls["extract_existing_stroke_kernel_pixels"] == 13, "Extraction settings must be recorded"
    identity = suggestion.suggester
    assert identity.adapter == "sketchloop.controlnet-strokes" and not identity.is_fake, f"Unexpected identity {identity}"
    assert "lllyasviel/control_v11p_sd15_scribble@" in str(identity.model_id), f"Unexpected model ID {identity.model_id}"
    description = suggester.describe()
    assert "ControlNet scribble" in description and "CPU" in description and "float32" in description, description

    # The redrawn rectangle is not suggested, and only the circle below it is a part, with a box around it.
    assert len(suggestion.parts) == 1, f"Expected only the added circle, got {len(suggestion.parts)} parts"
    left, top, width, height = suggestion.parts[0].box
    assert abs(left - 214) <= 3 and abs(top - 248) <= 3 and abs(width - 85) <= 4, f"Unexpected box {suggestion.parts[0].box}"
    assert height >= 80, f"The circle part must be whole, got {suggestion.parts[0].box}"

    # The masks and the RGB line drawing are stored at the canvas size, and the seed and timings are recorded.
    mask = suggestion.parts[0].mask
    assert (mask.path, mask.width, mask.height) == ("part-1.png", 512, 360), f"Unexpected mask {mask}"
    line_drawing = suggestion.line_drawing
    assert (line_drawing.path, line_drawing.mode) == ("line-drawing.png", "RGB"), f"Unexpected line drawing {line_drawing}"
    assert set(output.payloads) == {"line-drawing.png", "part-1.png"}, f"Unexpected payloads {sorted(output.payloads)}"
    assert suggestion.seed == 3, f"Expected the requested seed, got {suggestion.seed}"
    assert set(suggestion.timings_seconds) == {"generation", "extraction"}, f"Unexpected timings {suggestion.timings_seconds}"


def test_minimal_request_records_what_it_could_not_know() -> None:
    """
    Without a seed, a tokenizer, or guidance above 1, a random seed is recorded, tokens are unavailable, and no negative prompt.
    """
    request, payload = make_canvas()
    request = StrokeSuggestionRequest(canvas=request.canvas, prompt="", controls={"guidance_scale": 1.0})
    runner = AddingRunner()
    suggestion = ControlNetStrokeSuggester(device="cpu", runner=runner).suggest(request, payload).suggestion
    controls = suggestion.effective.controls

    # The drawn seed is the one the pipeline used.
    assert isinstance(suggestion.seed, int) and suggestion.seed == runner.calls[0]["seed"], f"Seed {suggestion.seed} not recorded"

    # Without a tokenizer, the token count and truncation say why they are missing.
    assert isinstance(controls["prompt_tokens"], Unavailable), f"Expected unavailable tokens, got {controls['prompt_tokens']}"
    assert isinstance(controls["prompt_truncated"], Unavailable), "Truncation cannot be known without a tokenizer"

    # At guidance 1 the pipeline ignores the negative prompt, so neither the call nor the record has one.
    assert "negative_prompt" not in runner.calls[0], "A negative prompt must not be passed at guidance 1"
    assert suggestion.effective.negative_prompt is None, "A negative prompt the pipeline ignores must not be recorded"


def test_bad_requests_and_failures_are_refused_in_one_line(tmp_path: Path) -> None:
    """
    Steps that run nothing, float16 on the CPU, canvas bytes of another image, and a failing pipeline each raise one line.
    """
    request, payload = make_canvas()

    # Steps and strength that run no step are refused before any model loads, since this suggester has none.
    no_step = StrokeSuggestionRequest(canvas=request.canvas, prompt="", seed=3, controls={"steps": 1, "strength": 0.5})
    with pytest.raises(UnsupportedConfigurationError) as raised:
        ControlNetStrokeSuggester(tmp_path, device="cpu").suggest(no_step, payload)
    assert [issue.field for issue in raised.value.issues] == ["controls.steps"], f"Unexpected issues {raised.value.issues}"

    # float16 on the CPU is refused when the suggester is created, naming the setting to change.
    with pytest.raises(StrokeSuggestionError, match="stroke_suggester.controlnet.precision"):
        ControlNetStrokeSuggester(device="cpu", precision="float16")

    # Canvas bytes that are not the requested canvas are refused before the pipeline runs.
    other = cv2.imencode(".png", np.full((CANVAS_HEIGHT, CANVAS_WIDTH), 255, dtype=np.uint8))[1].tobytes()
    runner = AddingRunner()
    with pytest.raises(InvalidSketchError, match="do not match"):
        make_suggester(runner).suggest(request, other)
    assert not runner.calls, "The pipeline must not run on the wrong canvas"

    # A failing pipeline keeps only the cause's first line and adds a recovery action.
    with pytest.raises(StrokeSuggestionError) as failed:
        make_suggester(FailingRunner()).suggest(request, payload)
    expected = "Next-stroke suggestion failed: synthetic sampling failure. Free memory and retry."
    assert str(failed.value) == expected, f"Unexpected message {str(failed.value)!r}"


def test_missing_weights_reported_before_library_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A missing model folder fails with one line naming the folder and the download script, before torch or diffusers load.
    """
    monkeypatch.setitem(sys.modules, "diffusers", None)
    request, payload = make_canvas()
    suggester = ControlNetStrokeSuggester(tmp_path, device="cpu")

    # Both the warm-up and a suggestion name the missing folder, not the blocked library import.
    for attempt in (lambda: suggester.load(512, 360), lambda: suggester.suggest(request, payload)):
        with pytest.raises(StrokeSuggestionError) as raised:
            attempt()
        message = str(raised.value)
        assert "stable-diffusion-v1-5 is missing" in message and "download_models.py" in message, f"Unhelpful error {message!r}"
        assert "\n" not in message, "The error must be a single line"


def test_load_warms_up_once_per_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    load runs one blank suggestion per canvas size, and a suggestion reports a load only when it loaded the models.
    """
    runner = AddingRunner()
    loads = []

    def load_stub() -> tuple[PipelineRunner, Any]:
        """
        Count the load and return the stub runner and tokenizer.
        """
        loads.append(time.perf_counter())
        return runner, count_words

    # Loading twice at one size loads and warms up once, at that size, on a blank canvas.
    warmed = ControlNetStrokeSuggester(device="cpu")
    monkeypatch.setattr(warmed, "_load_pipeline", load_stub)
    warmed.load(512, 360)
    warmed.load(512, 360)
    assert (len(loads), len(runner.calls)) == (1, 1), f"Expected one load and one warm-up, got {len(loads)}, {len(runner.calls)}"
    warm_up = runner.calls[0]
    assert (warm_up["width"], warm_up["height"]) == (512, 360) and warm_up["image"].min() == 1, "Warm-up must use a blank canvas"

    # After a warm-up a suggestion reports no load, while a lazy first suggestion reports one.
    request, payload = make_canvas()
    assert warmed.suggest(request, payload).load_seconds is None, "A warmed suggester must report no load"
    lazy = ControlNetStrokeSuggester(device="cpu")
    monkeypatch.setattr(lazy, "_load_pipeline", load_stub)
    first, second = lazy.suggest(request, payload), lazy.suggest(request, payload)
    assert first.load_seconds is not None and second.load_seconds is None, "Only the loading call reports the load"


def erase_chair_legs(photo_payload: bytes) -> bytes:
    """
    Paint over the legs of the example chair photo with the surrounding paper, leaving the back and seat to continue.
    """
    photo = cv2.imdecode(np.frombuffer(photo_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    legs = np.zeros_like(gray)
    legs[455:620, 560:960] = np.where(gray[455:620, 560:960] < 120, 255, 0)
    legless = cv2.inpaint(photo, cv2.dilate(legs, np.ones((9, 9), np.uint8)), 7, cv2.INPAINT_TELEA)
    return cv2.imencode(".png", legless)[1].tobytes()


@pytest.mark.model
def test_controlnet_suggester_real_model() -> None:
    """
    The real suggester, warmed up at 512x360, suggests new parts for the example photo's chair without legs.
    """
    photo = erase_chair_legs((ROOT / "examples" / "sketch-photo.jpg").read_bytes())
    strokes = extract_canvas_strokes(make_paper_canvas(photo, 0, 512, 360).payload)
    suggester = ControlNetStrokeSuggester(ROOT / "models")
    suggester.load(512, 360)

    # Suggest with a typed prompt for three seeds and once with an empty prompt.
    part_counts = {}
    for prompt, seed in (("a chair", 0), ("a chair", 1), ("a chair", 2), ("", 0)):
        output = suggester.suggest(StrokeSuggestionRequest(canvas=strokes.image, prompt=prompt, seed=seed), strokes.payload)
        suggestion = output.suggestion
        part_counts[(prompt, seed)] = len(suggestion.parts)

        # Each suggestion is labeled real, has masks of canvas size, and runs in seconds after the warm-up.
        assert not suggestion.suggester.is_fake, "The real suggester must not be labeled fake"
        sizes = {(part.mask.width, part.mask.height) for part in suggestion.parts}
        assert sizes <= {(512, 360)}, f"Masks must have the canvas size, got {sizes}"
        assert output.load_seconds is None, "The warm-up must have loaded the models"
        assert suggestion.timings_seconds["generation"] < 10, f"Generation took {suggestion.timings_seconds}"

    # A chair without legs gets new parts for the typed prompt with at least one seed, which the owner judges by eye.
    assert any(part_counts[("a chair", seed)] for seed in range(3)), f"Expected new parts for a chair, got {part_counts}"
