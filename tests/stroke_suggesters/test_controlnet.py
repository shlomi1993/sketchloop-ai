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


@pytest.mark.parametrize(("prompt", "expected_prompt", "template"), [
    ("  a ship  ", "a simple black line drawing of a ship, clean thin black outlines on white paper, coloring book page, minimal", "with_prompt"),  # noqa: E501
    ("   ", NEUTRAL_PROMPT, "neutral")
], ids=["padded", "spaces"])
def test_prompt_template_follows_the_typed_prompt(prompt: str, expected_prompt: str, template: str) -> None:
    """
    A typed prompt fills the line-drawing template, a blank one gives the neutral prompt, and both are recorded exactly.
    """
    request, payload = make_canvas()
    request = StrokeSuggestionRequest(canvas=request.canvas, prompt=prompt, seed=3)
    runner = AddingRunner()
    suggestion = make_suggester(runner).suggest(request, payload).suggestion

    # The pipeline gets and the record keeps the exact text, with the fixed negative prompt.
    assert runner.calls[0]["prompt"] == expected_prompt, f"Unexpected pipeline prompt {runner.calls[0]['prompt']!r}"
    assert suggestion.effective.prompt == expected_prompt, f"Unexpected recorded prompt {suggestion.effective.prompt!r}"
    assert suggestion.effective.negative_prompt == NEGATIVE_PROMPT, "The negative prompt must be recorded"

    # The template and the token count are recorded, and a short prompt is not truncated.
    controls = suggestion.effective.controls
    assert controls["prompt_template"] == template, f"Expected template {template}, got {controls['prompt_template']}"
    assert controls["prompt_tokens"] == count_words(expected_prompt), f"Unexpected token count {controls['prompt_tokens']}"
    assert controls["prompt_truncated"] is False, "A short prompt must not be marked truncated"
    assert request.prompt == prompt, "The request must keep the prompt exactly as typed"


def test_long_prompt_is_marked_truncated() -> None:
    """
    A 100-word prompt exceeds the 77-token limit and is recorded as truncated, not refused.
    """
    request, payload = make_canvas()
    request = StrokeSuggestionRequest(canvas=request.canvas, prompt=" ".join(["word"] * 100), seed=3)
    controls = make_suggester(AddingRunner()).suggest(request, payload).suggestion.effective.controls

    # The count is over the limit, so the end of the prompt was cut.
    assert controls["prompt_tokens"] > PROMPT_TOKEN_LIMIT, f"Expected over {PROMPT_TOKEN_LIMIT} tokens, got {controls}"
    assert controls["prompt_truncated"] is True, "A prompt over the token limit must be marked truncated"


def test_pipeline_gets_the_canvas_and_research_settings() -> None:
    """
    The pipeline receives the canvas black on white, its inverse as control, the research defaults, and the canvas size.
    """
    request, payload = make_canvas()
    runner = AddingRunner()
    suggestion = make_suggester(runner).suggest(request, payload).suggestion
    call = runner.calls[0]

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

    # The record names the real suggester and every effective setting.
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


def test_new_parts_exclude_the_existing_strokes() -> None:
    """
    The redrawn rectangle is not suggested, the added circle is one part, and every image is stored with its bytes.
    """
    request, payload = make_canvas()
    output = make_suggester(AddingRunner()).suggest(request, payload)
    suggestion = output.suggestion

    # Only the circle below the rectangle is a part, with a box around it.
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


def test_random_seed_is_drawn_and_recorded() -> None:
    """
    A request without a seed gets a random one, which the pipeline receives and the record keeps.
    """
    request, payload = make_canvas()
    request = StrokeSuggestionRequest(canvas=request.canvas, prompt="")
    runner = AddingRunner()
    suggestion = make_suggester(runner).suggest(request, payload).suggestion

    # The drawn seed is the one the pipeline used.
    assert isinstance(suggestion.seed, int) and suggestion.seed == runner.calls[0]["seed"], f"Seed {suggestion.seed} not recorded"


def test_low_guidance_records_no_negative_prompt() -> None:
    """
    At guidance 1 the pipeline ignores the negative prompt, so none is passed or recorded.
    """
    request, payload = make_canvas()
    request = StrokeSuggestionRequest(canvas=request.canvas, prompt="", seed=3, controls={"guidance_scale": 1.0})
    runner = AddingRunner()
    suggestion = make_suggester(runner).suggest(request, payload).suggestion

    # Neither the call nor the record has a negative prompt.
    assert "negative_prompt" not in runner.calls[0], "A negative prompt must not be passed at guidance 1"
    assert suggestion.effective.negative_prompt is None, "A negative prompt the pipeline ignores must not be recorded"


@pytest.mark.parametrize(("controls", "field"), [
    ({"strength": 0.3}, "controls.strength"),
    ({"steps": 1, "strength": 0.5}, "controls.steps")
], ids=["low-strength", "no-executed-step"])
def test_bad_controls_refused_before_loading(tmp_path: Path, controls: dict[str, Any], field: str) -> None:
    """
    An unknown or out-of-range control, or steps and strength that run no step, are refused before any model loads.
    """
    request, payload = make_canvas()
    request = StrokeSuggestionRequest(canvas=request.canvas, prompt="", seed=3, controls=controls)

    # The suggester has no models, so reaching the load would raise a different error.
    with pytest.raises(UnsupportedConfigurationError) as raised:
        ControlNetStrokeSuggester(tmp_path, device="cpu").suggest(request, payload)
    fields = [issue.field for issue in raised.value.issues]
    assert fields == [field], f"Expected an issue for {field}, got {fields}"


def test_mismatched_canvas_bytes_refused() -> None:
    """
    Canvas bytes that are not the requested canvas are refused, so the record names the image actually used.
    """
    request, _ = make_canvas()
    other = cv2.imencode(".png", np.full((CANVAS_HEIGHT, CANVAS_WIDTH), 255, dtype=np.uint8))[1].tobytes()

    # The suggester refuses the bytes before running the pipeline.
    runner = AddingRunner()
    with pytest.raises(InvalidSketchError, match="do not match"):
        make_suggester(runner).suggest(request, other)
    assert not runner.calls, "The pipeline must not run on the wrong canvas"


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


def test_pipeline_failure_raises_one_line() -> None:
    """
    A failing pipeline raises a one-line StrokeSuggestionError with its first line and a recovery action.
    """
    request, payload = make_canvas()

    # The message keeps only the cause's first line.
    with pytest.raises(StrokeSuggestionError) as raised:
        make_suggester(FailingRunner()).suggest(request, payload)
    expected = "Next-stroke suggestion failed: synthetic sampling failure. Free memory and retry."
    assert str(raised.value) == expected, f"Unexpected message {str(raised.value)!r}"


def test_float16_refused_on_cpu() -> None:
    """
    float16 on the CPU is refused when the suggester is created, naming the setting to change.
    """
    with pytest.raises(StrokeSuggestionError, match="stroke_suggester.controlnet.precision"):
        ControlNetStrokeSuggester(device="cpu", precision="float16")


def test_load_warms_up_once_per_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    load runs one blank suggestion per canvas size, and a suggestion reports a load only when it loaded the models.
    """
    runner = AddingRunner()
    loads = []

    def load_stub() -> tuple[PipelineRunner, Any]:
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


def test_describe_names_the_method_device_and_precision() -> None:
    """
    The description names the model, the device, and the precision.
    """
    description = make_suggester(AddingRunner()).describe()

    # One sentence with the run conditions.
    assert "ControlNet scribble" in description and "CPU" in description and "float32" in description, description


def test_injected_runner_without_tokenizer_records_unavailable_tokens() -> None:
    """
    Without a tokenizer, the token count and truncation are recorded as unavailable rather than guessed.
    """
    request, payload = make_canvas()
    suggester = ControlNetStrokeSuggester(device="cpu", runner=AddingRunner())
    controls = suggester.suggest(request, payload).suggestion.effective.controls

    # Both values say why they are missing.
    assert isinstance(controls["prompt_tokens"], Unavailable), f"Expected unavailable tokens, got {controls['prompt_tokens']}"
    assert isinstance(controls["prompt_truncated"], Unavailable), "Truncation cannot be known without a tokenizer"


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
