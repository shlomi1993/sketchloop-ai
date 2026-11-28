"""
Tests of the T2I-Adapter sketch backend with an injected pipeline, plus one opt-in real-model test.
"""

import cv2
import hashlib
import numpy as np
import pytest

from pathlib import Path
from typing import Any

from sketchloop.backends.diffusers import GenerationBackendError, PipelineRunner
from sketchloop.backends.t2i_adapter import T2IAdapterSketchGenerator
from sketchloop.domain import GenerationRequest, Guidance, ImageRef
from sketchloop.preprocessing import preprocess_sketch


ROOT = Path(__file__).resolve().parents[2]


class RecordingRunner(PipelineRunner):
    """
    Test-only stand-in for the Diffusers pipeline that records each call and returns a flat gray image.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, seed: int, **pipeline_kwargs: Any) -> np.ndarray:
        """
        Record the seed and pipeline arguments and return a gray RGB image of the requested size.
        """
        self.calls.append({"seed": seed, **pipeline_kwargs})
        return np.full((pipeline_kwargs["height"], pipeline_kwargs["width"], 3), 0.5, dtype=np.float32)


def make_sketch() -> tuple[ImageRef, bytes]:
    """
    Build a 64 x 48 white sketch with one black square, and its PNG reference and bytes.
    """
    pixels = np.full((48, 64), 255, dtype=np.uint8)
    pixels[10:20, 10:20] = 0
    payload = cv2.imencode(".png", pixels)[1].tobytes()
    sketch = ImageRef(path="sketch.png", width=64, height=48, mode="L", media_type="image/png",
                      sha256=hashlib.sha256(payload).hexdigest())
    return sketch, payload


def test_t2i_adapter_maps_request() -> None:
    """
    A request maps to adapter arguments with a white-on-black PIL sketch and the adapter scale, and fast mode adds the LoRA.
    """
    runner = RecordingRunner()
    generator = T2IAdapterSketchGenerator(mode="quality", device="cpu", runner=runner)
    sketch, payload = make_sketch()
    guidance = Guidance(prompt="chair", negative_prompt="blurry", controls={"steps": 10, "conditioning_scale": 0.8})
    output = generator.generate(GenerationRequest(sketch=sketch, guidance=guidance, n_candidates=2, seed=7), payload)

    # The pipeline gets the guidance, mode defaults, and the conditioning scale under the adapter's argument name.
    first_call = {name: value for name, value in runner.calls[0].items() if name != "image"}
    expected = {"seed": 7, "prompt": "chair", "negative_prompt": "blurry", "width": 512, "height": 384,
                "num_inference_steps": 10, "guidance_scale": 7.5, "adapter_conditioning_scale": 0.8}
    assert first_call == expected, f"Unexpected pipeline arguments {first_call}"
    assert [call["seed"] for call in runner.calls] == [7, 8], "Each candidate must use seed + index"

    # The sketch arrives as a grayscale PIL image at 512 px, with the black square turned white on black.
    conditioning = runner.calls[0]["image"]
    assert conditioning.mode == "L" and conditioning.size == (512, 384), f"Expected a 512x384 L image, got {conditioning}"
    assert conditioning.getpixel((120, 120)) == 255 and conditioning.getpixel((400, 300)) == 0, "Strokes must be white on black"

    # The result names the adapter backend and model, not the ControlNet, and keeps the shared effective settings.
    result = output.result
    model_id = str(result.backend.model_id)
    assert result.backend.adapter == "sketchloop.t2i-adapter", f"Unexpected adapter name {result.backend.adapter}"
    assert "TencentARC/t2iadapter_sketch_sd15v2" in model_id and "scribble" not in model_id, f"Unexpected model {model_id}"
    assert "lcm-lora" not in model_id, f"Quality mode must not use the LCM-LoRA, got {model_id}"
    effective = result.effective.controls
    recorded = {name: effective[name] for name in ("mode", "scheduler", "resolution", "seeds", "conditioning_scale")}
    expected_settings = {"mode": "quality", "scheduler": "UniPCMultistepScheduler", "resolution": "512x384", "seeds": "7 8",
                         "conditioning_scale": 0.8}
    assert recorded == expected_settings, f"Unexpected effective settings {recorded}"

    # Fast mode uses the LCM settings and a default adapter scale of 1.0.
    runner = RecordingRunner()
    generator = T2IAdapterSketchGenerator(mode="fast", device="cpu", runner=runner)
    output = generator.generate(GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair"), seed=1), payload)
    call = runner.calls[0]
    defaults = (call["num_inference_steps"], call["guidance_scale"], call["adapter_conditioning_scale"])
    assert defaults == (4, 1.5, 1.0), f"Expected 4 steps, guidance 1.5, and scale 1.0, got {defaults}"
    assert "controlnet_conditioning_scale" not in call, "The adapter pipeline takes no ControlNet scale"

    # The recorded model includes the LoRA, and the description names the adapter and the run conditions.
    model_id = str(output.result.backend.model_id)
    assert "latent-consistency/lcm-lora-sdv1-5" in model_id, f"Fast mode must record the LCM-LoRA, got {model_id}"
    description = generator.describe()
    assert description.startswith("Using the t2i-adapter backend with "), f"Expected the t2i-adapter name, got {description}"
    assert "T2I-Adapter sketch" in description, f"The description must name the adapter, got {description}"
    assert "on CPU in fast mode and float32, running 4 steps" in description, f"Unexpected description {description}"


def test_t2i_adapter_missing_model(tmp_path: Path) -> None:
    """
    A missing adapter folder fails with one line that names the folder and the download script.
    """
    (tmp_path / "stable-diffusion-v1-5").mkdir()
    generator = T2IAdapterSketchGenerator(tmp_path, mode="quality", device="cpu")
    sketch, payload = make_sketch()

    # Generating without the adapter folder fails with a backend error.
    with pytest.raises(GenerationBackendError) as raised:
        generator.generate(GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair")), payload)

    # The error names the adapter folder and the download script on one line.
    message = str(raised.value)
    assert "t2iadapter_sketch_sd15v2" in message and "download_models.py" in message, f"Unhelpful error {message!r}"
    assert "\n" not in message, "The error must be a single line"


@pytest.mark.model
def test_t2i_adapter_real_model() -> None:
    """
    The real adapter model turns the example photo into one RGB candidate in fast mode.
    """
    processed = preprocess_sketch((ROOT / "examples" / "sketch-photo.jpg").read_bytes())
    generator = T2IAdapterSketchGenerator(ROOT / "models", mode="fast")
    request = GenerationRequest(sketch=processed.image, guidance=Guidance(prompt="modern chair"), seed=0)
    output = generator.generate(request, processed.payload)

    # The real backend is labeled real and gives a decodable, non-uniform candidate.
    candidate = output.result.candidates[0]
    pixels = cv2.imdecode(np.frombuffer(output.payloads[candidate.image.path], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert not output.result.backend.is_fake, "The real backend must not be labeled fake"
    assert pixels is not None and pixels.std() > 1, "The candidate must decode to a non-uniform image"
