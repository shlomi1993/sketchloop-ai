import cv2
import hashlib
import numpy as np
import pytest

from pathlib import Path
from typing import Any

from sketchloop.diffusers_backend import DiffusersSketchGenerator, GenerationBackendError
from sketchloop.domain import GenerationRequest, Guidance, ImageRef
from sketchloop.preprocessing import preprocess_sketch


ROOT = Path(__file__).resolve().parents[1]


class RecordingRunner:
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


def test_diffusers_maps_request() -> None:
    """
    A request maps to pipeline arguments with an inverted 512 px scribble, seed + index seeds, and recorded settings.
    """
    runner = RecordingRunner()
    generator = DiffusersSketchGenerator(mode="quality", device="cpu", runner=runner)
    sketch, payload = make_sketch()
    guidance = Guidance(prompt="chair", negative_prompt="blurry", controls={"steps": 10})
    output = generator.generate(GenerationRequest(sketch=sketch, guidance=guidance, n_candidates=2, seed=7), payload)

    # The pipeline gets the guidance, mode defaults, requested controls, and one seed per candidate.
    first_call = {name: value for name, value in runner.calls[0].items() if name != "image"}
    expected = {"seed": 7, "prompt": "chair", "negative_prompt": "blurry", "width": 512, "height": 384,
                "num_inference_steps": 10, "guidance_scale": 7.5, "controlnet_conditioning_scale": 1.0}
    assert first_call == expected, f"Unexpected pipeline arguments {first_call}"
    assert [call["seed"] for call in runner.calls] == [7, 8], "Each candidate must use seed + index"

    # The black square becomes white on a black background, as the scribble model expects.
    conditioning = runner.calls[0]["image"]
    inverted = conditioning[0, 120, 120].tolist() == [1, 1, 1] and conditioning[0, 300, 400].tolist() == [0, 0, 0]
    assert conditioning.shape == (1, 384, 512, 3) and inverted, "Conditioning must be white strokes on black at 512 px"

    # The result records a real in-process backend and the settings it actually used.
    result = output.result
    effective = result.effective.controls
    recorded = {name: effective[name] for name in ("mode", "device", "dtype", "scheduler", "resolution", "seeds")}
    expected_settings = {"mode": "quality", "device": "cpu", "dtype": "float32", "scheduler": "UniPCMultistepScheduler",
                         "resolution": "512x384", "seeds": "7 8"}
    assert recorded == expected_settings, f"Unexpected effective settings {recorded}"
    assert result.backend.execution == "in_process" and "scribble" in str(result.backend.model_id), "Backend not identified"
    assert [candidate.image.mode for candidate in result.candidates] == ["RGB", "RGB"], "Candidates must be RGB images"


def test_diffusers_missing_model(tmp_path: Path) -> None:
    """
    A missing model folder fails with one line that names the folder and the download script.
    """
    generator = DiffusersSketchGenerator(tmp_path, mode="fast", device="cpu")
    sketch, payload = make_sketch()

    # Generating without the model folder fails with a backend error.
    with pytest.raises(GenerationBackendError) as raised:
        generator.generate(GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair")), payload)

    # The error names the missing folder and the download script on one line.
    message = str(raised.value)
    assert "stable-diffusion-v1-5" in message and "download_models.py" in message, f"Unhelpful error {message!r}"
    assert "\n" not in message, "The error must be a single line"


@pytest.mark.model
def test_diffusers_real_model() -> None:
    """
    The real model turns the example photo into one RGB candidate in fast mode.
    """
    processed = preprocess_sketch((ROOT / "examples" / "sketch-photo.jpg").read_bytes())
    generator = DiffusersSketchGenerator(ROOT / "models", mode="fast")
    request = GenerationRequest(sketch=processed.image, guidance=Guidance(prompt="modern chair"), seed=0)
    output = generator.generate(request, processed.payload)

    # The real backend is labeled real and gives a decodable, non-uniform candidate.
    candidate = output.result.candidates[0]
    pixels = cv2.imdecode(np.frombuffer(output.payloads[candidate.image.path], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert not output.result.backend.is_fake, "The real backend must not be labeled fake"
    assert pixels is not None and pixels.std() > 1, "The candidate must decode to a non-uniform image"
