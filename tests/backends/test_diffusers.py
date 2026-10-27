"""
Tests of the diffusers backend with an injected pipeline, plus one opt-in real-model test.
"""

import cv2
import hashlib
import numpy as np
import pytest

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sketchloop.backends.diffusers import (DiffusersSketchGenerator, GenerationBackendError, PipelineRunner, TorchDevice,
                                           describe_torch_runtime)
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

    # The description names the device, mode, precision, and default step count, so the person expects the run time.
    description = generator.describe()
    assert "on CPU in quality mode and float32, running 20 steps" in description, f"Unexpected description {description}"


@pytest.mark.parametrize(("device", "precision", "expected"), [
    ("cuda", None, "float16"),
    ("mps", None, "float16"),
    ("mps", "float32", "float32"),
    ("cpu", None, "float32"),
], ids=["cuda-default", "mps-default", "mps-float32", "cpu-default"])
def test_diffusers_precision(device: str, precision: str | None, expected: str) -> None:
    """
    float16 is the default on a GPU, float32 can still be chosen there, and the precision is recorded as the effective dtype.
    """
    generator = DiffusersSketchGenerator(mode="quality", device=device, precision=precision, runner=RecordingRunner())
    sketch, payload = make_sketch()
    output = generator.generate(GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair"), seed=1), payload)

    # The record and the description both name the precision the models ran in.
    dtype = output.result.effective.controls["dtype"]
    assert dtype == expected, f"Expected dtype {expected}, got {dtype}"
    assert f"and {expected}," in generator.describe(), f"The description must name {expected}"


def test_diffusers_refuses_float16_on_cpu() -> None:
    """
    float16 on the CPU is refused when the backend is created, before any model is loaded.
    """
    with pytest.raises(GenerationBackendError, match="float16 needs a GPU"):
        DiffusersSketchGenerator(mode="fast", device="cpu", precision="float16")


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


@pytest.mark.usefixtures("counting_clock")
def test_diffusers_reports_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An injected runner loads nothing, and a lazy load is timed in the first call only.
    """
    sketch, payload = make_sketch()
    request = GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair"), seed=1)
    injected = DiffusersSketchGenerator(mode="quality", device="cpu", runner=RecordingRunner()).generate(request, payload)

    # Patch the load to return a stub, which the counting clock times as one 7 s step.
    lazy = DiffusersSketchGenerator(mode="quality", device="cpu")
    monkeypatch.setattr(lazy, "_load_runner", RecordingRunner)
    first, second = lazy.generate(request, payload), lazy.generate(request, payload)

    # Only the call that loaded reports the load, and an injected runner never does.
    loads = (injected.load_seconds, first.load_seconds, second.load_seconds)
    assert loads == (None, 7.0, None), f"Expected loads None, 7.0, None, got {loads}"


class FailingRunner(PipelineRunner):
    """
    Test-only pipeline stand-in that fails every call, like sampling that runs out of memory.
    """

    def __call__(self, seed: int, **pipeline_kwargs: Any) -> np.ndarray:
        """
        Raise as a failed sampling step would.
        """
        raise RuntimeError("synthetic sampling failure")


@pytest.mark.usefixtures("counting_clock")
def test_diffusers_failure_keeps_load_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When the lazy load succeeds but sampling fails, the error notes how long the load took.
    """
    sketch, payload = make_sketch()
    generator = DiffusersSketchGenerator(mode="quality", device="cpu")
    monkeypatch.setattr(generator, "_load_runner", FailingRunner)

    # Generation fails with a backend error that carries the load time as a note.
    with pytest.raises(GenerationBackendError) as raised:
        generator.generate(GenerationRequest(sketch=sketch, guidance=Guidance(prompt="chair"), seed=1), payload)
    notes = getattr(raised.value, "__notes__", [])
    assert notes == ["Model load took 7.00 s before generation failed."], f"Expected the load time note, got {notes}"


def test_describe_torch_runtime() -> None:
    """
    The runtime facts name the torch thread count on every device and the GPU on CUDA only.
    """
    torch_stub = SimpleNamespace(get_num_threads=lambda: 6, cuda=SimpleNamespace(get_device_name=lambda index: "Synthetic GPU"))

    # CPU and MPS record only threads, and CUDA adds the accelerator name.
    assert describe_torch_runtime(torch_stub, TorchDevice.CPU) == {"torch_threads": 6}, "CPU must record only threads"
    assert describe_torch_runtime(torch_stub, TorchDevice.MPS) == {"torch_threads": 6}, "MPS must record only threads"
    cuda = describe_torch_runtime(torch_stub, TorchDevice.CUDA)
    assert cuda == {"torch_threads": 6, "accelerator": "Synthetic GPU"}, f"CUDA must add the accelerator, got {cuda}"


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
