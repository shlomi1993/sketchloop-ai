import cv2
import hashlib
import numpy as np
import random
import uuid

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol

from sketchloop.capture import InvalidSketchError
from sketchloop.domain import (BackendIdentity, Candidate, ControlValue, EffectiveSettings, GenerationRequest,
                               GenerationResult, ImageRef, SketchLoopError, first_line)
from sketchloop.generation import ControlSpec, GenerationOutput, GeneratorCapabilities, validate_request
from sketchloop.preprocessing import STROKE_CONTRAST


# The scribble ControlNet works at 512 px, and Stable Diffusion needs sides divisible by 8.
CONDITIONING_LONG_SIDE_PIXELS: Final = 512
LATENT_SIDE_MULTIPLE: Final = 8
LCM_LORA_WEIGHTS: Final = "pytorch_lora_weights.safetensors"
MISSING_EXTRA_MESSAGE: Final = 'The diffusers backend needs the generation extra. Run pip install -e ".[generation]" and retry.'


class TorchDevice(StrEnum):
    """
    Torch devices the diffusers backend can run on.
    """
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class GenerationMode(StrEnum):
    """
    Speed and quality trade-off of the diffusers backend.
    """
    FAST = "fast"
    QUALITY = "quality"


class GenerationBackendError(SketchLoopError, RuntimeError):
    """
    A real generation backend cannot load its models or fails while generating.
    """


@dataclass(frozen=True, slots=True)
class ModelSource:
    """
    A Hugging Face repository, its pinned revision, its local folder name, and the files the backend needs from it.
    """
    folder: str  # Local folder name under models/
    repo_id: str  # Hugging Face repository ID
    revision: str  # Pinned commit SHA to download
    allow_patterns: tuple[str, ...]  # File patterns the backend needs, so unused weights are skipped


@dataclass(frozen=True, slots=True)
class ModeSettings:
    """
    Default sampling settings and scheduler class of one generation mode.
    """
    steps: int  # Default number of denoising steps
    guidance_scale: float  # Default prompt strength (classifier-free guidance)
    scheduler: str  # Diffusers scheduler class name, such as LCMScheduler


# Pinned revisions from the first-model research note, with only the safetensors weights and configs Diffusers loads.
BASE_MODEL: Final = ModelSource(
    folder="stable-diffusion-v1-5",
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    allow_patterns=("*.json", "*.txt", "*/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"),
)
CONTROLNET_MODEL: Final = ModelSource(
    folder="control_v11p_sd15_scribble",
    repo_id="lllyasviel/control_v11p_sd15_scribble",
    revision="3564ec7b87dd706d5fc760d21c9e90bb160d1e9f",
    allow_patterns=("config.json", "diffusion_pytorch_model.safetensors"),
)
LCM_LORA_MODEL: Final = ModelSource(
    folder="lcm-lora-sdv1-5",
    repo_id="latent-consistency/lcm-lora-sdv1-5",
    revision="cf2fced511dbe7e26c8d1d397e728fbab875db4b",
    allow_patterns=(LCM_LORA_WEIGHTS,),
)
MODEL_SOURCES: Final = (BASE_MODEL, CONTROLNET_MODEL, LCM_LORA_MODEL)

# Fast mode samples in 4 LCM-LoRA steps for CPUs, and quality mode uses the standard ControlNet setup.
MODE_SETTINGS: Final[dict[str, ModeSettings]] = {
    GenerationMode.FAST: ModeSettings(steps=4, guidance_scale=1.5, scheduler="LCMScheduler"),
    GenerationMode.QUALITY: ModeSettings(steps=20, guidance_scale=7.5, scheduler="UniPCMultistepScheduler")
}


class PipelineRunner(Protocol):
    """
    Runs the loaded pipeline once for one seed, so tests can inject a stub instead of real weights.
    """

    def __call__(self, seed: int, **pipeline_kwargs: Any) -> np.ndarray:
        """
        Generate one image.

        Args:
            seed (int): Seed for this candidate's random generator.
            **pipeline_kwargs (Any): Keyword arguments for the Diffusers pipeline call.

        Returns:
            np.ndarray: RGB image of shape (height, width, 3) with values in [0, 1].
        """
        ...


def select_device() -> TorchDevice:
    """
    Pick the fastest available torch device: CUDA, then Apple MPS, then CPU.

    Returns:
        TorchDevice: The torch device.
    """
    # Import torch only here, so the core package works without the generation extra.
    try:
        import torch
    except ImportError as error:
        raise GenerationBackendError(MISSING_EXTRA_MESSAGE) from error

    # Prefer an NVIDIA GPU, then Apple Silicon, then the CPU.
    if torch.cuda.is_available():
        return TorchDevice.CUDA
    if torch.backends.mps.is_available():
        return TorchDevice.MPS
    return TorchDevice.CPU


def scribble_conditioning(sketch_payload: bytes) -> np.ndarray:
    """
    Turn a dark-on-light sketch into white strokes on black, 512 px on the long side, as the scribble ControlNet expects.

    Args:
        sketch_payload (bytes): Encoded sketch image.

    Returns:
        np.ndarray: Grayscale conditioning image with values 0 or 255 and sides divisible by 8.
    """
    gray = cv2.imdecode(np.frombuffer(sketch_payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise InvalidSketchError("Sketch cannot be decoded for generation. Save it as PNG or JPEG and retry.")

    # Scale the long side to 512 px, rounding both sides to the latent grid.
    height, width = gray.shape
    scale = CONDITIONING_LONG_SIDE_PIXELS / max(width, height)
    multiple = LATENT_SIDE_MULTIPLE
    size = tuple(max(multiple, round(side * scale / multiple) * multiple) for side in (width, height))
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # Mark pixels clearly darker than the typical paper level as strokes, using the preprocessing threshold.
    strokes = resized < np.median(resized) - STROKE_CONTRAST
    return np.where(strokes, 255, 0).astype(np.uint8)


def encode_rgb_png(image: np.ndarray) -> tuple[bytes, int, int]:
    """
    Encode a pipeline image with values in [0, 1] as an 8-bit RGB PNG.

    Args:
        image (np.ndarray): RGB image of shape (height, width, 3).

    Returns:
        tuple[bytes, int, int]: PNG bytes, height, and width.
    """
    rgb = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    encoded, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not encoded:
        raise GenerationBackendError("Generated image cannot be encoded as PNG. Retry the generation.")

    return buffer.tobytes(), rgb.shape[0], rgb.shape[1]


def local_revision(folder: Path) -> str:
    """
    Read the commit a Hugging Face download recorded for a model folder.

    Args:
        folder (Path): Local model folder.

    Returns:
        str: The first 12 characters of the commit hash, or "unknown-revision" when none was recorded.
    """
    # Each downloaded file's metadata starts with the commit hash it came from.
    metadata = next(folder.glob(".cache/huggingface/download/**/*.metadata"), None)
    return metadata.read_text(encoding="utf-8").split("\n", 1)[0][:12] if metadata else "unknown-revision"


class DiffusersSketchGenerator:
    """
    Stable Diffusion 1.5 with the ControlNet v1.1 scribble model, loaded lazily from local folders.
    """

    def __init__(self, model_dir: Path = Path("models"), *, mode: GenerationMode | None = None,
                 device: TorchDevice | None = None, runner: PipelineRunner | None = None) -> None:
        self._model_dir = model_dir
        self.device = TorchDevice(device) if device else select_device()
        default_mode = GenerationMode.FAST if self.device == TorchDevice.CPU else GenerationMode.QUALITY
        self.mode = GenerationMode(mode) if mode else default_mode
        self._runner = runner

        # Negative prompts need classifier-free guidance, which the low LCM guidance scale effectively disables.
        settings = MODE_SETTINGS[self.mode]
        controls = (
            ControlSpec(name="steps", kind="int", minimum=1, maximum=50, default=settings.steps),
            ControlSpec(name="guidance_scale", kind="float", minimum=0, maximum=20, default=settings.guidance_scale),
            ControlSpec(name="conditioning_scale", kind="float", minimum=0, maximum=2, default=1.0)
        )
        supports_negative_prompt = self.mode == GenerationMode.QUALITY
        self._capabilities = GeneratorCapabilities(controls=controls, max_candidates=8,
                                                   supports_negative_prompt=supports_negative_prompt, supports_seed=True)

    @property
    def capabilities(self) -> GeneratorCapabilities:
        """
        Return the backend's capabilities in the current mode.
        """
        return self._capabilities

    def _list_model_sources(self) -> tuple[ModelSource, ...]:
        return MODEL_SOURCES if self.mode == GenerationMode.FAST else (BASE_MODEL, CONTROLNET_MODEL)

    def _load_runner(self) -> PipelineRunner:
        # Report a missing download before the slow library imports.
        for source in self._list_model_sources():
            folder = self._model_dir / source.folder
            if not folder.is_dir():
                message = f"Model folder {folder.as_posix()} is missing. Run python scripts/download_models.py and retry."
                raise GenerationBackendError(message)

        # Import the heavy libraries only when a real model is needed.
        try:
            import diffusers
            import torch
            import transformers
        except ImportError as error:
            raise GenerationBackendError(MISSING_EXTRA_MESSAGE) from error

        # Hide harmless library warnings (missing torchvision, LoRA keys absent from the text encoder), keeping progress bars.
        diffusers.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

        # Load the base model with the scribble ControlNet but no safety checker, whose weights are not downloaded.
        dtype = torch.float16 if self.device == TorchDevice.CUDA else torch.float32
        controlnet_folder = str(self._model_dir / CONTROLNET_MODEL.folder)
        try:
            controlnet = diffusers.ControlNetModel.from_pretrained(controlnet_folder, dtype=dtype)
            pipeline = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=str(self._model_dir / BASE_MODEL.folder),
                controlnet=controlnet,
                dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
            scheduler_class = getattr(diffusers, MODE_SETTINGS[self.mode].scheduler)
            pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)

            # Fast mode fuses the LCM-LoRA into the model so 4 steps give a usable image.
            if self.mode == GenerationMode.FAST:
                pipeline.load_lora_weights(str(self._model_dir / LCM_LORA_MODEL.folder), weight_name=LCM_LORA_WEIGHTS)
                pipeline.fuse_lora()

            pipeline.to(self.device)
        except Exception as error:
            message = f"Cannot load the diffusers models: {first_line(error).rstrip('.')}. Rerun scripts/download_models.py."
            raise GenerationBackendError(message) from error

        # Seed a fresh CPU generator per candidate, which Diffusers recommends for reproducibility across devices.
        def run(seed: int, **pipeline_kwargs: Any) -> np.ndarray:
            generator = torch.Generator(TorchDevice.CPU).manual_seed(seed)
            return pipeline(**pipeline_kwargs, generator=generator, output_type="np").images[0]

        return run

    def _build_pipeline_kwargs(self, request: GenerationRequest, controls: dict[str, ControlValue],
                         sketch_payload: bytes) -> dict[str, Any]:
        # Diffusers reads a NumPy image as a batch of RGB values in [0, 1], shaped (1, height, width, 3).
        conditioning = scribble_conditioning(sketch_payload)
        height, width = conditioning.shape
        conditioning_batch = np.repeat(conditioning[None, :, :, None], 3, axis=3).astype(np.float32) / 255
        pipeline_kwargs: dict[str, Any] = {
            "prompt": request.guidance.prompt, "image": conditioning_batch,
            "width": width, "height": height, "num_inference_steps": controls["steps"],
            "guidance_scale": controls["guidance_scale"], "controlnet_conditioning_scale": controls["conditioning_scale"]
        }

        # Pass a negative prompt only when one was given, since the fast mode does not support it.
        if request.guidance.negative_prompt is not None:
            pipeline_kwargs["negative_prompt"] = request.guidance.negative_prompt

        return pipeline_kwargs

    def _generate_candidates(self, n_candidates: int, base_seed: int,
                             pipeline_kwargs: dict[str, Any]) -> tuple[list[Candidate], dict[str, bytes]]:
        # Load the models on first use, so creating the backend stays fast.
        if self._runner is None:
            self._runner = self._load_runner()

        # Generate candidates one at a time to keep peak memory low on CPU.
        candidates = []
        payloads = {}
        for index in range(n_candidates):
            seed = base_seed + index
            try:
                image = self._runner(seed, **pipeline_kwargs)
            except Exception as error:
                message = f"Diffusers generation failed: {first_line(error).rstrip('.')}. Free memory and retry."
                raise GenerationBackendError(message) from error

            payload, image_height, image_width = encode_rgb_png(image)
            candidate_id = uuid.uuid4().hex
            image_ref = ImageRef(path=f"candidates/candidate-{index + 1}.png", width=image_width, height=image_height, mode="RGB",
                                 media_type="image/png", sha256=hashlib.sha256(payload).hexdigest())
            candidates.append(Candidate(id=candidate_id, index=index, image=image_ref, seed=seed))
            payloads[image_ref.path] = payload

        return candidates, payloads

    def _identify_backend(self) -> BackendIdentity:
        # Name each model with the revision its local folder was downloaded at.
        sources = self._list_model_sources()
        model_id = " + ".join(f"{source.repo_id}@{local_revision(self._model_dir / source.folder)}" for source in sources)
        return BackendIdentity(adapter="sketchloop.diffusers", adapter_version="1", execution="in_process", model_id=model_id)

    def _collect_effective_settings(self, request: GenerationRequest, controls: dict[str, ControlValue],
                            candidates: list[Candidate], resolution: str) -> EffectiveSettings:
        # Combine the controls with the mode, device, scheduler, and seeds actually used. session.json records library versions.
        dtype = "float16" if self.device == TorchDevice.CUDA else "float32"
        settings: dict[str, ControlValue] = {
            "mode": self.mode.value,
            "device": self.device.value,
            "dtype": dtype,
            "scheduler": MODE_SETTINGS[self.mode].scheduler,
            "resolution": resolution,
            "seeds": " ".join(str(candidate.seed) for candidate in candidates),
            "safety_checker": False,
        }
        return EffectiveSettings(prompt=request.guidance.prompt, negative_prompt=request.guidance.negative_prompt,
                                 controls=controls | settings)

    def generate(self, request: GenerationRequest, sketch_payload: bytes) -> GenerationOutput:
        """
        Validate the request, condition the model on the sketch, and generate one image per candidate.

        Args:
            request (GenerationRequest): Request to generate from.
            sketch_payload (bytes): Encoded bytes of the request's sketch image.

        Returns:
            GenerationOutput: Candidates and their PNG bytes.
        """
        # Refuse bytes that are not the requested sketch, so the record names the image actually used.
        if hashlib.sha256(sketch_payload).hexdigest() != request.sketch.sha256:
            raise InvalidSketchError(f"Sketch bytes do not match {request.sketch.path}. Pass the requested sketch's bytes.")

        # Validate before the slow model load, then apply the mode defaults to unset controls.
        validate_request(request, self._capabilities)
        controls = {spec.name: spec.default for spec in self._capabilities.controls} | dict(request.guidance.controls)
        pipeline_kwargs = self._build_pipeline_kwargs(request, controls, sketch_payload)

        # Draw a base seed when none is given and record it, so each candidate can be regenerated.
        base_seed = request.seed if request.seed is not None else random.randrange(2**31)
        candidates, payloads = self._generate_candidates(request.n_candidates, base_seed, pipeline_kwargs)

        # Record the backend and every setting the pipeline actually used.
        resolution = f"{pipeline_kwargs['width']}x{pipeline_kwargs['height']}"
        effective = self._collect_effective_settings(request, controls, candidates, resolution)
        result = GenerationResult(candidates=tuple(candidates), backend=self._identify_backend(), effective=effective)
        return GenerationOutput(result=result, payloads=payloads)
