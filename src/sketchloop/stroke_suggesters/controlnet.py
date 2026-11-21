import cv2
import hashlib
import numpy as np
import random
import threading
import time

from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from sketchloop.backends.diffusers import (BASE_MODEL, CONTROLNET_MODEL, MISSING_LIBRARY_MESSAGE, MODEL_SOURCES,
                                           DiffusersRunner, GenerationBackendError, PipelineRunner, TorchDevice,
                                           TorchPrecision, describe_torch_runtime, encode_rgb_png, fuse_lcm_lora,
                                           local_revision, pick_default_precision, require_model_folders, select_device)
from sketchloop.capture import InvalidSketchError
from sketchloop.domain import (BackendIdentity, ColorMode, ControlValue, EffectiveSettings, ExecutionKind, ImageRef, IssueCode,
                               Unavailable, UnsupportedConfigurationError, ValidationIssue, first_line)
from sketchloop.generation import ControlKind, ControlSpec, control_issues
from sketchloop.preprocessing import encode_grayscale_png
from sketchloop.stroke_suggesters.line_parts import (LinePartParams, count_skeleton_pixels, decode_canvas, extract_new_lines,
                                                     group_parts, list_extraction_settings)
from sketchloop.stroke_suggestion import (LINE_DRAWING_FILE, StrokeSuggester, StrokeSuggestion, StrokeSuggestionError,
                                          StrokeSuggestionOutput, StrokeSuggestionRequest, SuggestedPart, name_part_file)


# Prompts from the research note, which keep the output a clean line drawing in the person's style.
PROMPT_TEMPLATE: Final = "a simple black line drawing of {prompt}, clean thin black outlines on white paper, coloring book page, minimal"  # noqa: E501
NEUTRAL_PROMPT: Final = "a simple black line drawing, clean thin black outlines on white paper, coloring book page, minimal"
NEGATIVE_PROMPT: Final = "shading, color, gray, texture, photo, shadow, hatching"

# The CLIP text encoder reads at most 77 tokens, counting its start and end tokens, and cuts the rest.
PROMPT_TOKEN_LIMIT: Final = 77

# Canvas pixels darker than this are the person's existing strokes.
STROKE_LEVEL: Final = 128
SCHEDULER: Final = "LCMScheduler"
MODEL_LABEL: Final = "Stable Diffusion 1.5, ControlNet scribble, and LCM-LoRA"

# Counts the tokens a text takes in the pipeline's tokenizer, start and end tokens included.
TokenCounter = Callable[[str], int]


class PromptTemplate(StrEnum):
    WITH_PROMPT = "with_prompt"
    NEUTRAL = "neutral"


def build_line_drawing_prompt(prompt: str) -> tuple[str, PromptTemplate]:
    # The participant's own prompt is the only statement of what they draw, so a blank one gives a neutral drawing.
    subject = prompt.strip()
    if not subject:
        return NEUTRAL_PROMPT, PromptTemplate.NEUTRAL

    return PROMPT_TEMPLATE.format(prompt=subject), PromptTemplate.WITH_PROMPT


def count_executed_steps(steps: int, strength: float) -> int:
    # Image-to-image skips the first steps by strength, as diffusers' get_timesteps does.
    return min(int(steps * strength), steps)


def make_image_batch(gray: np.ndarray) -> np.ndarray:
    # Diffusers reads a NumPy image as a batch of RGB values in [0, 1], shaped (1, height, width, 3).
    return np.repeat(gray[None, :, :, None], 3, axis=3).astype(np.float32) / 255


class ControlNetStrokeSuggester(StrokeSuggester):

    # Saved with every suggestion, so never rename it.
    ADAPTER_NAME: Final = "sketchloop.controlnet-strokes"
    ADAPTER_VERSION: Final = "1"

    def __init__(self, models_dir: Path = Path("models"), *, device: TorchDevice | None = None,
                 precision: TorchPrecision | None = None, runner: PipelineRunner | None = None,
                 count_tokens: TokenCounter | None = None) -> None:
        self._models_dir = models_dir
        self.device = TorchDevice(device) if device else select_device()
        self.precision = TorchPrecision(precision) if precision else pick_default_precision(self.device)
        if self.precision == TorchPrecision.FLOAT16 and self.device == TorchDevice.CPU:
            message = "float16 needs a GPU, but the stroke suggester runs on the CPU. Set stroke_suggester.controlnet.precision to float32."  # noqa: E501
            raise StrokeSuggestionError(message)

        self._runner = runner
        self._count_tokens = count_tokens
        self._runtime: dict[str, ControlValue] = {}
        self._warmed_sizes: set[tuple[int, int]] = set()
        self._load_lock = threading.Lock()
        self._params = LinePartParams()

        # Research defaults, where guidance above 1 makes the negative prompt apply and strength 0.85 keeps the strokes.
        self._controls = (
            ControlSpec(name="strength", kind=ControlKind.FLOAT, minimum=0.5, maximum=1.0, default=0.85),
            ControlSpec(name="steps", kind=ControlKind.INT, minimum=1, maximum=20, default=5),
            ControlSpec(name="guidance_scale", kind=ControlKind.FLOAT, minimum=1.0, maximum=5.0, default=1.5),
            ControlSpec(name="conditioning_scale", kind=ControlKind.FLOAT, minimum=0, maximum=2, default=0.8),
        )

    @property
    def controls(self) -> tuple[ControlSpec, ...]:
        return self._controls

    def describe(self) -> str:
        steps = next(spec.default for spec in self._controls if spec.name == "steps")
        return f"Suggesting next strokes with {MODEL_LABEL} image-to-image on {self.device.upper()} in {self.precision}, running {steps} scheduled steps."  # noqa: E501

    def _load_pipeline(self) -> tuple[PipelineRunner, TokenCounter]:
        # Report a missing download before the slow library imports.
        try:
            require_model_folders(self._models_dir, MODEL_SOURCES)
        except GenerationBackendError as error:
            raise StrokeSuggestionError(str(error)) from error

        # Import the heavy libraries only when the real model is needed.
        try:
            import diffusers
            import torch
            import transformers
        except ImportError as error:
            raise StrokeSuggestionError(MISSING_LIBRARY_MESSAGE) from error

        # Hide harmless library warnings, such as LoRA keys absent from the text encoder.
        diffusers.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

        # Build a separate img2img pipeline, switch to the LCM scheduler, and fuse the LCM-LoRA, as in the research note.
        dtype = getattr(torch, self.precision)
        try:
            controlnet = diffusers.ControlNetModel.from_pretrained(str(self._models_dir / CONTROLNET_MODEL.folder), dtype=dtype)
            pipeline = diffusers.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path=str(self._models_dir / BASE_MODEL.folder),
                controlnet=controlnet,
                dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
            pipeline.scheduler = diffusers.LCMScheduler.from_config(pipeline.scheduler.config)
            fuse_lcm_lora(pipeline, self._models_dir)
            pipeline.to(self.device)
        except Exception as error:
            message = f"Cannot load the next-stroke suggestion models: {first_line(error).rstrip('.')}. Run python scripts/download_models.py and retry."  # noqa: E501
            raise StrokeSuggestionError(message) from error

        self._runtime = describe_torch_runtime(torch, self.device)
        tokenizer = pipeline.tokenizer
        return DiffusersRunner(pipeline, torch), lambda text: len(tokenizer(text).input_ids)

    def _ensure_loaded(self) -> float | None:
        # Wait for a load or warm-up in another thread, and load here only when nothing has.
        with self._load_lock:
            if self._runner is not None:
                return None

            started = time.perf_counter()
            self._runner, self._count_tokens = self._load_pipeline()
            return time.perf_counter() - started

    def load(self, width: int, height: int) -> None:
        self._ensure_loaded()

        # Run once on a blank canvas, since the first call at a new size on MPS took 2 to 12 s longer.
        with self._load_lock:
            if (width, height) in self._warmed_sizes:
                return

            blank = np.full((height, width), 255, dtype=np.uint8)
            self._run_pipeline(0, self._build_pipeline_kwargs(blank, NEUTRAL_PROMPT, self._fill_defaults({})))
            self._warmed_sizes.add((width, height))

    def _fill_defaults(self, requested: Mapping[str, ControlValue]) -> dict[str, ControlValue]:
        return {spec.name: spec.default for spec in self._controls} | dict(requested)

    def _check_controls(self, request: StrokeSuggestionRequest) -> dict[str, ControlValue]:
        # Refuse unknown or out-of-range controls, then steps and strength so low that no step would run.
        issues = list(control_issues(request.controls, self._controls, "controls"))
        controls = self._fill_defaults(request.controls)
        if not issues and count_executed_steps(controls["steps"], controls["strength"]) < 1:
            message = f"steps {controls['steps']} at strength {controls['strength']} runs no step. Raise steps or strength."
            issues.append(ValidationIssue(field="controls.steps", code=IssueCode.OUT_OF_RANGE, message=message))

        if issues:
            raise UnsupportedConfigurationError(issues)

        return controls

    def _build_pipeline_kwargs(self, canvas: np.ndarray, prompt: str, controls: Mapping[str, ControlValue]) -> dict[str, Any]:
        # Start from the canvas, black on white, and condition on its inverse, as the scribble model expects.
        height, width = canvas.shape
        pipeline_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": make_image_batch(canvas),
            "control_image": make_image_batch(255 - canvas),
            "strength": controls["strength"],
            "num_inference_steps": controls["steps"],
            "guidance_scale": controls["guidance_scale"],
            "controlnet_conditioning_scale": controls["conditioning_scale"],
            "width": width,
            "height": height,
        }

        # The pipeline ignores a negative prompt at guidance 1 or below, so pass one only when it applies.
        if controls["guidance_scale"] > 1:
            pipeline_kwargs["negative_prompt"] = NEGATIVE_PROMPT

        return pipeline_kwargs

    def _run_pipeline(self, seed: int, pipeline_kwargs: Mapping[str, Any]) -> np.ndarray:
        try:
            return self._runner(seed, **pipeline_kwargs)
        except Exception as error:
            message = f"Next-stroke suggestion failed: {first_line(error).rstrip('.')}. Free memory and retry."
            raise StrokeSuggestionError(message) from error

    def _describe_prompt(self, prompt: str, template: PromptTemplate) -> dict[str, ControlValue | Unavailable]:
        # Count tokens with the pipeline's tokenizer, and record the count as unavailable when none was loaded.
        if self._count_tokens is None:
            missing = Unavailable("no tokenizer was loaded with the injected pipeline")
            return {"prompt_template": template.value, "prompt_tokens": missing, "prompt_truncated": missing}

        tokens = self._count_tokens(prompt)
        return {"prompt_template": template.value, "prompt_tokens": tokens, "prompt_truncated": tokens > PROMPT_TOKEN_LIMIT}

    def _extract_parts(self, line_drawing: np.ndarray, canvas: np.ndarray,
                       max_parts: int) -> tuple[list[SuggestedPart], dict[str, bytes]]:
        # Keep the drawing's new lines away from the existing strokes and group them into parts, largest first.
        gray = cv2.cvtColor((np.clip(line_drawing, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
        existing = np.where(canvas < STROKE_LEVEL, 255, 0).astype(np.uint8)
        masks = group_parts(extract_new_lines(gray, existing, self._params), self._params, max_parts)

        # Encode each part's mask and measure its box, size, and skeleton.
        parts = []
        payloads = {}
        for index, mask in enumerate(masks):
            path = name_part_file(index)
            mask_ref, payloads[path] = encode_grayscale_png(mask, path, "Part mask cannot be encoded as PNG. Retry.")
            box = tuple(int(value) for value in cv2.boundingRect(mask))
            parts.append(SuggestedPart(index=index, mask=mask_ref, box=box, pixel_count=cv2.countNonZero(mask),
                                       skeleton_pixels=count_skeleton_pixels(mask)))

        return parts, payloads

    def _identify_suggester(self) -> BackendIdentity:
        # Name each model with the revision its local folder was downloaded at.
        model_id = " + ".join(f"{source.repo_id}@{local_revision(self._models_dir / source.folder)}" for source in MODEL_SOURCES)
        return BackendIdentity(adapter=self.ADAPTER_NAME, adapter_version=self.ADAPTER_VERSION,
                               execution=ExecutionKind.IN_PROCESS, model_id=model_id)

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        # Validate before the slow model load, and refuse bytes that are not the requested canvas.
        controls = self._check_controls(request)
        if hashlib.sha256(canvas_payload).hexdigest() != request.canvas.sha256:
            raise InvalidSketchError(f"Canvas bytes do not match {request.canvas.path}. Pass the requested canvas's bytes.")

        canvas = decode_canvas(request, canvas_payload)
        load_seconds = self._ensure_loaded()

        # Draw the line drawing from the canvas, with a recorded seed.
        prompt, template = build_line_drawing_prompt(request.prompt)
        pipeline_kwargs = self._build_pipeline_kwargs(canvas, prompt, controls)
        seed = request.seed if request.seed is not None else random.randrange(2**31)
        started = time.perf_counter()
        line_drawing = self._run_pipeline(seed, pipeline_kwargs)
        generation_seconds = time.perf_counter() - started

        # A drawing of another size would give masks that do not overlay the canvas.
        if line_drawing.shape[:2] != canvas.shape:
            raise StrokeSuggestionError(f"Next-stroke suggestion returned {line_drawing.shape[1]}x{line_drawing.shape[0]}, not the canvas size. Retry.")  # noqa: E501

        # Extract the parts and keep the model's whole output, which cannot be regenerated exactly elsewhere.
        started = time.perf_counter()
        parts, payloads = self._extract_parts(line_drawing, canvas, request.max_parts)
        line_drawing_payload, height, width = encode_rgb_png(line_drawing)
        line_drawing_ref = ImageRef(path=LINE_DRAWING_FILE, width=width, height=height, mode=ColorMode.RGB,
                                    media_type="image/png", sha256=hashlib.sha256(line_drawing_payload).hexdigest())
        payloads[LINE_DRAWING_FILE] = line_drawing_payload
        extraction_seconds = time.perf_counter() - started

        # Record the exact text and every setting the pipeline and extraction used.
        settings: dict[str, ControlValue | Unavailable] = {
            **controls,
            **self._describe_prompt(prompt, template),
            "executed_steps": count_executed_steps(controls["steps"], controls["strength"]),
            "device": self.device.value,
            "dtype": self.precision.value,
            "scheduler": SCHEDULER,
            "lcm_lora_fused": True,
            "width": width,
            "height": height,
            **self._runtime,
            **list_extraction_settings(self._params),
        }
        effective = EffectiveSettings(prompt=prompt, negative_prompt=pipeline_kwargs.get("negative_prompt"), controls=settings)
        suggestion = StrokeSuggestion(suggester=self._identify_suggester(), effective=effective, seed=seed,
                                      line_drawing=line_drawing_ref, parts=tuple(parts),
                                      timings_seconds={"generation": generation_seconds, "extraction": extraction_seconds})
        return StrokeSuggestionOutput(suggestion=suggestion, payloads=payloads, load_seconds=load_seconds)
