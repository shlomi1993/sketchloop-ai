"""
Style prompt suggestions for the page, from a fixed list or from a local model that names the sketched object.
"""

import cv2
import io
import numpy as np
import threading

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from sketchloop.backends.diffusers import ModelSource, TorchDevice, select_device
from sketchloop.domain import SketchLoopError, first_line


# Pinned in the prompt suggestion research note, with only the configs, tokenizer, and safetensors weights.
FLORENCE_MODEL: Final = ModelSource(
    folder="florence-2-base",
    repo_id="florence-community/Florence-2-base",
    revision="00921df66db728a9ceb750f5eca43e5c203a2051",
    allow_patterns=("*.json", "*.txt", "model.safetensors"),
)
CAPTION_TASK: Final = "<CAPTION>"
NEUTRAL_SUBJECT: Final = "the sketched object"
SUBJECT_MAX_WORDS: Final = 6
SIGNATURE_SIZE: Final = 64  # Side in pixels of the downscaled sketch compared between frames
SKETCH_CHANGE_THRESHOLD: Final = 4.0  # Mean gray-level difference, of 255, that counts as a changed drawing

# Each style becomes one suggestion for the named object, short enough for the 77-token prompt limit of SD 1.5.
STYLE_TEMPLATES: Final = (
    "{subject}, product design render, studio lighting, white background",
    "{subject}, Scandinavian minimalist design, light oak wood, soft daylight",
    "{subject}, industrial design, brushed steel and black leather, dramatic lighting",
    "{subject}, mid-century modern, walnut and brass, warm interior photo",
    "{subject}, concept art, watercolor and ink, muted colors",
)

# Caption openings that describe the medium rather than the object, longest first.
FRAMING_PREFIXES: Final = (
    "a black and white drawing of",
    "a black and white sketch of",
    "a black and white image of",
    "a pencil drawing of",
    "a pencil sketch of",
    "a line drawing of",
    "a drawing of",
    "a sketch of",
    "an image of",
    "the image shows",
)


class SuggestionError(SketchLoopError, RuntimeError):
    """
    The suggestion model cannot be loaded or fails while captioning.
    """


def extract_subject(caption: str) -> str:
    """
    Turn a caption such as "A drawing of a lounge chair." into the object it names, "a lounge chair".
    """
    subject = caption.strip().lower().rstrip(".")
    for prefix in FRAMING_PREFIXES:
        if subject.startswith(prefix):
            subject = subject[len(prefix):].strip()
            break

    return " ".join(subject.split()[:SUBJECT_MAX_WORDS]) or NEUTRAL_SUBJECT


def fill_templates(subject: str) -> tuple[str, ...]:
    """
    Build one style prompt per template for the given object.
    """
    return tuple(template.format(subject=subject) for template in STYLE_TEMPLATES)


def compute_sketch_signature(sketch_payload: bytes) -> np.ndarray:
    """
    Shrink an encoded sketch to a small grayscale image, so frames can be compared without camera noise mattering.
    """
    gray = cv2.imdecode(np.frombuffer(sketch_payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(gray, (SIGNATURE_SIZE, SIGNATURE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def has_sketch_changed(previous: np.ndarray | None, current: np.ndarray) -> bool:
    """
    Tell whether the drawing changed enough since the previous signature, which a first sketch always has.
    """
    return previous is None or float(np.abs(previous - current).mean()) > SKETCH_CHANGE_THRESHOLD


class PromptSuggester(ABC):
    """
    Suggests style prompts for a sketch, separately from generation, evaluation, and selection.
    """

    @abstractmethod
    def suggest(self, sketch_payload: bytes) -> tuple[str, ...]:
        """
        Suggest style prompts for the sketch the next round would use.

        Args:
            sketch_payload (bytes): Encoded sketch, preprocessed unless the person chose the raw sketch.

        Returns:
            tuple[str, ...]: Suggested prompts, best first.
        """
        ...

    def warm_up(self) -> None:
        """
        Load any model ahead of the first suggestion, which suggesters without a model skip.
        """


class TemplateSuggester(PromptSuggester):
    """
    Suggests the fixed style templates without looking at the sketch, so suggestions work with no model.
    """

    def suggest(self, sketch_payload: bytes) -> tuple[str, ...]:
        """
        Fill the style templates with a neutral object name.
        """
        return fill_templates(NEUTRAL_SUBJECT)


class FlorenceSuggester(PromptSuggester):
    """
    Names the sketched object with a local Florence-2 caption and fills the style templates with it.
    """

    def __init__(self, models_dir: Path = Path("models"), *, captioner: Callable[[bytes], str] | None = None) -> None:
        self._model_folder = models_dir / FLORENCE_MODEL.folder
        self._captioner = captioner
        self._load_lock = threading.Lock()

    def _load_captioner(self) -> Callable[[bytes], str]:
        # Report a missing download before the slow library imports.
        if not self._model_folder.is_dir():
            message = f"Model folder {self._model_folder.as_posix()} is missing. Run python scripts/download_models.py and retry."
            raise SuggestionError(message)

        # Import the heavy libraries only when the model is needed.
        import torch

        from PIL import Image
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        # Load the model in float16 on a GPU and float32 on the CPU, like the generator.
        device = select_device()
        dtype = torch.float32 if device == TorchDevice.CPU else torch.float16
        try:
            processor = AutoProcessor.from_pretrained(str(self._model_folder))
            model = Florence2ForConditionalGeneration.from_pretrained(str(self._model_folder), dtype=dtype).to(device).eval()
        except Exception as error:
            raise SuggestionError(f"Cannot load the suggestion model: {first_line(error).rstrip('.')}.") from error

        def caption(sketch_payload: bytes) -> str:
            # Caption greedily, so the same sketch always gets the same suggestions.
            image = Image.open(io.BytesIO(sketch_payload)).convert("RGB")
            inputs: Any = processor(text=CAPTION_TASK, images=image, return_tensors="pt").to(device, dtype)
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=40, num_beams=3, do_sample=False)
            text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            return processor.post_process_generation(text, task=CAPTION_TASK, image_size=image.size)[CAPTION_TASK]

        return caption

    def warm_up(self) -> None:
        """
        Load the captioning model and run it once now, so the first suggestion does not wait for either.
        """
        with self._load_lock:
            if self._captioner is None:
                captioner = self._load_captioner()

                # Caption a blank image once, since the first run on a GPU is several seconds slower than the rest.
                captioner(cv2.imencode(".png", np.full((64, 64, 3), 255, dtype=np.uint8))[1].tobytes())
                self._captioner = captioner

    def suggest(self, sketch_payload: bytes) -> tuple[str, ...]:
        """
        Caption the sketch, loading the model on first use, and fill the style templates with the object it names.
        """
        self.warm_up()
        return fill_templates(extract_subject(self._captioner(sketch_payload)))
