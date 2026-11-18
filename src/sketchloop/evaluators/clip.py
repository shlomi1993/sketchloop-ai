"""
CLIP evaluator that scores each candidate by how close its image is to the prompt in CLIP's shared embedding space.
"""

import io
import math
import numpy as np
import threading

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeAlias

from sketchloop.backends.diffusers import ModelSource, select_device
from sketchloop.domain import Candidate, GenerationRequest, first_line
from sketchloop.evaluation import EvaluationError, Evaluator


# Pinned in the CLIP research note. ViT-B/32 has only pickled weights, which scripts/download_models.py converts once.
CLIP_B32_MODEL: Final = ModelSource(
    folder="clip-vit-base-patch32",
    repo_id="openai/clip-vit-base-patch32",
    revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    allow_patterns=("*.json", "*.txt", "pytorch_model.bin"),
)
CLIP_L14_MODEL: Final = ModelSource(
    folder="clip-vit-large-patch14",
    repo_id="openai/clip-vit-large-patch14",
    revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
    allow_patterns=("*.json", "*.txt", "model.safetensors"),
)
SCORE_MEANING: Final = "cosine similarity of the candidate image and the prompt cut to 77 tokens, from -1 to 1, a research signal and not design quality"  # noqa: E501

# Embeds candidate images and one prompt, returning one image embedding per row and the prompt's embedding.
Embedder: TypeAlias = Callable[[Sequence[bytes], str], tuple[np.ndarray, np.ndarray]]


def compute_cosine_similarities(image_embeddings: np.ndarray, text_embedding: np.ndarray) -> list[float]:
    """
    Compute the cosine similarity of each image embedding with the text embedding.

    Args:
        image_embeddings (np.ndarray): One image embedding per row.
        text_embedding (np.ndarray): Embedding of the prompt.

    Returns:
        list[float]: Similarity per image, in row order, NaN when an embedding has zero length.
    """
    norms = np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(text_embedding)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarities = image_embeddings @ text_embedding / norms
    return [float(similarity) for similarity in similarities]


class ClipEvaluator(Evaluator):
    """
    Evaluator that scores candidates by their CLIP similarity to the prompt, loading the model on first use.
    """

    def __init__(self, model: ModelSource, models_dir: Path = Path("models"), *, embedder: Embedder | None = None) -> None:
        self.model = model
        self.model_folder = models_dir / model.folder
        self._embedder = embedder
        self._load_lock = threading.Lock()

    def _load_embedder(self) -> Embedder:
        # Report a missing download before the slow library imports.
        if not self.model_folder.is_dir():
            message = f"Model folder {self.model_folder.as_posix()} is missing. Run python scripts/download_models.py from the folder with your settings file and retry."  # noqa: E501
            raise EvaluationError(message)

        # Import the heavy libraries only when the model is needed.
        import torch

        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        # Load only safetensors weights, in float32 on the generator's device, from the local folder with no network.
        device = select_device()
        try:
            processor = CLIPProcessor.from_pretrained(str(self.model_folder))
            model = CLIPModel.from_pretrained(str(self.model_folder), dtype=torch.float32, use_safetensors=True).to(device).eval()
        except Exception as error:
            message = f"Cannot load the CLIP model: {first_line(error).rstrip('.')}. Rerun scripts/download_models.py."
            raise EvaluationError(message) from error

        def embed(image_payloads: Sequence[bytes], prompt: str) -> tuple[np.ndarray, np.ndarray]:
            # Embed every candidate and the prompt in one pass, cutting the prompt to CLIP's 77-token limit.
            images = [Image.open(io.BytesIO(payload)).convert("RGB") for payload in image_payloads]
            inputs: Any = processor(text=[prompt], images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            return outputs.image_embeds.float().cpu().numpy(), outputs.text_embeds[0].float().cpu().numpy()

        return embed

    def evaluate(self, candidates: Sequence[Candidate], payloads: Mapping[str, bytes],
                 request: GenerationRequest) -> Mapping[str, float]:
        """
        Score each candidate by the cosine similarity of its image and the request's prompt.

        Args:
            candidates (Sequence[Candidate]): Candidates to score.
            payloads (Mapping[str, bytes]): Image bytes by candidate image path.
            request (GenerationRequest): Request whose prompt the images are compared with.

        Returns:
            Mapping[str, float]: Similarity from -1 to 1 by candidate ID.
        """
        # Load the model on first use only, once even when the page scores from several threads.
        with self._load_lock:
            if self._embedder is None:
                self._embedder = self._load_embedder()

        image_payloads = [payloads[candidate.image.path] for candidate in candidates]
        image_embeddings, text_embedding = self._embedder(image_payloads, request.guidance.prompt)
        similarities = compute_cosine_similarities(image_embeddings, text_embedding)

        # Refuse an undefined similarity here, since a round record accepts only finite scores.
        if not all(math.isfinite(similarity) for similarity in similarities):
            raise EvaluationError(f"CLIP gave a non-finite score {similarities}. Check the candidate images and retry.")

        return dict(zip((candidate.id for candidate in candidates), similarities, strict=True))

    def describe_scores(self) -> Mapping[str, str]:
        """
        Name the CLIP model and revision that produced the scores and what a score means.
        """
        return {"model": self.model.repo_id, "revision": self.model.revision, "score": SCORE_MEANING}
