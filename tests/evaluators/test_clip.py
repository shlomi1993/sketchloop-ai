"""
Tests of the CLIP evaluator with a stub embedder, so no weights are loaded, plus one real ViT-B/32 run.
"""

import math
import numpy as np
import pytest

from collections.abc import Sequence
from pathlib import Path

from sketchloop.backends.diffusers import ModelSource
from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import GenerationRequest, Guidance, ImageRef
from sketchloop.evaluation import EvaluationError
from sketchloop.evaluators.clip import CLIP_B32_MODEL, CLIP_L14_MODEL, ClipEvaluator, compute_cosine_similarities
from sketchloop.generation import GenerationOutput


ROOT = Path(__file__).resolve().parents[2]
SKETCH = ImageRef(path="sketch.png", width=8, height=8, mode="L", media_type="image/png", sha256="a" * 64)


def generate_candidates(count: int) -> tuple[GenerationRequest, GenerationOutput]:
    """
    Generate fake candidates for a chair prompt, to score without a real generator.
    """
    request = GenerationRequest(sketch=SKETCH, guidance=Guidance(prompt="modern chair"), n_candidates=count, seed=0)
    return request, FakeGenerator().generate(request, b"unused by the fake")


class StubEmbedder:
    """
    Test-only embedder that returns fixed embeddings and remembers what it was asked to embed.
    """

    def __init__(self, image_embeddings: list[list[float]], text_embedding: list[float]) -> None:
        self.image_embeddings = np.array(image_embeddings, dtype=np.float32)
        self.text_embedding = np.array(text_embedding, dtype=np.float32)
        self.calls: list[tuple[int, str]] = []

    def __call__(self, image_payloads: Sequence[bytes], prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Record the call and return the fixed embeddings.
        """
        self.calls.append((len(image_payloads), prompt))
        return self.image_embeddings, self.text_embedding


def test_cosine_similarities() -> None:
    """
    Similarity ignores vector length and ranges from -1 for opposite to 1 for the same direction.
    """
    similarities = compute_cosine_similarities(np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]]), np.array([5.0, 0.0]))
    assert similarities == pytest.approx([1.0, 0.0, -1.0]), f"Expected 1, 0, and -1, got {similarities}"


def test_clip_scores_each_candidate_against_the_prompt() -> None:
    """
    Each candidate gets its own finite score under its ID, from one embedding call with every image and the prompt.
    """
    request, output = generate_candidates(2)
    embedder = StubEmbedder([[1.0, 0.0], [0.6, 0.8]], [1.0, 0.0])
    scores = ClipEvaluator(CLIP_B32_MODEL, embedder=embedder).evaluate(output.result.candidates, output.payloads, request)

    # Scores follow candidate order and are keyed by candidate ID.
    candidate_ids = [candidate.id for candidate in output.result.candidates]
    assert list(scores) == candidate_ids, f"Expected scores keyed by {candidate_ids}, got {list(scores)}"
    assert list(scores.values()) == pytest.approx([1.0, 0.6]), f"Expected cosine scores 1.0 and 0.6, got {scores}"
    assert all(math.isfinite(score) and isinstance(score, float) for score in scores.values()), f"Bad scores {scores}"

    # The embedder saw both images and the request's prompt once.
    assert embedder.calls == [(2, "modern chair")], f"Expected one call with two images, got {embedder.calls}"


def test_clip_refuses_non_finite_scores() -> None:
    """
    A zero embedding, whose similarity is undefined, fails with a typed error instead of reaching the record.
    """
    request, output = generate_candidates(1)
    evaluator = ClipEvaluator(CLIP_B32_MODEL, embedder=StubEmbedder([[0.0, 0.0]], [1.0, 0.0]))
    with pytest.raises(EvaluationError, match="non-finite"):
        evaluator.evaluate(output.result.candidates, output.payloads, request)


def test_clip_loads_lazily_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Creating the evaluator loads nothing, and the model loads on the first evaluation only.
    """
    loads: list[Path] = []

    def load_stub(evaluator: ClipEvaluator) -> StubEmbedder:
        """
        Count the load and return a stub embedder instead of the model.
        """
        loads.append(evaluator.model_folder)
        return StubEmbedder([[1.0, 0.0]], [1.0, 0.0])

    monkeypatch.setattr(ClipEvaluator, "_load_embedder", load_stub)
    evaluator = ClipEvaluator(CLIP_L14_MODEL, tmp_path)
    assert loads == [], "Creating the evaluator must not load the model"

    # Two evaluations load the model from the models folder once.
    request, output = generate_candidates(1)
    evaluator.evaluate(output.result.candidates, output.payloads, request)
    evaluator.evaluate(output.result.candidates, output.payloads, request)
    assert loads == [tmp_path / CLIP_L14_MODEL.folder], f"Expected one load from the L/14 folder, got {loads}"


def test_clip_load_ahead_of_scoring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    load() loads the model once, and a later load or evaluation does not load it again.
    """
    loads: list[Path] = []

    def load_stub(evaluator: ClipEvaluator) -> StubEmbedder:
        """
        Count the load and return a stub embedder instead of the model.
        """
        loads.append(evaluator.model_folder)
        return StubEmbedder([[1.0, 0.0]], [1.0, 0.0])

    monkeypatch.setattr(ClipEvaluator, "_load_embedder", load_stub)
    evaluator = ClipEvaluator(CLIP_B32_MODEL, tmp_path)
    evaluator.load()
    assert loads == [tmp_path / CLIP_B32_MODEL.folder], f"load() must load the model once, got {loads}"

    # A second load and an evaluation reuse the loaded model.
    evaluator.load()
    request, output = generate_candidates(1)
    evaluator.evaluate(output.result.candidates, output.payloads, request)
    assert len(loads) == 1, f"The model must load only once, got {loads}"


def test_clip_missing_download(tmp_path: Path) -> None:
    """
    A missing model folder fails with one line that names the folder and the download script.
    """
    request, output = generate_candidates(1)
    with pytest.raises(EvaluationError) as raised:
        ClipEvaluator(CLIP_B32_MODEL, tmp_path).evaluate(output.result.candidates, output.payloads, request)

    # The message is one line naming the missing folder and how to fetch it.
    message = str(raised.value)
    assert "\n" not in message, f"Expected a one-line message, got {message!r}"
    assert CLIP_B32_MODEL.folder in message and "download_models.py" in message, f"Unhelpful error {message!r}"


@pytest.mark.parametrize("model", [CLIP_B32_MODEL, CLIP_L14_MODEL], ids=["b32", "l14"])
def test_clip_describes_its_scores(model: ModelSource) -> None:
    """
    The evaluator names its repository, pinned revision, and what the score means, for the round record.
    """
    details = ClipEvaluator(model).describe_scores()
    assert details["model"] == model.repo_id and details["revision"] == model.revision, f"Bad identity {details}"
    assert "cosine similarity" in details["score"] and "not design quality" in details["score"], f"Bad meaning {details}"


def test_clip_models_are_pinned_safetensors() -> None:
    """
    Both models are pinned to a commit, and the large one downloads only its safetensors weights.
    """
    for model in (CLIP_B32_MODEL, CLIP_L14_MODEL):
        assert len(model.revision) == 40, f"{model.repo_id} must be pinned to a commit SHA, got {model.revision}"

    # ViT-B/32 has only a pickled .bin, which the download script converts, and ViT-L/14 has safetensors upstream.
    assert "model.safetensors" in CLIP_L14_MODEL.allow_patterns, f"Bad L/14 patterns {CLIP_L14_MODEL.allow_patterns}"
    assert "pytorch_model.bin" in CLIP_B32_MODEL.allow_patterns, f"Bad B/32 patterns {CLIP_B32_MODEL.allow_patterns}"


@pytest.mark.model
@pytest.mark.skipif(not (ROOT / "models" / CLIP_B32_MODEL.folder).is_dir(), reason="ViT-B/32 weights are not downloaded")
def test_clip_b32_real_model() -> None:
    """
    The real ViT-B/32 scores two fake candidates with finite similarities in the cosine range.
    """
    request, output = generate_candidates(2)
    evaluator = ClipEvaluator(CLIP_B32_MODEL, ROOT / "models")
    scores = evaluator.evaluate(output.result.candidates, output.payloads, request)
    assert len(scores) == 2 and all(-1.0 <= score <= 1.0 for score in scores.values()), f"Bad real scores {scores}"
