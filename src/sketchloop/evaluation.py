from collections.abc import Mapping, Sequence
from typing import Protocol

from sketchloop.domain import Candidate, GenerationRequest


class Evaluator(Protocol):
    """
    Interface for optional candidate scoring, kept separate from generation and human selection.
    """

    def evaluate(self, candidates: Sequence[Candidate], payloads: Mapping[str, bytes],
                 request: GenerationRequest) -> Mapping[str, float]:
        """
        Score candidates without changing them.

        Args:
            candidates (Sequence[Candidate]): Candidates to score.
            payloads (Mapping[str, bytes]): Image bytes by candidate image path.
            request (GenerationRequest): Request the candidates were generated from.

        Returns:
            Mapping[str, float]: Score by candidate ID, empty when the evaluator gives no scores.
        """
        ...


class NoOpEvaluator:
    """
    Evaluator that gives no scores, used until a real scorer such as CLIP is added.
    """

    def evaluate(self, candidates: Sequence[Candidate], payloads: Mapping[str, bytes],
                 request: GenerationRequest) -> Mapping[str, float]:
        """
        Return no scores.
        """
        return {}
