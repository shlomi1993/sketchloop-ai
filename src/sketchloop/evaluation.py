"""
Evaluator interface for optional candidate scoring, kept separate from generation and selection.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from sketchloop.domain import Candidate, GenerationRequest, SketchLoopError


class EvaluationError(SketchLoopError, RuntimeError):
    """
    An evaluator cannot load its model or fails while scoring.
    """


class Evaluator(ABC):
    """
    Base class for optional candidate scoring, kept separate from generation and human selection.
    """

    @abstractmethod
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

    def describe_scores(self) -> Mapping[str, str]:
        """
        Say which model produced the scores and what they measure, for the round record, which evaluators without scores skip.
        """
        return {}


class NoOpEvaluator(Evaluator):
    """
    Evaluator that gives no scores, used until a real scorer such as CLIP is added.
    """

    def evaluate(self, candidates: Sequence[Candidate], payloads: Mapping[str, bytes],
                 request: GenerationRequest) -> Mapping[str, float]:
        """
        Return no scores.
        """
        return {}
