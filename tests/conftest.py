import pytest

from collections.abc import Mapping, Sequence
from pathlib import Path

from sketchloop.domain import Candidate, GenerationRequest
from sketchloop.evaluation import Evaluator


class ConstantEvaluator:
    """
    Evaluator that gives every candidate a score of 0.5.
    """

    def evaluate(self, candidates: Sequence[Candidate], payloads: Mapping[str, bytes],
                 request: GenerationRequest) -> Mapping[str, float]:
        """
        Score every candidate 0.5.
        """
        return {candidate.id: 0.5 for candidate in candidates}


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Run the test from its empty tmp_path, so a developer's own sketchloop.toml cannot change the settings it sees.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def constant_evaluator() -> Evaluator:
    """
    Provide an evaluator that scores every candidate 0.5, recorded under the class name ConstantEvaluator.
    """
    return ConstantEvaluator()
