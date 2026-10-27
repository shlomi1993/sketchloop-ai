import pytest

from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import load_sketch_file
from sketchloop.domain import Guidance, SketchSource, UnsupportedConfigurationError
from sketchloop.experiments import BenchmarkPlan, MemoryStore, compute_statistics, summarize_benchmark
from sketchloop.generation import Generator
from sketchloop.session import SketchSession


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


# The p95 of five values lies 80% of the way from the fourth to the fifth sorted value: 4 + 0.8 * (10 - 4).
@pytest.mark.parametrize(("durations", "expected"), [
    ([2.0], (1, 2.0, 2.0, 2.0, 2.0)),
    ([4.0, 1.0, 3.0, 10.0, 2.0], (5, 3.0, 8.8, 1.0, 10.0))
], ids=["one-value", "five-values"])
def test_compute_statistics(durations: list[float], expected: tuple[int, float, float, float, float]) -> None:
    """
    One value is its own median and p95, and five values give the inclusive interpolated p95.
    """
    stats = compute_statistics(durations)

    # Count, median, p95, minimum, and maximum match.
    values = (stats.count, stats.median_seconds, stats.p95_seconds, stats.min_seconds, stats.max_seconds)
    assert values == pytest.approx(expected), f"Expected count, median, p95, min, max {expected}, got {values}"


@pytest.mark.usefixtures("counting_clock")
def test_summarize_benchmark(first_load_generator: Generator) -> None:
    """
    The summary leaves out warm-up and failed rounds, gives a warm-up-only load no row, adds the total, and lists varied settings.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(first_load_generator, MemoryStore())

    # Round 1 loads the model, round 3 fails, and round 4 changes the steps.
    for number, controls, candidates in ((1, {}, 1), (2, {}, 1), (3, {}, 99), (4, {"steps": 8}, 1)):
        guidance = Guidance(prompt="chair", controls=controls)
        try:
            session.run_round(raw_sketch, raw_payload, guidance, source=SketchSource.FILE, n_candidates=candidates, seed=0)
        except UnsupportedConfigurationError:
            assert number == 3, f"Only round 3 may fail, but round {number} did"
    summary = summarize_benchmark(session.records, BenchmarkPlan(warmup_rounds=1, measured_rounds=3))

    # Round 1 is warm-up, round 3 failed, and rounds 2 and 4 are measured.
    roles = (summary.warmup_rounds, summary.measured_rounds, summary.failed_rounds)
    assert roles == ((1,), (2, 4), (3,)), f"Unexpected round roles {roles}"

    # Only the warm-up loaded the model, so model_load has no row, and the total follows the stages.
    stages = list(summary.stages)
    assert stages == ["preprocessing", "generation", "evaluation", "saving", "total"], f"Unexpected stages {stages}"
    assert all(stats.count == 2 for stats in summary.stages.values()), "Every row must cover both measured rounds"

    # The steps changed between measured rounds, and seeds are never reported as varying.
    assert summary.varying_settings == ("steps",), f"Expected steps to vary, got {summary.varying_settings}"


def test_summarize_benchmark_without_measured_round() -> None:
    """
    A benchmark whose only measured round failed has no measured rounds and no statistics.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), MemoryStore())
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE)
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE, n_candidates=99)
    summary = summarize_benchmark(session.records, BenchmarkPlan(warmup_rounds=1, measured_rounds=1))

    # Round 1 is warm-up, round 2 failed, and nothing is measured.
    roles = (summary.warmup_rounds, summary.measured_rounds, summary.failed_rounds, dict(summary.stages))
    assert roles == ((1,), (), (2,), {}), f"Expected warm-up 1, failed 2, and no statistics, got {roles}"
