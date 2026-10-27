import statistics

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from sketchloop.domain import Unavailable, require
from sketchloop.experiments.records import BenchmarkPlan, RoundRecord, RoundStage, RoundStatus


TOTAL_KEY: Final = "total"

# The 19th of the 20-quantile cut points is the 95th percentile.
P95_QUANTILES: Final = 20
P95_INDEX: Final = 18


@dataclass(frozen=True, slots=True, kw_only=True)
class StageStatistics:
    """
    Distribution of one stage's durations over the measured rounds that timed it.
    """
    count: int  # Measured rounds that timed this stage
    median_seconds: float  # Median duration
    p95_seconds: float  # 95th percentile, interpolated between measured rounds
    min_seconds: float  # Shortest duration
    max_seconds: float  # Longest duration


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkSummary:
    """
    Statistics of a benchmark session, recomputed from its saved rounds.
    """
    plan: BenchmarkPlan  # The plan the session recorded
    warmup_rounds: tuple[int, ...]  # Round numbers left out as warm-up
    measured_rounds: tuple[int, ...]  # Completed round numbers the statistics cover
    failed_rounds: tuple[int, ...]  # Round numbers that failed, left out of the statistics
    stages: Mapping[str, StageStatistics]  # By stage name in RoundStage order, then "total"
    varying_settings: tuple[str, ...]  # Effective control names, other than seeds, that differ across measured rounds

    def __post_init__(self) -> None:
        # Copy into a read-only view so the statistics cannot change after the fact.
        object.__setattr__(self, "stages", MappingProxyType(dict(self.stages)))


def compute_statistics(durations_seconds: Sequence[float]) -> StageStatistics:
    """
    Compute the median, inclusive 95th percentile, minimum, and maximum of one or more durations.

    Args:
        durations_seconds (Sequence[float]): Durations in seconds, at least one.

    Returns:
        StageStatistics: The distribution, where a single value is its own 95th percentile.
    """
    # Python 3.11 and 3.12 refuse quantiles of one value, which is its own 95th percentile.
    require(len(durations_seconds) >= 1, "Statistics need at least one duration.")
    values = list(durations_seconds)
    p95 = statistics.quantiles(values, n=P95_QUANTILES, method="inclusive")[P95_INDEX] if len(values) > 1 else values[0]
    return StageStatistics(count=len(values), median_seconds=statistics.median(values), p95_seconds=p95,
                           min_seconds=min(values), max_seconds=max(values))


def list_varying_settings(records: Sequence[RoundRecord]) -> tuple[str, ...]:
    """
    Name the effective controls, other than seeds, whose values differ across completed rounds.
    """
    controls = [record.iteration.result.effective.controls for record in records]
    names = sorted({name for round_controls in controls for name in round_controls} - {"seeds"})
    return tuple(name for name in names if len({round_controls.get(name) for round_controls in controls}) > 1)


def summarize_benchmark(records: Sequence[RoundRecord], plan: BenchmarkPlan) -> BenchmarkSummary:
    """
    Summarize a benchmark session's rounds, leaving out warm-up and failed rounds.

    Args:
        records (Sequence[RoundRecord]): The session's saved rounds, complete or failed.
        plan (BenchmarkPlan): The plan the session recorded.

    Returns:
        BenchmarkSummary: Round numbers by role, per-stage and total statistics, and settings that varied.
    """
    # Rounds up to the warm-up count are warm-up, and later complete rounds are measured.
    complete = [record for record in records if record.status == RoundStatus.COMPLETE]
    warmup = tuple(record.number for record in complete if record.number <= plan.warmup_rounds)
    measured = [record for record in complete if record.number > plan.warmup_rounds]
    failed = tuple(record.number for record in records if record.status == RoundStatus.FAILED)

    # A stage gets statistics only from the measured rounds that timed it, and the total from recorded totals.
    stages: dict[str, StageStatistics] = {}
    for stage in RoundStage:
        durations = [record.timings_seconds[stage] for record in measured if stage in record.timings_seconds]
        if durations:
            stages[stage] = compute_statistics(durations)

    totals = [record.total_seconds for record in measured if not isinstance(record.total_seconds, Unavailable)]
    if totals:
        stages[TOTAL_KEY] = compute_statistics(totals)

    return BenchmarkSummary(plan=plan, warmup_rounds=warmup, measured_rounds=tuple(record.number for record in measured),
                        failed_rounds=failed, stages=stages, varying_settings=list_varying_settings(measured))
