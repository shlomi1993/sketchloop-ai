"""
Tests of the benchmark subcommand with the fake backend.
"""

import pytest
import re

from pathlib import Path

from rich.console import Console

from sketchloop.backends.diffusers import TorchPrecision
from sketchloop.backends.fake import FakeGenerator
from sketchloop.cli import main
from sketchloop.cli.benchmark import print_benchmark_conditions
from sketchloop.components import CAPTURE_SOURCES
from sketchloop.experiments import BenchmarkPlan, load_session, summarize_benchmark


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


def list_round_seeds(runs_dir: Path) -> list[object]:
    """
    Return the first candidate seed of every round in the only session under the runs folder.
    """
    rounds = load_session(next(runs_dir.iterdir())).rounds
    return [record.iteration.result.candidates[0].seed for record in rounds]


def test_cli_benchmark_reports_statistics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A benchmark saves warm-up and measured rounds without a selection and prints the conditions, statistics, and caveats.
    """
    exit_code = main(["benchmark", str(EXAMPLE_SKETCH), "--prompt", "x", "--rounds", "3", "--warmup", "1", "--candidates", "1",
                      "--seed", "7"])
    output = " ".join(capsys.readouterr().out.split())
    session_dirs = list((tmp_path / "runs").iterdir())
    saved = load_session(session_dirs[0])

    # The benchmark saves four rounds with the plan, the given seed, and no selection, since selection is a human event.
    assert exit_code == 0 and len(session_dirs) == 1, f"Expected one benchmark session, got {exit_code} and {session_dirs}"
    plan = saved.session.benchmark
    assert plan == BenchmarkPlan(warmup_rounds=1, measured_rounds=3), f"Unexpected plan {plan}"
    assert [record.number for record in saved.rounds] == [1, 2, 3, 4], "The benchmark must save warm-up plus measured rounds"
    assert all(record.selection is None for record in saved.rounds), "A benchmark must record no selection"
    assert list_round_seeds(tmp_path / "runs") == [7, 7, 7, 7], "Every round must use the given seed"

    # Each round reports its total, and the conditions name the generator setup and the fake backend.
    assert "Round 1 of 4 (warm-up): total" in output and "Round 4 of 4 (measured): total" in output, "Rounds must be reported"
    assert "Generator setup: " in output and "outside every round" in output, "The conditions must name the generator setup"
    assert "FAKE BACKEND" in output, "The conditions must label the fake backend"

    # The table has a row per timed stage and a total, the warm-up round is shown apart, and few rounds get the p95 note.
    rows = re.findall(r"│ ([a-z ]+?) +│", output)
    assert rows == ["capture", "preprocessing", "generation", "evaluation", "saving", "total"], f"Unexpected rows {rows}"
    assert "Warm-up round 1: capture " in output, "The warm-up round must be shown separately"
    assert "p95 is interpolated from 3 rounds, so treat it as indicative." in output, "Few rounds must get the p95 note"


@pytest.mark.parametrize(("settings_text", "expected_seed"), [(None, 0), ("[run]\nseed = 9\n", 9)], ids=["default", "settings"])
def test_cli_benchmark_seed(tmp_path: Path, settings_text: str | None, expected_seed: int) -> None:
    """
    Without --seed a benchmark pins seed 0 unless the settings file sets run.seed.
    """
    # Write the settings file when the case has one.
    if settings_text is not None:
        (tmp_path / "sketchloop.toml").write_text(settings_text, encoding="utf-8")

    exit_code = main(["benchmark", str(EXAMPLE_SKETCH), "--prompt", "x", "--rounds", "1", "--warmup", "1", "--candidates", "1"])

    # Both rounds use the pinned seed.
    assert exit_code == 0, f"The benchmark must succeed, got exit code {exit_code}"
    assert list_round_seeds(tmp_path / "runs") == [expected_seed] * 2, f"Every round must use seed {expected_seed}"


@pytest.mark.parametrize("flag", [["--rounds", "0"], ["--warmup", "-1"]], ids=["no-rounds", "negative-warmup"])
def test_cli_benchmark_rejects_counts(tmp_path: Path, flag: list[str]) -> None:
    """
    Zero measured rounds or a negative warm-up exit with code 2 before anything is saved.
    """
    with pytest.raises(SystemExit) as raised:
        main(["benchmark", str(EXAMPLE_SKETCH), "--prompt", "x", *flag])

    # argparse refuses the count, and no runs folder is created.
    assert raised.value.code == 2, f"Expected exit code 2, got {raised.value.code}"
    assert not (tmp_path / "runs").exists(), "A refused benchmark must save nothing"


def refuse_camera(*args: object) -> None:
    """
    Fail the test if the benchmark creates the camera source.
    """
    raise AssertionError("benchmark must never use the camera")


def test_cli_benchmark_ignores_camera_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A settings file naming the camera does not apply to a benchmark, which always reads its sketch file.
    """
    (tmp_path / "sketchloop.toml").write_text('[capture]\nsource = "camera"\n', encoding="utf-8")
    monkeypatch.setitem(CAPTURE_SOURCES, "camera", refuse_camera)
    exit_code = main(["benchmark", str(EXAMPLE_SKETCH), "--prompt", "x", "--rounds", "1", "--warmup", "0", "--candidates", "1"])

    # The round records the file source.
    rounds = load_session(next((tmp_path / "runs").iterdir())).rounds
    assert exit_code == 0 and [record.source for record in rounds] == ["file"], f"Expected one file round, got {rounds}"


def test_cli_benchmark_failing_round(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A refused round ends the benchmark with one error line, leaves the failed round saved, and leaves nothing to summarize.
    """
    exit_code = main(["benchmark", str(EXAMPLE_SKETCH), "--prompt", "x", "--candidates", "99"])
    errors = capsys.readouterr().err
    folder = next((tmp_path / "runs").iterdir())
    saved = load_session(folder)
    rounds = saved.rounds

    # The benchmark exits with code 1, one error line, and a saved failed round 1.
    assert exit_code == 1 and errors.count("\n") == 1, f"Expected one error line and exit code 1, got {exit_code}: {errors!r}"
    assert [(record.number, record.status) for record in rounds] == [(1, "failed")], f"Expected a failed round 1, got {rounds}"

    # Without a completed measured round, the conditions and show both say there is nothing to report.
    print_benchmark_conditions(saved, summarize_benchmark(rounds, saved.session.benchmark), Console())
    show_exit = main(["show", str(folder)])
    output = " ".join(capsys.readouterr().out.split())
    assert "Benchmark conditions" in output and "No measured round completed." in output, "The conditions must say none completed"
    assert show_exit == 0 and "No measured round completed, so there are no statistics." in output, "show must say so too"


@pytest.mark.parametrize(("flags", "settings_text", "expected"), [
    (["--precision", "float16"], None, TorchPrecision.FLOAT16),
    ([], '[generator.diffusers]\nprecision = "float32"\n', TorchPrecision.FLOAT32),
    ([], None, None)
], ids=["flag", "settings-file", "device-default"])
def test_cli_benchmark_passes_precision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str],
                                        settings_text: str | None, expected: TorchPrecision | None) -> None:
    """
    A --precision flag or a settings value reaches the diffusers backend, and leaving both out keeps the device default.
    """
    received = []

    def make_recording_generator(*args: object, precision: TorchPrecision | None = None, **kwargs: object) -> FakeGenerator:
        """
        Record the precision the command asked for and return the fake backend.
        """
        received.append(precision)
        return FakeGenerator()

    monkeypatch.setattr("sketchloop.cli.common.make_generator", make_recording_generator)

    # A settings file in the working folder is found without --config.
    if settings_text is not None:
        (tmp_path / "sketchloop.toml").write_text(settings_text, encoding="utf-8")
    exit_code = main(["benchmark", str(EXAMPLE_SKETCH), "--prompt", "chair", "--backend", "diffusers", "--rounds", "1",
                      "--warmup", "0", "--candidates", "1", "--runs-dir", str(tmp_path / "runs"), *flags])

    # The benchmark ran, and the backend got the chosen precision or None for the device default.
    assert exit_code == 0, f"The benchmark must succeed, got exit code {exit_code}"
    assert received == [expected], f"Expected precision {expected}, got {received}"
