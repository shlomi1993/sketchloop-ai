"""
Tests of the rerun subcommand's linked session and comparison report.
"""

import json
import pytest

from collections.abc import Mapping, Sequence
from pathlib import Path

from sketchloop.backends.diffusers import TorchDevice, TorchPrecision
from sketchloop.backends.fake import FakeGenerator
from sketchloop.cli import main
from sketchloop.components import EVALUATORS
from sketchloop.domain import Candidate, GenerationRequest, Unavailable
from sketchloop.evaluation import Evaluator
from sketchloop.evaluators.clip import CLIP_B32_MODEL
from sketchloop.experiments import MISSING_SOURCE_REASON, load_session
from sketchloop.settings import Settings


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


def refuse_generator(*args: object, **kwargs: object) -> None:
    """
    Fail the test if a command creates this generator.
    """
    raise AssertionError("This command must not create this generator")


def test_cli_rerun_links_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Rerunning a fake round saved before sources, with a diffusers settings file, makes a linked fake session in the file's folder.
    """
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    config_dir = tmp_path / "config"
    runs_dir = config_dir / "runs"
    config_dir.mkdir()
    config = config_dir / "rerun.toml"
    config.write_text('[generator]\nbackend = "diffusers"\n\n[storage]\nruns_dir = "runs"\n', encoding="utf-8")
    main(["dev", "run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", "2", "--runs-dir", str(runs_dir)])
    original = next(runs_dir.iterdir())
    recorded_source = load_session(original).rounds[0].source

    # Name another CPU in the original session, standing in for a rerun on a different machine.
    session_file = original / "session.json"
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    session_data["environment"]["hardware"] = {"cpu": "Synthetic CPU", "logical_cores": 2, "memory_bytes": 1024}
    session_file.write_text(json.dumps(session_data), encoding="utf-8")

    # Strip input.source to stand in for a round saved before sources were tracked.
    record_file = original / "round-1" / "round.json"
    data = json.loads(record_file.read_text(encoding="utf-8"))
    del data["input"]["source"]
    record_file.write_text(json.dumps(data), encoding="utf-8")
    original_source = load_session(original).rounds[0].source

    # The file asks for diffusers, but a rerun recreates the recorded fake backend, so diffusers must never be created.
    monkeypatch.setattr("sketchloop.components.DiffusersSketchGenerator", refuse_generator)
    rerun_exit = main(["dev", "rerun", str(original), "--round", "1", "--config", str(config)])
    rerun_output = capsys.readouterr().out

    rerun_folders = [path for path in runs_dir.iterdir() if path != original]
    monkeypatch.setattr("sketchloop.components.FakeGenerator", refuse_generator)
    show_exit = main(["dev", "show", str(rerun_folders[0])]) if rerun_folders else None
    show_output = capsys.readouterr().out

    # Both commands succeed, the rerun is saved in the file's runs folder, and it records which original round it repeats.
    assert rerun_exit == 0 and show_exit == 0, f"Rerun and show must succeed, got {rerun_exit} and {show_exit}"
    assert len(rerun_folders) == 1, f"The rerun must be saved in the settings file's runs folder, got {rerun_folders}"
    rerun_folder = rerun_folders[0]
    link = load_session(rerun_folder).session.rerun_of
    assert link.session_id == original.name and link.round_number == 1, f"The rerun must link to round 1, got {link}"

    # The deterministic fake gives identical candidates, the rerun never claims exact replay, and show reports the comparison.
    assert "Candidates with identical bytes: 1, 2 of 2" in rerun_output, "The fake rerun must give identical candidates"
    assert "reproduced" not in rerun_output.lower(), "A rerun must never claim exact replay"
    assert "Candidates with identical bytes: 1, 2 of 2" in show_output, "show must compare a rerun with its original"

    # A run from a file records file, the older record loads as unavailable, and the rerun copies that.
    expected_source = Unavailable(MISSING_SOURCE_REASON)
    rerun_source = load_session(rerun_folder).rounds[0].source
    assert recorded_source == "file", f"run from a file must record source file, got {recorded_source}"
    assert original_source == expected_source, f"Expected original source {expected_source}, got {original_source}"
    assert rerun_source == expected_source, f"The rerun must copy the original source {expected_source}, got {rerun_source}"

    # The rerun reuses the stored sketch without a capture, records its own setup time, and lists the changed CPU.
    rerun_saved = load_session(rerun_folder)
    assert "capture" not in rerun_saved.rounds[0].timings_seconds, "A rerun reuses the stored sketch, so it has no capture"
    assert isinstance(rerun_saved.session.generator_setup_seconds, float), "The rerun must record its generator setup"
    assert "cpu: Synthetic CPU -> " in " ".join(rerun_output.split()), "The comparison must list the changed CPU"

    # show prints the rerun round's source, naming the reason it is unavailable.
    shown_source = "(source: unavailable (recorded before input sources were tracked))"
    assert shown_source in " ".join(show_output.split()), f"show must print {shown_source}"


def record_float16_round(runs_dir: Path, monkeypatch: pytest.MonkeyPatch, adapter: str) -> Path:
    """
    Run one fake round, then record it as made by the given adapter in float16, and return its session folder.
    """
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    main(["dev", "run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--runs-dir", str(runs_dir)])
    original = next(runs_dir.iterdir())
    record_file = original / "round-1" / "round.json"
    data = json.loads(record_file.read_text(encoding="utf-8"))
    data["result"]["backend"]["adapter"] = adapter
    data["result"]["effective"]["controls"]["dtype"] = "float16"
    record_file.write_text(json.dumps(data), encoding="utf-8")
    return original


def test_cli_rerun_uses_recorded_precision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    """
    A rerun creates its backend with the recorded precision, and a diffusers float16 round is refused on a CPU-only machine.
    """
    original = record_float16_round(tmp_path / "runs", monkeypatch, "sketchloop.fake")
    received = []

    def make_recording_generator(settings: Settings) -> FakeGenerator:
        """
        Record the precision the rerun asked for and return the fake backend.
        """
        received.append(settings.generator.diffusers.precision)
        return FakeGenerator()

    monkeypatch.setattr("sketchloop.cli.common.make_generator", make_recording_generator)
    exit_code = main(["dev", "rerun", str(original), "--round", "1", "--runs-dir", str(tmp_path / "runs")])

    # The rerun succeeded and asked for the recorded float16.
    assert exit_code == 0, f"The rerun must succeed, got exit code {exit_code}"
    assert received == [TorchPrecision.FLOAT16], f"Expected the recorded float16, got {received}"

    # A diffusers round in float16 fails on a CPU-only machine with one line, creating no backend.
    diffusers_original = record_float16_round(tmp_path / "cpu-runs", monkeypatch, "sketchloop.diffusers")
    capsys.readouterr()
    monkeypatch.setattr("sketchloop.cli.rerun.select_device", lambda: TorchDevice.CPU)
    monkeypatch.setattr("sketchloop.cli.common.make_generator", refuse_generator)
    cpu_exit = main(["dev", "rerun", str(diffusers_original), "--round", "1", "--runs-dir", str(tmp_path / "cpu-runs")])
    captured = capsys.readouterr()
    output = " ".join((captured.out + captured.err).split())
    assert cpu_exit == 1, f"The rerun must fail, got exit code {cpu_exit}"
    expected = "Round 1 ran in float16 on a GPU, so it cannot be rerun on this CPU-only machine."
    assert expected in output, f"Unexpected output {output}"


class RecordingEvaluator(Evaluator):
    """
    Test evaluator standing in for CLIP, recorded under its own class name and giving no scores.
    """

    def evaluate(self, candidates: Sequence[Candidate], payloads: Mapping[str, bytes],
                 request: GenerationRequest) -> Mapping[str, float]:
        """
        Return no scores.
        """
        return {}


@pytest.mark.parametrize(("recorded", "settings_evaluator", "expected_evaluator", "expects_note"), [
    ({"evaluator": "ClipEvaluator", "score_details": {"model": CLIP_B32_MODEL.repo_id}}, "none", "RecordingEvaluator", False),
    ({"evaluator": "ConstantEvaluator", "score_details": {}}, "none", "NoOpEvaluator", True)
], ids=["clip-b32", "unknown"])
def test_cli_rerun_reuses_recorded_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                             recorded: dict[str, object] | None, settings_evaluator: str,
                                             expected_evaluator: str, expects_note: bool) -> None:
    """
    A rerun scores with the evaluator the round recorded, and falls back to the settings evaluator with a note otherwise.
    """
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    runs_dir = tmp_path / "runs"
    main(["dev", "run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", "1", "--runs-dir", str(runs_dir)])
    original = next(runs_dir.iterdir())

    # Rewrite the recorded evaluator to stand in for a CLIP round or an evaluator rerun cannot recreate.
    if recorded is not None:
        record_file = original / "round-1" / "round.json"
        data = json.loads(record_file.read_text(encoding="utf-8"))
        data["evaluation"] |= recorded
        record_file.write_text(json.dumps(data), encoding="utf-8")

    # The settings name another evaluator, and clip-b32 builds a stand-in so no model is needed.
    monkeypatch.setitem(EVALUATORS, "clip-b32", lambda settings: RecordingEvaluator())
    config = tmp_path / "rerun.toml"
    config.write_text(f'[evaluator]\nname = "{settings_evaluator}"\n', encoding="utf-8")
    capsys.readouterr()
    exit_code = main(["dev", "rerun", str(original), "--round", "1", "--config", str(config), "--runs-dir", str(runs_dir)])
    output = " ".join(capsys.readouterr().out.split())

    # The rerun round records the expected evaluator.
    rerun_folder = next(path for path in runs_dir.iterdir() if path != original)
    rerun_evaluator = load_session(rerun_folder).rounds[0].evaluator
    assert exit_code == 0, f"The rerun must succeed, got exit code {exit_code}"
    assert rerun_evaluator == expected_evaluator, f"Expected the rerun to score with {expected_evaluator}, got {rerun_evaluator}"

    # Only a fallback prints which settings evaluator replaced the recorded one.
    note = "The recorded evaluator ConstantEvaluator cannot be recreated, so the rerun scores with none from the settings."
    assert (note in output) == expects_note, f"Expected the fallback note only when falling back, got {output!r}"
