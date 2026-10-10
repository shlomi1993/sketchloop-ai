import json
import pytest

from collections.abc import Callable
from pathlib import Path

from sketchloop.capture import load_sketch_file
from sketchloop.cli import main
from sketchloop.domain import Guidance, Unavailable, UnsupportedConfigurationError, record_no_selection, select_candidates
from sketchloop.experiments import ExperimentRecordError, describe_failure, load_session
from sketchloop.fakes import FakeGenerator
from sketchloop.session import SketchSession


EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch-photo.jpg"


def run_two_rounds(runs_dir: Path) -> SketchSession:
    """
    Run two fake rounds, picking candidate 1 in round 1 and none in a raw round 2.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), runs_dir)
    round1 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), n_candidates=2, seed=7)
    session.record_selection(select_candidates(round1.iteration, [round1.iteration.result.candidates[0].id]))
    round2 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="lamp", controls={"steps": 8}), use_raw=True)
    session.record_selection(record_no_selection(round2.iteration))
    return session


def test_records_round_trip(tmp_path: Path) -> None:
    """
    Two rounds and a failed third load back with equal iterations, steps, selections, lineage, and checksums.
    """
    session = run_two_rounds(tmp_path)
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="vase"), n_candidates=99)
    (session.folder / "round-1" / "round.json.tmp").write_text("{", encoding="utf-8")

    saved = load_session(session.folder)
    round1, round2, round3 = saved.rounds
    outcomes = session.rounds

    # The saved records reload into the same iterations, preprocessing steps, and round-to-round lineage.
    iterations_equal = [round1.iteration, round2.iteration] == [outcome.iteration for outcome in outcomes]
    assert iterations_equal, "Loaded requests, results, and effective settings must equal the session's"
    assert round1.steps == outcomes[0].steps and round2.steps == (), "Preprocessing steps must load unchanged"
    assert round2.iteration.parent_id == round1.iteration.id, "Round 2 must link to round 1"

    # Selections, stage timings, the failed round, and every file checksum survive the save and load.
    selections = [round1.selection, round2.selection]
    assert selections == [session.selections[outcome.iteration.id] for outcome in outcomes], f"Selections changed: {selections}"
    assert set(round1.timings_seconds) == {"preprocessing", "generation", "evaluation"}, "Round 1 must time every stage"
    assert round3.status == "failed" and round3.failure.stage == "generation", f"Round 3 must fail in generation: {round3}"
    assert saved.problems == () and saved.checked_files == 8, f"All files must match: {saved.problems}"

    # Package versions are recorded as real versions or as unavailable with a reason, never invented.
    versions = list(saved.session.environment.packages.values())
    assert all(isinstance(value, str | Unavailable) for value in versions), f"Unexpected versions {versions}"


@pytest.mark.parametrize(("damage", "problem"), [("flip", "does not match its recorded checksum"), ("delete", "is missing")])
def test_damaged_candidate_listed(tmp_path: Path, damage: str, problem: str) -> None:
    """
    A changed or deleted candidate file is listed as one problem instead of failing the load.
    """
    session = run_two_rounds(tmp_path)
    candidate = session.folder / "round-1" / "candidates" / "candidate-1.png"
    if damage == "flip":
        payload = bytearray(candidate.read_bytes())
        payload[-1] ^= 1
        candidate.write_bytes(payload)
    else:
        candidate.unlink()

    # The damaged file is reported as exactly one problem naming the file, while the session still loads.
    problems = load_session(session.folder).problems
    assert problems == (f"round-1/candidates/candidate-1.png {problem}.",), f"Expected one problem, got {problems}"


@pytest.mark.parametrize(("file_name", "edit", "message"), [
    ("round-1/round.json", lambda data: data["input"]["raw_sketch"].update(path="../x.png"), "outside its round folder"),
    ("session.json", lambda data: data.update(schema_version=2), "schema version 2"),
    ("round-2/round.json", lambda data: data.update(parent_id="0" * 32), "does not link to round-1")
])
def test_invalid_records_rejected(tmp_path: Path, file_name: str, edit: Callable[[dict], None], message: str) -> None:
    """
    Path traversal, an unknown schema version, and broken lineage each refuse to load.
    """
    session = run_two_rounds(tmp_path)
    record_file = session.folder / file_name
    data = json.loads(record_file.read_text(encoding="utf-8"))
    edit(data)
    record_file.write_text(json.dumps(data), encoding="utf-8")

    # Loading refuses the damaged record with an error that names the reason.
    with pytest.raises(ExperimentRecordError, match=message):
        load_session(session.folder)


def refuse_generator(*args: object, **kwargs: object) -> None:
    """
    Fail the test if a command creates a generator.
    """
    raise AssertionError("show must not create a generator")


def test_cli_rerun_links_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Rerunning a fake round makes a linked session with identical candidates, and show reads it without a generator.
    """
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    runs_dir = tmp_path / "runs"
    main(["run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", "2", "--runs-dir", str(runs_dir)])
    original = next(runs_dir.iterdir())
    rerun_exit = main(["rerun", str(original), "--round", "1", "--runs-dir", str(runs_dir)])
    rerun_output = capsys.readouterr().out

    rerun_folder = next(path for path in runs_dir.iterdir() if path != original)
    monkeypatch.setattr("sketchloop.cli.FakeGenerator", refuse_generator)
    monkeypatch.setattr("sketchloop.cli.DiffusersSketchGenerator", refuse_generator)
    show_exit = main(["show", str(rerun_folder)])
    show_output = capsys.readouterr().out

    # Both commands succeed, and the rerun session records which original round it repeats.
    link = load_session(rerun_folder).session.rerun_of
    assert rerun_exit == 0 and show_exit == 0, f"Rerun and show must succeed, got {rerun_exit} and {show_exit}"
    assert link.session_id == original.name and link.round_number == 1, f"The rerun must link to round 1, got {link}"

    # The deterministic fake gives identical candidates, the rerun never claims exact replay, and show reports the comparison.
    assert "Candidates with identical bytes: 1, 2 of 2" in rerun_output, "The fake rerun must give identical candidates"
    assert "reproduced" not in rerun_output.lower(), "A rerun must never claim exact replay"
    assert "Candidates with identical bytes: 1, 2 of 2" in show_output, "show must compare a rerun with its original"


# Build each path from pieces so this file never contains a literal home path for the publication guard to flag.
@pytest.mark.parametrize("path", ["C:" + "\\Users\\someone\\models\\unet.bin", "/" + "home/someone/.cache/unet.bin"])
def test_failure_message_hides_paths(path: str) -> None:
    """
    A saved failure message replaces absolute paths, which can name the user, with a placeholder.
    """
    failure = describe_failure("generation", OSError(f"Cannot open '{path}': access denied\nsecond line"))

    # Only the first line is kept, with the path replaced.
    expected = "Cannot open '<path>': access denied"
    assert failure.message == expected and failure.error == "OSError", f"Unexpected failure record {failure}"
