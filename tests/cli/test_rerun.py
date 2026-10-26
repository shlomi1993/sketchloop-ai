import json
import pytest

from pathlib import Path

from sketchloop.cli import main
from sketchloop.domain import Unavailable
from sketchloop.experiments import MISSING_SOURCE_REASON, load_session


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


def refuse_generator(*args: object, **kwargs: object) -> None:
    """
    Fail the test if a command creates this generator.
    """
    raise AssertionError("This command must not create this generator")


@pytest.mark.parametrize("drop_source", [False, True], ids=["with-source", "recorded-before-sources"])
def test_cli_rerun_links_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                              drop_source: bool) -> None:
    """
    Rerunning a fake round with a diffusers settings file makes a linked fake session in the file's runs folder.
    """
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    config_dir = tmp_path / "config"
    runs_dir = config_dir / "runs"
    config_dir.mkdir()
    config = config_dir / "rerun.toml"
    config.write_text('[generator]\nbackend = "diffusers"\n\n[storage]\nruns_dir = "runs"\n', encoding="utf-8")
    main(["run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", "2", "--runs-dir", str(runs_dir)])
    original = next(runs_dir.iterdir())
    recorded_source = load_session(original).rounds[0].source

    # Strip input.source to stand in for a round saved before sources were tracked.
    if drop_source:
        record_file = original / "round-1" / "round.json"
        data = json.loads(record_file.read_text(encoding="utf-8"))
        del data["input"]["source"]
        record_file.write_text(json.dumps(data), encoding="utf-8")

    original_source = load_session(original).rounds[0].source

    # The file asks for diffusers, but a rerun recreates the recorded fake backend, so diffusers must never be created.
    monkeypatch.setattr("sketchloop.backends.DiffusersSketchGenerator", refuse_generator)
    rerun_exit = main(["rerun", str(original), "--round", "1", "--config", str(config)])
    rerun_output = capsys.readouterr().out

    rerun_folders = [path for path in runs_dir.iterdir() if path != original]
    monkeypatch.setattr("sketchloop.backends.FakeGenerator", refuse_generator)
    show_exit = main(["show", str(rerun_folders[0])]) if rerun_folders else None
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

    # A run from a file records file, an older record loads as unavailable, and the rerun copies whichever it finds.
    expected_source = Unavailable(MISSING_SOURCE_REASON) if drop_source else "file"
    rerun_source = load_session(rerun_folder).rounds[0].source
    assert recorded_source == "file", f"run from a file must record source file, got {recorded_source}"
    assert original_source == expected_source, f"Expected original source {expected_source}, got {original_source}"
    assert rerun_source == expected_source, f"The rerun must copy the original source {expected_source}, got {rerun_source}"

    # show prints the rerun round's source, naming the reason when it is unavailable.
    shown_source = "(source: unavailable (recorded before input sources were tracked))" if drop_source else "(source: file)"
    assert shown_source in " ".join(show_output.split()), f"show must print {shown_source}"
