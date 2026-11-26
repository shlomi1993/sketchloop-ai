"""
Tests of the study subcommand: the export printout, its default folder, the Git check, and the summary table.
"""

import pytest
import re
import shutil
import subprocess

from pathlib import Path
from rich.console import Console

from sketchloop.cli import main
from sketchloop.cli.study import GIT_MISSING_NOTE, HANDOVER_NOTE, OUTSIDE_GIT_NOTE, check_ignored_by_git
from sketchloop.experiments import KEY_FILE, ExportError


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


def run_study(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    # Run the command and return its exit code and its output and errors with line wrapping removed.
    exit_code = main(["study", *arguments])
    captured = capsys.readouterr()
    return exit_code, " ".join((captured.out + captured.err).split())


def test_export_prints_counts_missing_reasons_skipped_sessions_and_the_handover_note(
        study_runs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output = run_study(["export", "--runs-dir", str(study_runs), "--output", str(tmp_path / "export")], capsys)

    # The export succeeds outside Git, saying the check was skipped, and prints the counts and every missing reason.
    assert exit_code == 0, f"Expected success, got {exit_code}: {output}"
    assert OUTSIDE_GIT_NOTE in output, f"Expected the skipped Git check in {output}"
    assert "Exported 2 drawings to" in output and "5 missing." in output, f"Missing counts in {output}"
    assert "Missing: p3_car drawing 1: not finished: the server stopped during it" in output, f"Missing reason in {output}"
    assert "Missing: p8_car drawing 1: final canvas is missing or changed" in output, f"Missing reason in {output}"

    # The broken session is named, the others counted, and the key's place and the handover rule are given.
    assert "Skipped p9_broken: session.json is not valid JSON" in output, f"Missing skipped session in {output}"
    assert "Passed over 3 runs-folder entries without study drawings." in output, f"Missing passed-over count in {output}"
    assert f"/export/{KEY_FILE}" in output and HANDOVER_NOTE in output, f"Missing key or handover note in {output}"


def test_export_defaults_to_a_new_folder_under_the_settings_export_dir(study_runs: Path, tmp_path: Path,
                                                                       capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "sketchloop.toml").write_text('[study]\nexport_dir = "exports"\n', encoding="utf-8")
    exit_code, output = run_study(["export", "--runs-dir", str(study_runs), "--match", "p1_*"], capsys)

    # The export lands in one export-<UTC time> folder under the configured folder.
    assert exit_code == 0, f"Expected success, got {exit_code}: {output}"
    folders = [path.name for path in (tmp_path / "exports").iterdir()]
    assert len(folders) == 1 and re.fullmatch(r"export-\d{8}-\d{6}", folders[0]), f"Unexpected export folders {folders}"
    assert (tmp_path / "exports" / folders[0] / KEY_FILE).is_file(), "The default folder must hold the key"


def test_export_refusal_is_one_line_with_exit_code_1(study_runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code, output = run_study(["export", "--runs-dir", str(study_runs), "--output", str(study_runs / "export")], capsys)

    # An output inside the runs folder is refused with one line, and nothing is created.
    assert exit_code == 1 and "is inside the runs folder" in output, f"Expected a refusal, got {exit_code}: {output}"
    assert not (study_runs / "export").exists(), "A refused export must create nothing"


def run_git(folder: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(folder), *arguments], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is not installed")
def test_git_check_refuses_a_folder_git_does_not_ignore(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    run_git(repository, "init", "--quiet")
    (repository / ".gitignore").write_text("/ignored/\n", encoding="utf-8")
    console = Console(record=True)

    # A folder under an ignored path passes, even before it exists, and one under a tracked path is refused.
    check_ignored_by_git(repository / "ignored" / "export-1", console)
    with pytest.raises(ExportError, match="is not ignored by Git, so participant data could be committed"):
        check_ignored_by_git(repository / "tracked" / "export-1", console)
    assert OUTSIDE_GIT_NOTE not in console.export_text(), "Inside a repository the check must run"


def test_git_check_is_skipped_outside_a_repository_or_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(record=True, width=200)

    # Outside any working tree, the check is skipped with one line.
    check_ignored_by_git(tmp_path / "export-1", console)
    assert OUTSIDE_GIT_NOTE in console.export_text(), "Expected the outside-Git note"

    # Without Git, the check is skipped with one line too.
    def refuse_git(*_: object, **__: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr("sketchloop.cli.study.subprocess.run", refuse_git)
    check_ignored_by_git(tmp_path / "export-1", console)
    assert GIT_MISSING_NOTE in console.export_text(), "Expected the missing-Git note"


def test_summary_prints_one_row_per_study_drawing(study_runs: Path, capsys: pytest.CaptureFixture[str],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "220")
    exit_code, output = run_study(["summary", "--runs-dir", str(study_runs)], capsys)

    # Each study drawing has one row, and p1's row has its condition, counts, prompt origins, and final drawing.
    assert exit_code == 0, f"Expected success, got {exit_code}: {output}"
    rows = re.findall(r"(p\d_car) 1 with", output)
    assert rows == ["p1_car", "p2_car", "p3_car", "p4_car", "p6_car", "p7_car", "p8_car"], f"Unexpected rows {rows}"
    first_row = r"p1_car 1 with AI finished \d+ s 2 1/0 0/0/1 1 0/1/0 edited yes"
    assert re.search(first_row, output), f"Unexpected p1_car row in {output}"
    second_row = r"p2_car 1 without AI finished \d+ s 1 0/0 0/0/0 0 0/0/0 typed yes"
    assert re.search(second_row, output), f"Unexpected p2_car row in {output}"

    # The open drawing's window is decided by summary and marked, and missing final drawings and skipped sessions are named.
    open_row = r"p3_car 1 with AI not finished - 1 1/0 \d/\d/\d\* 0 1/0/0 - no"
    assert re.search(open_row, output), f"Unexpected p3_car row in {output}"
    assert "* includes a result computed by summary from an open window, not recorded." in output, f"Missing note in {output}"
    assert "p4_car drawing 1 has no final drawing: finished without a final frame" in output, f"Missing reason in {output}"
    assert "Skipped p9_broken" in output and "Passed over 3 runs-folder entries" in output, f"Missing notes in {output}"


def test_summary_without_study_drawings_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "runs").mkdir()
    exit_code, output = run_study(["summary", "--runs-dir", str(tmp_path / "runs"), "--match", "p1_*"], capsys)

    # An empty match is reported, not an error.
    assert exit_code == 0 and "No study drawings in" in output and "match 'p1_*'" in output, f"Unexpected output {output}"
