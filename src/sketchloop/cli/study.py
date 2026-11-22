"""
The study subcommand, which exports the final study drawings under random codes or prints one row per study drawing.
"""

import argparse
import subprocess

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from rich import box
from rich.console import Console
from rich.table import Table
from typing import Final

from sketchloop.cli.arguments import StudyAction
from sketchloop.cli.show import CONDITION_WORDS, format_duration
from sketchloop.domain import Unavailable
from sketchloop.experiments import (DRAWINGS_FOLDER, KEY_FILE, DrawingEnd, DrawingRecord, ExportError, ExportStatus,
                                    FoundStudyDrawings, PromptOrigin, PromptOriginKind, PromptSuggestionStatus, SavedSession,
                                    StudyExport, SuggestionStatus, decide_suggestion, export_study_drawings, find_study_drawings,
                                    plan_export_entry)
from sketchloop.follow import PROVISIONAL_RULE, FollowResult


HANDOVER_NOTE: Final = "Hand over only the drawings folder. key.csv and export.json link codes to experiments and stay on this machine."  # noqa: E501
GIT_MISSING_NOTE: Final = "Git is not installed, so whether the export folder is ignored by Git was not checked."
OUTSIDE_GIT_NOTE: Final = "The export folder is outside a Git working tree, so whether it is ignored by Git was not checked."
COMPUTED_MARK: Final = "*"
COMPUTED_NOTE: Final = "* includes a result computed by summary from an open window, not recorded."
RESULTS_COLUMN: Final = "Followed/\nnot/\nundetermined"
SUMMARY_COLUMNS: Final = ("Experiment", "Drawing", "Condition", "State", "Length", "Captures", "Strokes\nshown/\nfailed",
                          RESULTS_COLUMN, "Prompt\nsuggestions", "Prompts\ntyped/\nsuggested/\nedited", "Final\nprompt",
                          "Final\ndrawing")
STATE_WORDS: Final[dict[DrawingEnd | None, str]] = {
    None: "not finished",
    DrawingEnd.FINISHED: "finished",
    DrawingEnd.NEW_SESSION: "new session",
}


def find_existing_folder(path: Path) -> Path:
    # The export folder may not exist yet, so Git runs in its nearest existing parent.
    return next(folder for folder in (path, *path.parents) if folder.is_dir())


def run_git(folder: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(folder), *arguments], capture_output=True, text=True, check=False)


def check_ignored_by_git(output: Path, console: Console) -> None:
    resolved = output.resolve()
    folder = find_existing_folder(resolved)

    # Skip the check, saying so, when Git is missing or the folder is outside a working tree.
    try:
        inside = run_git(folder, "rev-parse", "--is-inside-work-tree")
    except OSError:
        console.print(GIT_MISSING_NOTE, style="yellow")
        return

    if inside.returncode != 0 or inside.stdout.strip() != "true":
        console.print(OUTSIDE_GIT_NOTE, style="yellow")
        return

    # Refuse a folder Git would offer to commit, and one Git cannot judge.
    ignored = run_git(folder, "check-ignore", "--quiet", "--", str(resolved))
    if ignored.returncode == 1:
        raise ExportError(f"Export folder {output.as_posix()} is not ignored by Git, so participant data could be committed. Choose a folder under data/ or another ignored folder.")  # noqa: E501

    if ignored.returncode != 0:
        reason = ignored.stderr.strip().splitlines()[0] if ignored.stderr.strip() else f"exit code {ignored.returncode}"
        raise ExportError(f"Git cannot tell whether export folder {output.as_posix()} is ignored: {reason}. Choose a folder under data/.")  # noqa: E501


def print_search_notes(skipped: tuple[str, ...], passed_over: int, console: Console) -> None:
    # Name each session that failed to load, and count the entries that hold no study drawing.
    for line in skipped:
        console.print(f"Skipped {line}", style="yellow", markup=False, soft_wrap=True)

    if passed_over:
        console.print(f"Passed over {passed_over} runs-folder entries without study drawings.")


def print_export(export: StudyExport, console: Console) -> None:
    # Give the counts and where the drawings are, then each missing drawing with its reason.
    exported = [entry for entry in export.entries if entry.status == ExportStatus.EXPORTED]
    missing = [entry for entry in export.entries if entry.status == ExportStatus.MISSING]
    drawings_folder = (export.folder / DRAWINGS_FOLDER).as_posix()
    summary = f"Exported {len(exported)} drawings to {drawings_folder}, {len(missing)} missing."
    console.print(summary, markup=False, soft_wrap=True)
    for entry in missing:
        console.print(f"Missing: {entry.experiment} drawing {entry.drawing}: {entry.reason}", style="yellow", markup=False,
                      soft_wrap=True)

    # Say where the key is and that it stays here.
    print_search_notes(export.skipped, export.passed_over, console)
    console.print(f"Key: {(export.folder / KEY_FILE).as_posix()}", markup=False, soft_wrap=True)
    console.print(HANDOVER_NOTE, style="bold")


def export_study(args: argparse.Namespace, console: Console) -> None:
    # Default to a new time-named folder under the configured export folder, and check Git before writing anything.
    settings = args.settings
    output = args.output or settings.study.export_dir / f"export-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    check_ignored_by_git(output, console)
    print_export(export_study_drawings(settings.storage.runs_dir, output, match=args.match), console)


def count_results(record: DrawingRecord) -> tuple[Counter[FollowResult], bool]:
    # Count recorded decisions, and decide an open window from its scores as show does, saying it was computed.
    rule = record.rule if record.rule is not None else PROVISIONAL_RULE if record.follow_detector else None
    results, computed = Counter(), False
    for suggestion in record.suggestions:
        if suggestion.status != SuggestionStatus.SHOWN:
            continue

        decision = suggestion.decision
        if decision is None and rule is not None:
            decision, computed = decide_suggestion(record, suggestion, rule), True
        if decision is not None:
            results[decision.result] += 1

    return results, computed


def describe_results(record: DrawingRecord) -> str:
    results, computed = count_results(record)
    counts = "/".join(str(results[result]) for result in FollowResult)
    return f"{counts}{COMPUTED_MARK}" if computed else counts


def describe_prompt_suggestions(record: DrawingRecord) -> str:
    # Count shown prompt suggestions, adding failed ones when there were any.
    if record.prompt_suggestions is None:
        return "not recorded"

    statuses = Counter(item.status for item in record.prompt_suggestions)
    failed = statuses[PromptSuggestionStatus.FAILED]
    shown = str(statuses[PromptSuggestionStatus.SHOWN])
    return f"{shown}, {failed} failed" if failed else shown


def describe_prompt_origins(record: DrawingRecord) -> str:
    # Count how the prompts sent with stroke suggestions came about, adding those whose origin is unknown.
    origins = [suggestion.prompt_origin for suggestion in record.suggestions]
    kinds = Counter(origin.kind for origin in origins if isinstance(origin, PromptOrigin))
    counts = "/".join(str(kinds[kind]) for kind in PromptOriginKind)
    unknown = sum(isinstance(origin, Unavailable) for origin in origins)
    return f"{counts}, {unknown} unknown" if unknown else counts


def describe_final_prompt_origin(record: DrawingRecord) -> str:
    origin = record.final_prompt_origin
    if origin is None:
        return "-"

    return "not recorded" if isinstance(origin, Unavailable) else str(origin.kind)


def build_summary_row(saved: SavedSession, record: DrawingRecord, has_final_drawing: bool) -> list[str]:
    # Give the length of an ended drawing only, since an open one stopped at an unknown time.
    length = format_duration((record.ended_at - record.started_at).total_seconds()) if record.ended_at else "-"
    statuses = Counter(suggestion.status for suggestion in record.suggestions)
    strokes = f"{statuses[SuggestionStatus.SHOWN]}/{statuses[SuggestionStatus.FAILED]}"
    return [saved.session.name, str(record.number), CONDITION_WORDS[record.study.condition], STATE_WORDS[record.end], length,
            str(len(record.captures)), strokes, describe_results(record), describe_prompt_suggestions(record),
            describe_prompt_origins(record), describe_final_prompt_origin(record), "yes" if has_final_drawing else "no"]


def print_summary(found: FoundStudyDrawings, console: Console) -> None:
    # Tabulate one row per study drawing, deciding each final drawing as the export would.
    rows, missing = [], []
    for saved, record in found.drawings:
        entry, canvas = plan_export_entry(saved, record)
        rows.append(build_summary_row(saved, record, canvas is not None))
        missing += [f"{entry.experiment} drawing {entry.drawing} has no final drawing: {entry.reason}"] if canvas is None else []

    table = Table(box=box.SIMPLE_HEAD)
    for column in SUMMARY_COLUMNS:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*row)
    console.print(table)

    # Explain computed results, then say why each final drawing is missing.
    if any(row[SUMMARY_COLUMNS.index(RESULTS_COLUMN)].endswith(COMPUTED_MARK) for row in rows):
        console.print(COMPUTED_NOTE)
    for line in missing:
        console.print(line, style="yellow", markup=False, soft_wrap=True)


def summarize_study(args: argparse.Namespace, console: Console) -> None:
    # Read the runs folder only, and say so when no study drawing matches.
    runs_dir = args.settings.storage.runs_dir
    found = find_study_drawings(runs_dir, args.match)
    if found.drawings:
        print_summary(found, console)
    else:
        console.print(f"No study drawings in {runs_dir.as_posix()} match {args.match!r}.", markup=False, soft_wrap=True)
    print_search_notes(found.skipped, found.passed_over, console)


def run_study(args: argparse.Namespace, console: Console) -> None:
    actions = {StudyAction.EXPORT: export_study, StudyAction.SUMMARY: summarize_study}
    actions[args.study_action](args, console)
