import argparse

from collections.abc import Mapping
from rich.console import Console
from rich.table import Table
from typing import Final

from sketchloop.app import describe_selection
from sketchloop.cli.common import format_machine, format_timings
from sketchloop.domain import Unavailable
from sketchloop.experiments import (SESSION_FILE, ExperimentRecordError, RerunComparison, RoundRecord, SavedSession,
                                    compare_rounds, format_value, load_session)


REPLAY_LIMITS: Final = "A rerun restores the recorded conditions. Equal seeds do not guarantee identical images across devices, library versions, or model revisions."  # noqa: E501
OLDER_TIMINGS_NOTE: Final = "Recorded by an older sketchloop: generation may include the model load, and capture, saving, and total were not measured."  # noqa: E501


def format_controls(controls: Mapping[str, object]) -> str:
    """
    Format controls as name=value pairs on one line, or "none".
    """
    return ", ".join(f"{name}={format_value(value)}" for name, value in controls.items()) or "none"


def format_seconds(seconds: float | Unavailable) -> str:
    """
    Format a duration such as "3.21 s", or why it is unavailable.
    """
    return format_value(seconds) if isinstance(seconds, Unavailable) else f"{seconds:.2f} s"


def print_session_header(saved: SavedSession, console: Console) -> None:
    """
    Print a saved session's ID, creation time, Python, platform, hardware, generator setup, packages, and rerun link.

    Args:
        saved (SavedSession): Loaded session.
        console (Console): Console used for output.
    """
    session, environment = saved.session, saved.session.environment
    console.rule(f"Session {session.session_id}")
    console.print(f"Created: {session.created_at.isoformat()}")
    console.print(f"Python {environment.py_version} on {environment.platform}", markup=False)
    console.print(f"Hardware: {format_machine(environment.hardware)}", markup=False, soft_wrap=True)
    console.print(f"Generator setup: {format_seconds(session.generator_setup_seconds)}", markup=False)
    installed = ", ".join(f"{name} {version}" for name, version in environment.packages.items() if isinstance(version, str))
    console.print(f"Packages: {installed or 'none'}", markup=False, soft_wrap=True)
    if session.rerun_of is not None:
        console.print(f"Rerun of: session {session.rerun_of.session_id} round {session.rerun_of.round_number}")


def print_candidates(record: RoundRecord, console: Console) -> None:
    """
    Print a completed round's candidates with seed, score, selected mark, and image path.

    Args:
        record (RoundRecord): Completed round.
        console (Console): Console used for output.
    """
    table = Table()
    for column in ("#", "Seed", "Score", "Selected", "Candidate image"):
        table.add_column(column)

    # Mark the selected candidates and show a score only where the evaluator gave one.
    selected_ids = record.selection.selected_candidate_ids if record.selection else ()
    for candidate in record.iteration.result.candidates:
        score = record.scores.get(candidate.id)
        table.add_row(str(candidate.index + 1), format_value(candidate.seed), "" if score is None else f"{score:.3f}",
                      "yes" if candidate.id in selected_ids else "", candidate.image.path)
    console.print(table)


def print_round_record(record: RoundRecord, console: Console) -> None:
    """
    Print a saved round's request, input, preprocessing, timings, and either its result and selection or its failure.

    Args:
        record (RoundRecord): Saved round.
        console (Console): Console used for output.
    """
    # Show what was asked, from which input, and how long each stage took.
    console.rule(f"Round {record.number} ({record.status})")
    if record.request is not None:
        guidance = record.request.guidance
        console.print(f"Prompt: {guidance.prompt}", markup=False, soft_wrap=True)
        console.print(f"Negative prompt: {guidance.negative_prompt or 'none'}", markup=False, soft_wrap=True)
        console.print(f"Requested: {format_controls(guidance.controls)}, seed {format_value(record.request.seed)}", markup=False)

    sketch_kind = "raw sketch" if record.use_raw else f"processed sketch, rotation {record.rotation}"
    age, source = record.frame_age_seconds, format_value(record.source)
    frame_age = "" if isinstance(age, Unavailable) else f", frame age {age:.2f} s"
    console.print(f"Input: {sketch_kind}, from {record.raw_sketch.path} (source: {source}){frame_age}", markup=False)
    console.print(f"Preprocessing: {', '.join(step.name for step in record.steps) or 'skipped'}", soft_wrap=True)
    console.print(f"Timings: {format_timings(record.timings_seconds, record.total_seconds)}", soft_wrap=True)
    if isinstance(record.total_seconds, Unavailable):
        console.print(OLDER_TIMINGS_NOTE, style="yellow")

    if record.failure is not None:
        failure = record.failure
        console.print(f"Failed during {failure.stage}: {failure.error}: {failure.message}", style="red", markup=False)
        return

    # Show the backend, the settings it used, the candidates, and the selection.
    backend = record.iteration.result.backend
    model = format_value(backend.model_id)
    console.print(f"Backend: {backend.adapter} {backend.adapter_version}, model {model}", markup=False, soft_wrap=True)
    if backend.is_fake:
        console.print("FAKE BACKEND: these images are deterministic noise, not output from a real model.", style="yellow")

    console.print(f"Effective: {format_controls(record.iteration.result.effective.controls)}", markup=False, soft_wrap=True)
    print_candidates(record, console)
    selection = describe_selection(record.iteration, record.selection) if record.selection else "not recorded"
    console.print(f"Selection: {selection}")


def print_comparison(comparison: RerunComparison, console: Console) -> None:
    """
    Print which candidates have identical bytes, each difference from the original round, and the replay limits.

    Args:
        comparison (RerunComparison): Comparison of a rerun round with its original.
        console (Console): Console used for output.
    """
    console.rule("Compared with the original round")
    identical = ", ".join(str(number) for number in comparison.identical_candidates) or "none"
    console.print(f"Candidates with identical bytes: {identical} of {comparison.candidate_count}")
    for difference in comparison.differences:
        console.print(difference, markup=False, soft_wrap=True)

    if not comparison.differences:
        console.print("No differences in backend, settings, or environment.")

    console.print(REPLAY_LIMITS)


def show_session(args: argparse.Namespace, console: Console) -> None:
    """
    Print a saved session's records and integrity problems without creating a generator.

    Args:
        args (argparse.Namespace): Parsed show options.
        console (Console): Console used for output.
    """
    saved = load_session(args.session_folder)
    print_session_header(saved, console)
    for record in saved.rounds:
        print_round_record(record, console)

    # Report missing or changed files, or confirm that all match.
    console.rule("Integrity")
    for problem in saved.problems:
        console.print(problem, style="red", markup=False)

    if not saved.problems:
        console.print(f"All {saved.checked_files} files match their recorded checksums.", style="green")

    # Compare a rerun with its original when the original session is still next to it.
    link = saved.session.rerun_of
    original_folder = saved.folder.parent / link.session_id if link else None
    if link is not None and (original_folder / SESSION_FILE).is_file():
        try:
            print_comparison(compare_rounds(load_session(original_folder), link.round_number, saved), console)
        except ExperimentRecordError as error:
            console.print(f"Cannot compare with the original session: {error}", style="yellow", markup=False)
