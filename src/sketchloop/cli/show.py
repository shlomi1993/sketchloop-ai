"""
The show subcommand, which prints a saved session's records and lists missing or changed files.
"""

import argparse

from collections.abc import Mapping, Sequence
from rich.console import Console
from rich.table import Table
from typing import Final

from sketchloop.app import describe_selection
from sketchloop.cli.common import format_machine, format_timings
from sketchloop.domain import Unavailable
from sketchloop.experiments import (SESSION_FILE, BenchmarkSummary, DrawingRecord, ExperimentRecordError, RerunComparison,
                                    RoundRecord, SavedSession, SuggestionRecord, SuggestionStage, compare_rounds,
                                    decide_suggestion, format_value, load_session, summarize_benchmark)
from sketchloop.follow import PROVISIONAL_RULE, FollowDecision, FollowResult, FollowRule


REPLAY_LIMITS: Final = "A rerun restores the recorded conditions. Equal seeds do not guarantee identical images across devices, library versions, or model revisions."  # noqa: E501
OLDER_TIMINGS_NOTE: Final = "Recorded by an older sketchloop: generation may include the model load, and capture, saving, and total were not measured."  # noqa: E501
P95_RELIABLE_ROUNDS: Final = 20
COMPUTED_NOTE: Final = "computed by show, not recorded"
FAKE_SUGGESTER_WARNING: Final = "FAKE SUGGESTER: these parts are test shapes, not suggestions from a real model."


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
    if session.name is not None:
        console.print(f"Experiment: {session.name}", markup=False)

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
    if record.scores:
        console.print(f"Scores: {record.evaluator}, {format_controls(record.score_details)}", markup=False, soft_wrap=True)

    print_candidates(record, console)
    selection = describe_selection(record.iteration, record.selection) if record.selection else "not recorded"
    console.print(f"Selection: {selection}")


def format_duration(seconds: float) -> str:
    """
    Format seconds as "12 min 31 s", or "45 s" under a minute.
    """
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def format_offset(seconds: float) -> str:
    """
    Format seconds from a drawing's start as "+mm:ss".
    """
    minutes, rest = divmod(round(seconds), 60)
    return f"+{minutes:02d}:{rest:02d}"


def describe_prompt(suggestion: SuggestionRecord) -> str:
    """
    Quote the prompt exactly as typed, say "(empty)" for a blank one, and mark one the model's token limit cut.
    """
    prompt = suggestion.request.prompt
    truncated = suggestion.result is not None and suggestion.result.effective.controls.get("prompt_truncated") is True
    quoted = f'"{prompt}"' if prompt.strip() else "(empty)"
    return f"{quoted} (truncated)" if truncated else quoted


def describe_part_results(decision: FollowDecision) -> str:
    """
    List each part's result by its 1-based number, with the reason when it is undetermined, or the suggestion's own reason.
    """
    if not decision.parts:
        return decision.reason

    words = {FollowResult.FOLLOWED: "followed", FollowResult.NOT_FOLLOWED: "not followed"}
    results = [words.get(part.result) or f"undetermined ({part.reason})" for part in decision.parts]
    return ", ".join(f"part {part.part + 1} {result}" for part, result in zip(decision.parts, results, strict=True))


def describe_window(record: DrawingRecord, suggestion: SuggestionRecord) -> str:
    """
    Describe a suggestion's window: its length and captures, and what ended it, or that it is still open.
    """
    window = suggestion.window
    if window is None:
        return f"Window open, {len(record.captures) - suggestion.baseline_capture} captures so far"

    captures = window.last_capture - suggestion.baseline_capture if window.last_capture else 0
    seconds = (window.ended_at - suggestion.shown_at).total_seconds()
    return f"Window {format_duration(seconds)}, {captures} captures, ended by {window.end.replace('_', ' ')}"


def print_suggestion(record: DrawingRecord, suggestion: SuggestionRecord, rule: FollowRule | None, console: Console) -> None:
    """
    Print one suggestion in two lines: when, what was asked and returned with its timings, then its result and window.

    Args:
        record (DrawingRecord): Drawing holding the suggestion.
        suggestion (SuggestionRecord): Suggestion to print.
        rule (FollowRule | None): Rule to decide an undecided window with, or None when the drawing has no scores.
        console (Console): Console used for output.
    """
    shown_or_pressed = suggestion.shown_at or suggestion.requested_at
    offset = format_offset((shown_or_pressed - record.started_at).total_seconds())
    prompt = describe_prompt(suggestion) if suggestion.request else "no request"
    if suggestion.failure is not None:
        failure = suggestion.failure
        reason = f"failed during {failure.stage}: {failure.error}: {failure.message}"
        console.print(f"Suggestion {suggestion.number} at {offset}: {prompt}, {reason}", style="red", markup=False)
        return

    # Say what was asked and returned, and how long the model took, apart from any first load.
    timings = suggestion.timings_seconds
    parts = len(suggestion.result.parts)
    details = [prompt, f"{parts} parts" if parts else "no new strokes", f"seed {format_value(suggestion.result.seed)}"]
    stages = (SuggestionStage.GENERATION, SuggestionStage.MODEL_LOAD)
    details += [f"{stage.replace('_', ' ')} {timings[stage]:.2f} s" for stage in stages if stage in timings]
    console.print(f"Suggestion {suggestion.number} at {offset}: {', '.join(details)}", markup=False, soft_wrap=True)

    # Print the recorded decision, or decide an open or older window from its scores, saying so.
    decision, note = suggestion.decision, ""
    if decision is None and rule is not None:
        decision, note = decide_suggestion(record, suggestion, rule), f", {COMPUTED_NOTE}"
    outcome = f"{decision.result.replace('_', ' ')}{note}: {describe_part_results(decision)}" if decision else "not scored"
    console.print(f"  {outcome}. {describe_window(record, suggestion)}.", markup=False, soft_wrap=True)


def describe_drawing_state(record: DrawingRecord) -> str:
    """
    Say how a drawing ended and after how long, or that it is not finished.
    """
    if record.end is None:
        return "not finished"

    duration = format_duration((record.ended_at - record.started_at).total_seconds())
    return f"finished after {duration}" if record.end == "finished" else f"ended by a new session after {duration}"


def print_drawing_record(record: DrawingRecord, console: Console) -> None:
    """
    Print a saved drawing: its state, captures, canvas, suggester, follow method and rule, and each suggestion's result.

    Args:
        record (DrawingRecord): Saved drawing.
        console (Console): Console used for output.
    """
    console.rule(f"Drawing {record.number}")
    final = record.captures[record.final_capture - 1] if record.final_capture else None
    final_text = f", final drawing {(final.canvas or final.raw).path}" if final else ""
    canvas = f"canvas {record.canvas_width}x{record.canvas_height}"
    summary = f"{describe_drawing_state(record)}, {len(record.captures)} captures, {canvas}{final_text}"
    console.print(f"Drawing {record.number}: {summary}", markup=False, soft_wrap=True)
    if not record.suggestions_enabled:
        console.print("Suggestions: off")
        return

    # Name the suggester and the follow method, and warn when the parts came from the fake.
    results = [suggestion.result for suggestion in record.suggestions if suggestion.result is not None]
    first = results[0].suggester if results else None
    identity = f", {first.adapter} {first.adapter_version}, model {format_value(first.model_id)}" if first else ""
    method = record.follow_detector
    method_text = f"follow method {method.name} {method.version}" if method else "no follow method, so nothing was scored"
    console.print(f"Suggester: {record.suggester}{identity}, {method_text}", markup=False, soft_wrap=True)
    if any(result.suggester.is_fake for result in results):
        console.print(FAKE_SUGGESTER_WARNING, style="yellow")

    # Name the rule, applying the provisional one to a drawing saved before decisions were recorded.
    rule = record.rule if record.rule is not None else PROVISIONAL_RULE if method else None
    if rule is not None:
        recorded = "Rule" if record.rule is not None else "Rule, not recorded, so show applies the provisional one"
        console.print(f"{recorded}: {rule.describe()}", markup=False, soft_wrap=True)

    for suggestion in record.suggestions:
        print_suggestion(record, suggestion, rule, console)


def print_benchmark_summary(summary: BenchmarkSummary, records: Sequence[RoundRecord], console: Console) -> None:
    """
    Print a benchmark session's round counts, per-stage statistics table, warm-up timings, and caveats.

    Args:
        summary (BenchmarkSummary): Statistics recomputed from the saved rounds.
        records (Sequence[RoundRecord]): The session's saved rounds, for the warm-up timings.
        console (Console): Console used for output.
    """
    # Name the rounds by role, and warn when settings changed between the measured rounds.
    console.rule("Benchmark summary")
    plan = summary.plan
    console.print(f"Plan: {plan.warmup_rounds} warm-up and {plan.measured_rounds} measured rounds")
    for label, numbers in (("Warm-up", summary.warmup_rounds), ("Measured", summary.measured_rounds),
                           ("Failed", summary.failed_rounds)):
        console.print(f"{label} rounds: {', '.join(str(number) for number in numbers) or 'none'}")
    if summary.varying_settings:
        console.print(f"Settings varied across measured rounds: {', '.join(summary.varying_settings)}", style="yellow")

    # Tabulate each stage and the total in seconds, when any measured round completed.
    if not summary.stages:
        console.print("No measured round completed, so there are no statistics.", style="yellow")
    else:
        table = Table()
        for column in ("Stage", "Median", "p95", "Min", "Max", "N"):
            table.add_column(column, justify="left" if column == "Stage" else "right")
        for stage, stage_statistics in summary.stages.items():
            values = (stage_statistics.median_seconds, stage_statistics.p95_seconds, stage_statistics.min_seconds,
                      stage_statistics.max_seconds)
            table.add_row(stage.replace("_", " "), *(f"{value:.2f}" for value in values), str(stage_statistics.count))
        console.print(table)

    # Show each warm-up round separately, since it may include the model load.
    for record in (record for record in records if record.number in summary.warmup_rounds):
        console.print(f"Warm-up round {record.number}: {format_timings(record.timings_seconds, record.total_seconds)}",
                      soft_wrap=True)

    # A 95th percentile from few rounds is an interpolation rather than an observed tail.
    measured = len(summary.measured_rounds)
    if measured < P95_RELIABLE_ROUNDS:
        console.print(f"p95 is interpolated from {measured} rounds, so treat it as indicative.", style="yellow")


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

    for drawing in saved.drawings:
        print_drawing_record(drawing, console)

    # Recompute and print a benchmark session's statistics from its saved rounds.
    if saved.session.benchmark is not None:
        print_benchmark_summary(summarize_benchmark(saved.rounds, saved.session.benchmark), saved.rounds, console)

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
