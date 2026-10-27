import argparse
import math

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Final

from sketchloop.backends import GenerationBackend
from sketchloop.backends.diffusers import GenerationMode
from sketchloop.components import NO_RUN_SKETCH_MESSAGE, check_component_names
from sketchloop.domain import Rotation
from sketchloop.settings import Settings, SettingsError, find_settings_file, load_settings, override_settings


CONFIG_HELP: Final = "settings file, by default sketchloop.toml in the working folder when present"


class Command(StrEnum):
    """
    Subcommands of the sketchloop command.
    """
    RUN = "run"
    SHOW = "show"
    RERUN = "rerun"
    UI = "ui"
    BENCHMARK = "benchmark"


# Settings key each flag overrides, by command and the attribute argparse stores the flag in.
FLAG_SETTINGS: Final[dict[str, dict[str, str]]] = {
    Command.RUN: {
        "candidates": "run.candidates",
        "seed": "run.seed",
        "runs_dir": "storage.runs_dir",
        "raw": "preprocessing.raw",
        "rotate": "preprocessing.rotation",
        "backend": "generator.backend",
        "mode": "generator.diffusers.mode",
        "camera_index": "capture.camera_index",
    },
    Command.UI: {
        "candidates": "ui.candidates",
        "seed": "ui.seed",
        "runs_dir": "storage.runs_dir",
        "capture_interval": "ui.capture_interval_seconds",
        "backend": "generator.backend",
        "mode": "generator.diffusers.mode",
    },
    Command.RERUN: {"runs_dir": "storage.runs_dir"},
    Command.BENCHMARK: {
        "candidates": "run.candidates",
        "seed": "run.seed",
        "runs_dir": "storage.runs_dir",
        "raw": "preprocessing.raw",
        "rotate": "preprocessing.rotation",
        "backend": "generator.backend",
        "mode": "generator.diffusers.mode",
    },
}


def make_int_parser(minimum: int) -> Callable[[str], int]:
    """
    Make an argparse type that reads an integer of at least the minimum, so bad values fail before anything is saved.

    Args:
        minimum (int): Smallest accepted value.

    Returns:
        Callable[[str], int]: Function that parses the option text.
    """
    def parse_int(text: str) -> int:
        # Name the limit in the error, which argparse prints with the option name.
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")

        return value

    return parse_int


def parse_positive_seconds(text: str) -> float:
    """
    Read a time in seconds that is greater than zero, so argparse rejects a capture interval that would never fire.
    """
    seconds = float(text)
    if not (seconds > 0 and math.isfinite(seconds)):
        raise argparse.ArgumentTypeError(f"must be greater than 0 seconds, got {text}")

    return seconds


def check_run_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """
    Reject run or benchmark flags that contradict each other, before any settings file is read.

    Args:
        parser (argparse.ArgumentParser): The run or benchmark subcommand parser, used to report errors.
        args (argparse.Namespace): Parsed run or benchmark options.
    """
    # Take the sketch from at most one source, since the settings file may name the camera.
    if args.sketch is not None and getattr(args, "camera", False):
        parser.error("give exactly one of a sketch file or --camera")

    # Rotation is part of preprocessing, which --raw skips.
    if getattr(args, "rotate", 0) and getattr(args, "raw", False):
        parser.error("--rotate cannot be combined with --raw, which skips preprocessing")


def collect_flag_overrides(args: argparse.Namespace) -> dict[str, object]:
    """
    Collect the settings the given flags override, by dotted key, where a sketch file or --camera sets the capture source.

    Args:
        args (argparse.Namespace): Parsed options, where a flag that was not given is absent.

    Returns:
        dict[str, object]: Flag values by dotted settings key, such as "run.candidates".
    """
    overrides = {key: getattr(args, name) for name, key in FLAG_SETTINGS[args.command].items() if name in args}
    if args.command == Command.RUN and (args.sketch is not None or args.camera):
        overrides["capture.source"] = "camera" if args.camera else "file"

    # Benchmark always reads its sketch file, since a camera benchmark would need a person for every round.
    if args.command == Command.BENCHMARK:
        overrides["capture.source"] = "file"

    return overrides


def check_settings(args: argparse.Namespace, settings: Settings, label: str) -> None:
    """
    Reject effective settings that contradict each other once the file and the flags are combined.

    Args:
        args (argparse.Namespace): Parsed options, to tell flags from file values.
        settings (Settings): Settings after the flags were applied.
        label (str): Settings file name used in messages.
    """
    # Only the diffusers backend has modes, while a mode in the file's [generator.diffusers] is simply unused.
    if "mode" in args and settings.generator.backend != GenerationBackend.DIFFUSERS:
        raise SettingsError(f"--mode needs --backend {GenerationBackend.DIFFUSERS}")

    if args.command not in (Command.RUN, Command.BENCHMARK):
        return

    # Rotation is part of preprocessing, which raw skips, and a file source needs a sketch file.
    preprocessing = settings.preprocessing
    if preprocessing.rotation and preprocessing.raw:
        rotation = preprocessing.rotation
        raise SettingsError(f"preprocessing.rotation {rotation} cannot be combined with preprocessing.raw, which skips preprocessing. Change one in {label} or pass --rotate 0 or --no-raw.")  # noqa: E501

    if settings.capture.source == "file" and args.sketch is None:
        raise SettingsError(NO_RUN_SKETCH_MESSAGE)


def resolve_settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """
    Read the settings file, apply the given flags, and check the result, storing it in args.settings.

    Args:
        parser (argparse.ArgumentParser): The subcommand parser, used to report errors with exit code 2.
        args (argparse.Namespace): Parsed options, which gain settings and settings_file.
    """
    # Check the file's own values before flags replace any of them, so a typo in the file is always reported.
    try:
        settings_file = find_settings_file(args.config)
        label = settings_file.as_posix() if settings_file else "built-in settings"
        file_settings = load_settings(settings_file)
        check_component_names(file_settings, label)
        settings = override_settings(file_settings, collect_flag_overrides(args))
        check_settings(args, settings, label)
    except SettingsError as error:
        parser.error(str(error))

    args.settings_file = settings_file
    args.settings = settings


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse the sketchloop subcommand and its options.

    Args:
        argv (list[str] | None): Arguments to parse, or None for the process arguments.

    Returns:
        argparse.Namespace: Parsed options, with the subcommand in `command`, and for run, rerun, ui, and benchmark the effective
            `settings` and the `settings_file` they came from.
    """
    # Options that a settings file can also set are absent unless given, so the file and the defaults can apply.
    formatter = argparse.ArgumentDefaultsHelpFormatter
    unset = argparse.SUPPRESS
    parser = argparse.ArgumentParser(prog="sketchloop", description="Run, inspect, and rerun sketch sessions.")
    commands = parser.add_subparsers(dest="command", required=True)

    # Run a new session.
    run_parser = commands.add_parser(Command.RUN, help="run a new session", description="Generate alternatives from a sketch and pick some.", formatter_class=formatter)  # noqa: E501
    run_parser.add_argument("sketch", type=Path, nargs="?", help="PNG or JPEG sketch image to start from")
    run_parser.add_argument("--camera", action="store_true", help="capture the sketch from a webcam instead of a file (settings capture.source)")  # noqa: E501
    run_parser.add_argument("--camera-index", type=make_int_parser(0), default=unset, help="webcam to use with --camera (settings capture.camera_index, default 0)")  # noqa: E501
    run_parser.add_argument("--prompt", required=True, help="text guidance for generation")
    run_parser.add_argument("--candidates", type=make_int_parser(1), default=unset, help="number of candidates to generate (settings run.candidates, default 4)")  # noqa: E501
    run_parser.add_argument("--seed", type=make_int_parser(0), default=unset, help="base seed for reproducible candidates (settings run.seed, default random)")  # noqa: E501
    run_parser.add_argument("--runs-dir", type=Path, default=unset, help="folder for run outputs (settings storage.runs_dir, default runs)")  # noqa: E501
    run_parser.add_argument("--raw", action=argparse.BooleanOptionalAction, default=unset, help="generate from your original sketch instead of the processed one (settings preprocessing.raw, default off)")  # noqa: E501
    run_parser.add_argument("--rotate", type=int, choices=list(Rotation), default=unset, help="degrees to turn the sketch clockwise before processing (settings preprocessing.rotation, default 0)")  # noqa: E501
    run_parser.add_argument("--backend", type=GenerationBackend, choices=list(GenerationBackend), default=unset, help="generation backend to use (settings generator.backend, default fake)")  # noqa: E501
    run_parser.add_argument("--mode", type=GenerationMode, choices=list(GenerationMode), default=unset, help="diffusers mode (settings generator.diffusers.mode, default fast on CPU and quality on a GPU)")  # noqa: E501
    run_parser.add_argument("--config", type=Path, help=CONFIG_HELP)

    # Show or rerun a saved session.
    show_parser = commands.add_parser(Command.SHOW, help="show a saved session without loading a model", description="Show a saved session and check its files.")  # noqa: E501
    show_parser.add_argument("session_folder", type=Path, help="session folder, such as runs/<session>")
    rerun_parser = commands.add_parser(Command.RERUN, help="rerun a saved round as a new linked session", description="Rerun a saved round with its recorded sketch and settings.", formatter_class=formatter)  # noqa: E501
    rerun_parser.add_argument("session_folder", type=Path, help="session folder, such as runs/<session>")
    rerun_parser.add_argument("--round", type=make_int_parser(1), required=True, dest="round_number", help="round number to rerun")  # noqa: E501
    rerun_parser.add_argument("--runs-dir", type=Path, default=unset, help="folder for the new session (settings storage.runs_dir, default runs)")  # noqa: E501
    rerun_parser.add_argument("--config", type=Path, help=CONFIG_HELP)

    # Measure repeated rounds from one sketch file.
    benchmark_parser = commands.add_parser(Command.BENCHMARK, help="time repeated rounds from one sketch file", description="Run warm-up and measured rounds and print stage timing statistics.", formatter_class=formatter)  # noqa: E501
    benchmark_parser.add_argument("sketch", type=Path, help="PNG or JPEG sketch image to load for every round")
    benchmark_parser.add_argument("--prompt", required=True, help="text guidance for generation")
    benchmark_parser.add_argument("--rounds", type=make_int_parser(1), default=5, help="measured rounds the statistics cover")
    benchmark_parser.add_argument("--warmup", type=make_int_parser(0), default=1, help="rounds run first and left out of the statistics")  # noqa: E501
    benchmark_parser.add_argument("--candidates", type=make_int_parser(1), default=unset, help="number of candidates per round (settings run.candidates, default 4)")  # noqa: E501
    benchmark_parser.add_argument("--seed", type=make_int_parser(0), default=unset, help="base seed for every round (settings run.seed, default 0)")  # noqa: E501
    benchmark_parser.add_argument("--raw", action=argparse.BooleanOptionalAction, default=unset, help="generate from your original sketch instead of the processed one (settings preprocessing.raw, default off)")  # noqa: E501
    benchmark_parser.add_argument("--rotate", type=int, choices=list(Rotation), default=unset, help="degrees to turn the sketch clockwise before processing (settings preprocessing.rotation, default 0)")  # noqa: E501
    benchmark_parser.add_argument("--backend", type=GenerationBackend, choices=list(GenerationBackend), default=unset, help="generation backend to use (settings generator.backend, default fake)")  # noqa: E501
    benchmark_parser.add_argument("--mode", type=GenerationMode, choices=list(GenerationMode), default=unset, help="diffusers mode (settings generator.diffusers.mode, default fast on CPU and quality on a GPU)")  # noqa: E501
    benchmark_parser.add_argument("--runs-dir", type=Path, default=unset, help="folder for run outputs (settings storage.runs_dir, default runs)")  # noqa: E501
    benchmark_parser.add_argument("--config", type=Path, help=CONFIG_HELP)

    # Serve the local browser interface.
    ui_parser = commands.add_parser(Command.UI, help="open the local browser interface", description="Run sessions from a local browser page.", formatter_class=formatter)  # noqa: E501
    ui_parser.add_argument("--candidates", type=make_int_parser(1), default=unset, help="default number of candidates per round (settings ui.candidates, default 2)")  # noqa: E501
    ui_parser.add_argument("--seed", type=make_int_parser(0), default=unset, help="default base seed for reproducible candidates (settings ui.seed, default random)")  # noqa: E501
    ui_parser.add_argument("--runs-dir", type=Path, default=unset, help="folder for run outputs (settings storage.runs_dir, default runs)")  # noqa: E501
    ui_parser.add_argument("--capture-interval", type=parse_positive_seconds, default=unset, help="seconds between webcam frames the page keeps, so Generate uses a fresh one (settings ui.capture_interval_seconds, default 0.5)")  # noqa: E501
    ui_parser.add_argument("--backend", type=GenerationBackend, choices=list(GenerationBackend), default=unset, help="generation backend to use (settings generator.backend, default fake)")  # noqa: E501
    ui_parser.add_argument("--mode", type=GenerationMode, choices=list(GenerationMode), default=unset, help="diffusers mode (settings generator.diffusers.mode, default fast on CPU and quality on a GPU)")  # noqa: E501
    ui_parser.add_argument("--config", type=Path, help=CONFIG_HELP)
    args = parser.parse_args(argv)

    # Reject contradicting flags, then combine the settings file with the flags, before anything is loaded or saved.
    args.settings_file = None
    command_parsers = {
        Command.RUN: run_parser,
        Command.RERUN: rerun_parser,
        Command.UI: ui_parser,
        Command.BENCHMARK: benchmark_parser,
    }
    if args.command in (Command.RUN, Command.BENCHMARK):
        check_run_flags(command_parsers[args.command], args)

    if args.command in command_parsers:
        resolve_settings(command_parsers[args.command], args)

    return args
