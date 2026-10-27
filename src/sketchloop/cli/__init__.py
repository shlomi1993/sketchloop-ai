"""
Entry point of the sketchloop command, which dispatches to one module per subcommand.
"""

from rich.console import Console

from sketchloop.cli.arguments import Command, parse_args
from sketchloop.cli.benchmark import run_benchmark
from sketchloop.cli.rerun import rerun_round
from sketchloop.cli.run import run_session
from sketchloop.cli.serve import launch_ui
from sketchloop.cli.show import show_session
from sketchloop.domain import SketchLoopError


def main(argv: list[str] | None = None) -> int:
    """
    Run a sketchloop subcommand from the command line and report errors in one line.

    Args:
        argv (list[str] | None, optional): Command-line arguments. Defaults to the process arguments.

    Returns:
        int: Process exit code, 0 on success and 1 on a reported error.
    """
    args = parse_args(argv)
    console = Console()
    commands = {
        Command.RUN: run_session,
        Command.SHOW: show_session,
        Command.RERUN: rerun_round,
        Command.UI: launch_ui,
        Command.BENCHMARK: run_benchmark,
    }

    # Say which settings file applies before anything else, so a file found in the working folder is never a surprise.
    if args.settings_file is not None:
        console.print(f"Settings: {args.settings_file.as_posix()}", markup=False, soft_wrap=True)

    try:
        commands[args.command](args, console)
    except (SketchLoopError, OSError) as error:
        Console(stderr=True).print(f"sketchloop error: {error}", style="red", markup=False, highlight=False, soft_wrap=True)
        return 1

    return 0
