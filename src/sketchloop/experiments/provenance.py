"""
Environment and failure provenance for records, with absolute paths redacted from messages.
"""

import platform
import re

from importlib.metadata import PackageNotFoundError, version
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final

from sketchloop.domain import Unavailable, first_line
from sketchloop.experiments.hardware import collect_hardware
from sketchloop.experiments.records import Environment, RoundFailure, RoundStage


# Distributions whose versions can change generated output or how records are read.
TRACKED_PACKAGES: Final = (
    "sketchloop-ai",
    "numpy",
    "opencv-python",
    "rich",
    "diffusers",
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "huggingface_hub",
)

# Failure messages are split into quoted strings, kept whole since paths may contain spaces, and runs of other text.
MESSAGE_TOKEN_PATTERN: Final = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")

# Punctuation that can surround an unquoted path, such as a closing parenthesis or a trailing comma.
PATH_PUNCTUATION: Final = "()[]{},;:."


def read_package_version(name: str) -> str | Unavailable:
    """
    Return an installed distribution's version, or why it is unavailable.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return Unavailable("not installed")


def collect_environment() -> Environment:
    """
    Record the Python, platform, tracked package versions, and hardware of this process.

    Returns:
        Environment: The current environment, with Unavailable where a version or hardware fact cannot be read.
    """
    system = f"{platform.system()} {platform.release()} {platform.machine()}"
    packages = {name: read_package_version(name) for name in TRACKED_PACKAGES}
    return Environment(py_version=platform.python_version(), platform=system, packages=packages, hardware=collect_hardware())


def is_absolute_path(text: str) -> bool:
    """
    Tell whether text is an absolute POSIX or Windows path, including a network share.
    """
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def redact_token(match: re.Match[str]) -> str:
    """
    Replace a message token that is an absolute path with a placeholder, keeping its quotes or punctuation.

    Args:
        match (re.Match[str]): A quoted string or a run of non-space text from the message.

    Returns:
        str: The token, with an absolute path replaced by <path>.
    """
    # A quoted token is a path when its whole content is one.
    token = match.group()
    if len(token) > 1 and token[0] in "'\"" and token[-1] == token[0]:
        return f"{token[0]}<path>{token[0]}" if is_absolute_path(token[1:-1]) else token

    # An unquoted token is a path once the punctuation around it is set aside.
    core = token.strip(PATH_PUNCTUATION)
    return token.replace(core, "<path>", 1) if core and is_absolute_path(core) else token


def redact_paths(text: str) -> str:
    """
    Replace absolute paths, which can contain the user name, with a placeholder before the text is saved.
    """
    return MESSAGE_TOKEN_PATTERN.sub(redact_token, text)


def describe_failure(stage: RoundStage, error: BaseException) -> RoundFailure:
    """
    Summarize an exception as a round failure with its class name and the first line of its message, paths redacted.
    """
    return RoundFailure(stage=stage, error=type(error).__name__, message=redact_paths(first_line(error)))
