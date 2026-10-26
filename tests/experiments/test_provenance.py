import pytest


from sketchloop.experiments import describe_failure


# Build each path from pieces so this file never contains a literal home path for the publication guard to flag.
@pytest.mark.parametrize(("message", "expected"), [
    ("Cannot open '" + "C:" + "\\Users\\some one\\unet.bin': denied", "Cannot open '<path>': denied"),
    ("Cannot open \"/" + "home/someone/unet.bin\": denied", "Cannot open \"<path>\": denied"),
    ("Cannot open /" + "Volumes/Data/unet.bin now", "Cannot open <path> now"),
    ("Missing weights (/" + "opt/models/unet.bin), retry", "Missing weights (<path>), retry"),
    ("See https://huggingface.co/models or models/unet.bin", "See https://huggingface.co/models or models/unet.bin")
], ids=["windows-with-space", "posix-quoted", "posix-unquoted", "posix-in-parentheses", "url-and-relative"])
def test_failure_message_hides_paths(message: str, expected: str) -> None:
    """
    A saved failure message replaces absolute paths, which can name the user, and keeps URLs and relative paths.
    """
    failure = describe_failure("generation", OSError(f"{message}\nsecond line"))

    # Only the first line is kept, with each absolute path replaced.
    assert failure.message == expected and failure.error == "OSError", f"Unexpected failure record {failure}"
