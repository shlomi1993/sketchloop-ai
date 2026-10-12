from enum import StrEnum
from typing import Final

from sketchloop.diffusers_backend import DiffusersSketchGenerator, GenerationMode
from sketchloop.fakes import FakeGenerator
from sketchloop.generation import Generator


class GenerationBackend(StrEnum):
    """
    Generation backends the command line and the UI can create.
    """
    FAKE = "fake"
    DIFFUSERS = "diffusers"


# Backends a rerun can recreate, by the adapter name a round recorded.
RERUNNABLE_BACKENDS: Final = {
    "sketchloop.fake": GenerationBackend.FAKE,
    "sketchloop.diffusers": GenerationBackend.DIFFUSERS
}


def make_generator(backend: GenerationBackend, mode: GenerationMode | None = None) -> Generator:
    """
    Create a generation backend, without loading any model yet.

    Args:
        backend (GenerationBackend): Backend to create.
        mode (GenerationMode | None, optional): Diffusers mode, or None for the device default. Defaults to None.

    Returns:
        Generator: The generation backend.
    """
    if backend == GenerationBackend.FAKE:
        return FakeGenerator()

    return DiffusersSketchGenerator(mode=mode)
