from pathlib import Path
from typing import Final

from sketchloop.backends.diffusers import DiffusersSketchGenerator, GenerationMode
from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import GenerationBackend
from sketchloop.generation import Generator


# Backends a rerun can recreate, by the adapter name a round recorded.
RERUNNABLE_BACKENDS: Final = {
    FakeGenerator.ADAPTER_NAME: GenerationBackend.FAKE,
    DiffusersSketchGenerator.ADAPTER_NAME: GenerationBackend.DIFFUSERS
}


def make_generator(backend: GenerationBackend, mode: GenerationMode | None = None, *,
                   models_dir: Path = Path("models")) -> Generator:
    """
    Create a generation backend, without loading any model yet.

    Args:
        backend (GenerationBackend): Backend to create.
        mode (GenerationMode | None, optional): Diffusers mode, or None for the device default. Defaults to None.
        models_dir (Path, optional): Folder with the downloaded diffusers models, ignored by the fake. Defaults to models.

    Returns:
        Generator: The generation backend.
    """
    if backend == GenerationBackend.FAKE:
        return FakeGenerator()

    return DiffusersSketchGenerator(models_dir, mode=mode)
