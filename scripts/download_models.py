"""
Download the pinned model weights into the models folder the settings name, converting pickled weights to safetensors.
"""

import argparse

from pathlib import Path

from sketchloop.backends.diffusers import MODEL_SOURCES, ModelSource, first_line
from sketchloop.backends.t2i_adapter import T2I_ADAPTER_MODEL
from sketchloop.components import EVALUATOR_MODELS
from sketchloop.settings import Settings, SettingsError, find_settings_file, load_settings
from sketchloop.suggestion import FLORENCE_MODEL


BYTES_PER_GIGABYTE = 1e9
DOWNLOAD_SOURCES = (*MODEL_SOURCES, T2I_ADAPTER_MODEL, FLORENCE_MODEL)

# Converted names that differ from the pickled file's, since transformers loads model.safetensors.
CONVERTED_NAMES = {"pytorch_model.bin": "model.safetensors"}


def folder_bytes(folder: Path) -> int:
    """
    Sum the sizes of a model folder's files, ignoring the download cache.
    """
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and ".cache" not in path.parts)


def convert_pickled_weights(folder: Path) -> None:
    """
    Convert each pickled .bin weights file in a model folder to safetensors once, so the backends load only safetensors.

    Args:
        folder (Path): Downloaded model folder.
    """
    # Import torch only here, since --help and the other models do not need it.
    import torch

    from safetensors.torch import save_file

    # Skip a file already converted, and write through a temporary name so an interrupted run leaves no partial file.
    for pickled in folder.glob("*.bin"):
        converted = folder / CONVERTED_NAMES.get(pickled.name, pickled.with_suffix(".safetensors").name)
        if converted.exists():
            continue

        print(f"Converting {pickled.name} to {converted.name}.", flush=True)
        state_dict = torch.load(pickled, map_location="cpu", weights_only=True)
        partial = converted.with_name(f"{converted.name}.tmp")
        save_file({name: tensor.contiguous() for name, tensor in state_dict.items()}, str(partial))
        partial.replace(converted)


def find_settings(config: Path | None) -> Settings:
    """
    Read the same settings file the sketchloop command uses, and say which file and models folder apply.

    Args:
        config (Path | None): Settings file named by --config, or None for built-in defaults.

    Returns:
        Settings: The settings, whose models.dir is where the models go.
    """
    settings_file = find_settings_file(config)
    if settings_file is not None:
        print(f"Settings: {settings_file.as_posix()}")

    settings = load_settings(settings_file)
    print(f"Models folder: {settings.models.dir.as_posix()}", flush=True)
    return settings


def list_download_sources(settings: Settings, evaluator_names: list[str]) -> tuple[ModelSource, ...]:
    # Add the evaluator the settings name and the requested ones, once each, since evaluator weights are large.
    wanted = dict.fromkeys(name for name in (settings.evaluator.name, *evaluator_names) if name in EVALUATOR_MODELS)
    return (*DOWNLOAD_SOURCES, *(EVALUATOR_MODELS[name] for name in wanted))


def main(argv: list[str] | None = None) -> int:
    """
    Download the generation and prompt suggestion models, and the requested evaluators', at their pinned revisions.

    Args:
        argv (list[str] | None, optional): Command-line arguments. Defaults to the process arguments.

    Returns:
        int: Process exit code, 0 when every model is present.
    """
    parser = argparse.ArgumentParser(description="Download the generation, prompt suggestion, and chosen evaluator models into the models.dir folder, skipping files already present.")  # noqa: E501
    parser.add_argument("--config", type=Path, help="settings file to read explicitly. Omitted means built-in defaults")  # noqa: E501
    parser.add_argument("--evaluator", action="append", default=[], choices=list(EVALUATOR_MODELS), help="also download this evaluator's model, and repeat the option for more. The evaluator the settings file names is always downloaded")  # noqa: E501
    args = parser.parse_args(argv)

    # Import the download library only after parsing, so --help works before installation.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install SketchLoop first: pip install -e .")
        return 1

    # Use the folder the backend reads, so the download and the backend always agree.
    try:
        settings = find_settings(args.config)
    except SettingsError as error:
        print(error)
        return 1

    models_dir = settings.models.dir

    # Download each repository, where files whose recorded metadata already matches are skipped.
    total_bytes = 0
    downloaded_bytes = 0
    for source in list_download_sources(settings, args.evaluator):
        folder = models_dir / source.folder
        print(f"Downloading {source.repo_id} to {folder.as_posix()}.", flush=True)
        before_bytes = folder_bytes(folder) if folder.is_dir() else 0
        try:
            snapshot_download(source.repo_id, revision=source.revision, local_dir=folder, allow_patterns=source.allow_patterns)
        except Exception as error:
            print(f"Download of {source.repo_id} failed: {first_line(error)} Check the network and rerun.")
            return 1

        # Keep the pickled original, so the next run skips its download, but give the backend a safetensors copy.
        try:
            convert_pickled_weights(folder)
        except Exception as error:
            print(f"Converting {source.repo_id} failed: {first_line(error)} Delete {folder.as_posix()} and rerun.")
            return 1

        after_bytes = folder_bytes(folder)
        total_bytes += after_bytes
        downloaded_bytes += after_bytes - before_bytes

    print(f"Total: {total_bytes / BYTES_PER_GIGABYTE:.2f} GB, downloaded now: {downloaded_bytes / BYTES_PER_GIGABYTE:.2f} GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
