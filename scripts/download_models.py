import argparse

from pathlib import Path

from sketchloop.diffusers_backend import MODEL_SOURCES, first_line
from sketchloop.settings import SettingsError, find_settings_file, load_settings


BYTES_PER_GIGABYTE = 1e9


def folder_bytes(folder: Path) -> int:
    """
    Sum the sizes of a model folder's files, ignoring the download cache.
    """
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and ".cache" not in path.parts)


def find_models_dir(config: Path | None) -> Path:
    """
    Read the models folder from the same settings file the sketchloop command uses, and say which file and folder apply.

    Args:
        config (Path | None): Settings file named by --config, or None to look for sketchloop.toml.

    Returns:
        Path: The models folder from generator.diffusers.models_dir.
    """
    settings_file = find_settings_file(config)
    if settings_file is not None:
        print(f"Settings: {settings_file.as_posix()}")

    models_dir = load_settings(settings_file).generator.diffusers.models_dir
    print(f"Models folder: {models_dir.as_posix()}", flush=True)
    return models_dir


def main(argv: list[str] | None = None) -> int:
    """
    Download the diffusers backend's models at their pinned revisions into the models folder, skipping files already present.

    Args:
        argv (list[str] | None, optional): Command-line arguments. Defaults to the process arguments.

    Returns:
        int: Process exit code, 0 when every model is present.
    """
    parser = argparse.ArgumentParser(description="Download the diffusers backend's models into generator.diffusers.models_dir.")
    parser.add_argument("--config", type=Path, help="settings file, by default sketchloop.toml in the working folder when present")  # noqa: E501
    args = parser.parse_args(argv)

    # Import the download library only after parsing, so --help works before installation.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install SketchLoop first: pip install -e .")
        return 1

    # Use the folder the backend reads, so the download and the backend always agree.
    try:
        models_dir = find_models_dir(args.config)
    except SettingsError as error:
        print(error)
        return 1

    # Download each repository, where files whose recorded metadata already matches are skipped.
    total_bytes = 0
    downloaded_bytes = 0
    for source in MODEL_SOURCES:
        folder = models_dir / source.folder
        print(f"Downloading {source.repo_id} to {folder.as_posix()}.", flush=True)
        before_bytes = folder_bytes(folder) if folder.is_dir() else 0
        try:
            snapshot_download(source.repo_id, revision=source.revision, local_dir=folder, allow_patterns=source.allow_patterns)
        except Exception as error:
            print(f"Download of {source.repo_id} failed: {first_line(error)} Check the network and rerun.")
            return 1

        after_bytes = folder_bytes(folder)
        total_bytes += after_bytes
        downloaded_bytes += after_bytes - before_bytes

    print(f"Total: {total_bytes / BYTES_PER_GIGABYTE:.2f} GB, downloaded now: {downloaded_bytes / BYTES_PER_GIGABYTE:.2f} GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
