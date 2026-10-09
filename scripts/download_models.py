from pathlib import Path

from sketchloop.diffusers_backend import MODEL_SOURCES, first_line


ROOT = Path(__file__).resolve().parents[1]
BYTES_PER_GIGABYTE = 1e9


def folder_bytes(folder: Path) -> int:
    """
    Sum the sizes of a model folder's files, ignoring the download cache.
    """
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and ".cache" not in path.parts)


def main() -> int:
    """
    Download the diffusers backend's models at their pinned revisions into models/, skipping files already present.

    Returns:
        int: Process exit code, 0 when every model is present.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('Install the generation extra first: pip install -e ".[generation]"')
        return 1

    # Download each repository, where files whose recorded metadata already matches are skipped.
    total_bytes = 0
    downloaded_bytes = 0
    for source in MODEL_SOURCES:
        folder = ROOT / "models" / source.folder
        print(f"{source.repo_id} -> models/{source.folder}", flush=True)
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
