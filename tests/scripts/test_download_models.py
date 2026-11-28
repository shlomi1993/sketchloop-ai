"""
Tests of the model download script's source lists, evaluator option, and one-time conversion of pickled weights to safetensors.
"""

import huggingface_hub
import pytest
import torch

from pathlib import Path
from safetensors.torch import load_file

from download_models import DOWNLOAD_SOURCES, convert_pickled_weights
from download_models import main as download_main
from sketchloop.backends.t2i_adapter import T2I_ADAPTER_MODEL
from sketchloop.evaluators.clip import CLIP_B32_MODEL, CLIP_L14_MODEL


def test_convert_pickled_weights(tmp_path: Path) -> None:
    """
    A pickled .bin becomes an equal safetensors file once under the name its library loads, and a rerun leaves it alone.
    """
    weights = {"conv.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).t(), "conv.bias": torch.ones(2)}
    torch.save(weights, tmp_path / "diffusion_pytorch_model.bin")
    convert_pickled_weights(tmp_path)
    converted = tmp_path / "diffusion_pytorch_model.safetensors"

    # The safetensors file holds the same tensors, and no temporary file is left behind.
    loaded = load_file(converted)
    assert loaded.keys() == weights.keys(), f"Expected the same tensor names, got {sorted(loaded)}"
    assert all(torch.equal(loaded[name], tensor) for name, tensor in weights.items()), "Converted tensors must be equal"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["diffusion_pytorch_model.bin", converted.name], "Unexpected files"

    # A second run skips the existing file instead of converting again.
    converted.write_bytes(b"kept")
    convert_pickled_weights(tmp_path)
    assert converted.read_bytes() == b"kept", "An existing safetensors file must not be converted again"

    # A transformers pytorch_model.bin becomes model.safetensors, the name transformers loads.
    transformers_folder = tmp_path / "transformers"
    transformers_folder.mkdir()
    torch.save({"proj.weight": torch.ones(2, 2)}, transformers_folder / "pytorch_model.bin")
    convert_pickled_weights(transformers_folder)
    assert (transformers_folder / "model.safetensors").is_file(), f"Expected model.safetensors in {transformers_folder}"


@pytest.mark.usefixtures("isolated_cwd")
def test_download_evaluators_on_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Evaluator weights are fetched only for the evaluator the settings name and those passed with --evaluator, once each.
    """
    settings_evaluator, expected = "clip-b32", [CLIP_B32_MODEL, CLIP_L14_MODEL]
    flags = ["--evaluator", "clip-l14", "--evaluator", "clip-b32"]
    requested: list[tuple[str, str, Path]] = []

    def record_download(repo_id: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...]) -> None:
        """
        Record the requested repository instead of downloading it.
        """
        requested.append((repo_id, revision, local_dir))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", record_download)
    config = tmp_path / "sketchloop.toml"
    settings_lines = ["[models]", 'dir = "models"', "[evaluator]", f'name = "{settings_evaluator}"']
    config.write_text("\n".join(settings_lines), encoding="utf-8")
    exit_code = download_main(["--config", str(config), *flags])

    # The generation, adapter, and suggestion models always download, followed by exactly the requested evaluators.
    repo_ids = [repo_id for repo_id, revision, local_dir in requested]
    expected_ids = [source.repo_id for source in (*DOWNLOAD_SOURCES, *expected)]
    assert exit_code == 0, f"The download must succeed, got exit code {exit_code}"
    assert repo_ids == expected_ids, f"Expected downloads {expected_ids}, got {repo_ids}"
    assert T2I_ADAPTER_MODEL in DOWNLOAD_SOURCES, "The adapter weights must be downloaded"

    # Each evaluator lands in its own folder under the configured models folder at its pinned revision.
    evaluator_downloads = requested[len(DOWNLOAD_SOURCES):]
    expected_downloads = [(model.repo_id, model.revision, tmp_path / "models" / model.folder) for model in expected]
    assert evaluator_downloads == expected_downloads, f"Expected {expected_downloads}, got {evaluator_downloads}"
