"""
Tests of the model download script's source list and its one-time conversion of pickled weights to safetensors.
"""

import torch

from pathlib import Path
from safetensors.torch import load_file

from download_models import DOWNLOAD_SOURCES, convert_pickled_weights
from sketchloop.backends.t2i_adapter import T2I_ADAPTER_MODEL


def test_download_lists_adapter() -> None:
    """
    The download script fetches the T2I-Adapter sketch weights at their pinned revision.
    """
    assert T2I_ADAPTER_MODEL in DOWNLOAD_SOURCES, "The adapter weights must be downloaded"


def test_convert_pickled_weights(tmp_path: Path) -> None:
    """
    A pickled .bin becomes an equal safetensors file once, and a second run leaves the converted file alone.
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
