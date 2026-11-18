"""
Second real generation backend running Stable Diffusion 1.5 with the T2I-Adapter sketch model through diffusers.
"""

import numpy as np

from PIL import Image
from typing import Any, ClassVar, Final

from sketchloop.backends.diffusers import (BASE_MODEL, LCM_LORA_MODEL, LCM_LORA_WEIGHTS, DiffusersSketchGenerator,
                                           GenerationMode, ModelSource)
from sketchloop.domain import ControlValue


# The repository has only a pickled .bin, which scripts/download_models.py converts once to safetensors (ADR 0009).
ADAPTER_WEIGHTS: Final = "diffusion_pytorch_model.safetensors"
T2I_ADAPTER_MODEL: Final = ModelSource(
    folder="t2iadapter_sketch_sd15v2",
    repo_id="TencentARC/t2iadapter_sketch_sd15v2",
    revision="a73a564e35b0956deffd164e1ee463f967c3f952",
    allow_patterns=("config.json", "diffusion_pytorch_model.bin"),
)


class T2IAdapterSketchGenerator(DiffusersSketchGenerator):
    """
    Stable Diffusion 1.5 with the T2I-Adapter sketch model, sharing the diffusers backend's modes, controls, and records.
    """

    # Saved in every round record and looked up by rerun, so never rename it.
    ADAPTER_NAME: ClassVar[str] = "sketchloop.t2i-adapter"
    BACKEND_NAME: ClassVar[str] = "t2i-adapter"  # Name in generator.backend and --backend
    MODEL_LABEL: ClassVar[str] = "Stable Diffusion 1.5 and T2I-Adapter sketch"

    def _list_model_sources(self) -> tuple[ModelSource, ...]:
        sources = (BASE_MODEL, T2I_ADAPTER_MODEL)
        return (*sources, LCM_LORA_MODEL) if self.mode == GenerationMode.FAST else sources

    def _build_pipeline(self, diffusers: Any, dtype: Any) -> Any:
        # Load only the converted safetensors weights, never the pickled .bin, and skip the safety checker as the base does.
        adapter_folder = str(self._model_dir / T2I_ADAPTER_MODEL.folder)
        adapter = diffusers.T2IAdapter.from_pretrained(adapter_folder, dtype=dtype, use_safetensors=True)
        return diffusers.StableDiffusionAdapterPipeline.from_pretrained(
            str(self._model_dir / BASE_MODEL.folder),
            adapter=adapter,
            dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )

    def _fuse_lcm_lora(self, diffusers: Any, pipeline: Any) -> None:
        # The adapter pipeline has no load_lora_weights, so load the LoRA into its UNet through the loader's class methods.
        loader = diffusers.loaders.StableDiffusionLoraLoaderMixin
        state_dict, network_alphas = loader.lora_state_dict(str(self._model_dir / LCM_LORA_MODEL.folder),
                                                            weight_name=LCM_LORA_WEIGHTS)
        loader.load_lora_into_unet(state_dict, network_alphas, unet=pipeline.unet)
        pipeline.unet.fuse_lora()

    def _build_conditioning_kwargs(self, conditioning: np.ndarray, conditioning_scale: ControlValue) -> dict[str, Any]:
        # The adapter takes one grayscale channel as a PIL or tensor image, and a 2-D uint8 array becomes a mode L image.
        return {"image": Image.fromarray(conditioning), "adapter_conditioning_scale": conditioning_scale}
