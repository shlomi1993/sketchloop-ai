# Model: T2I-Adapter sketch for SD 1.5

Mou et al., *T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models*, AAAI 38(5), 4296-4304 (2024), [doi:10.1609/aaai.v38i5.28226](https://doi.org/10.1609/aaai.v38i5.28226), preprint [arXiv:2302.08453](https://arxiv.org/abs/2302.08453) (v2, 2023-03-20, CC BY 4.0). Weights: [TencentARC/t2iadapter_sketch_sd15v2](https://huggingface.co/TencentARC/t2iadapter_sketch_sd15v2), revision `a73a564e35b0956deffd164e1ee463f967c3f952` (last modified 2023-08-01, the only branch and no safetensors conversion ref). Also read: the [Diffusers T2I-Adapter API page](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/adapter), the [Diffusers LCM guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm), the [original repository](https://github.com/TencentARC/T2I-Adapter/tree/SD) (`SD` branch), and the installed `diffusers` 0.40.0 source with `torch` 2.14.1. Read 2026-11-18.

## Question and decision informed

Can the SD 1.5 sketch T2I-Adapter become a second real backend beside [SD 1.5 + ControlNet scribble](model-controlnet-scribble.md), selected through settings, to show that a real model can be added without changing orchestration or UI (R02, R08)? What does the backend need for loading, input, controls, fast and quality modes, devices, memory, and records?

## How it works (plain words)

A small convolutional network (77M parameters) turns the sketch into four feature maps, one per UNet encoder resolution, and adds them to the frozen SD UNet's encoder features. Unlike ControlNet, it is not a copy of the UNet and does not depend on the timestep, so it runs once per image, not once per denoising step.

## Findings

Labels: **Verified** means read in the named source or confirmed by running code. **Inference** is my reasoning. Nothing here comes from the proposal, which names no adapter model.

### Repository, files, and license

- Verified (Hub API): the repo holds `config.json` (208 bytes) and `diffusion_pytorch_model.bin` (308,017,523 bytes), plus a README and three example images. **There is no safetensors file** on any ref. The `diffusers/t2iadapter_sketch_sd15v2` original has the same `.bin` only.
- Verified (Hub API and card): license Apache 2.0, not gated, base model `runwayml/stable-diffusion-v1-5`. The project's `stable-diffusion-v1-5/stable-diffusion-v1-5` is a mirror of that repo ([SD 1.5 note](model-sd15.md)), so the existing `models/stable-diffusion-v1-5` folder can be reused.
- Verified (`config.json`): `T2IAdapter`, `adapter_type: full_adapter`, `channels: [320, 640, 1280, 1280]`, **`in_channels: 1`** (one grayscale channel), `num_res_blocks: 2`.
- Verified (installed `diffusers/models/model_loading_utils.py`): `.bin` files load through `torch.load(..., weights_only=True)`, which refuses arbitrary pickled code.
- Inference: the combined system stays bound by SD 1.5's CreativeML OpenRAIL-M, so the Apache adapter does not change the license question.
- Inference: the card's example code loads `t2iadapter_zoedepth_sd15v1`, not the sketch adapter, so it is a copy error and not a usage hint.

### Pipeline and call (diffusers 0.40.0)

- Verified (installed source): `diffusers.T2IAdapter` and `diffusers.StableDiffusionAdapterPipeline(vae, text_encoder, tokenizer, unet, adapter, scheduler, safety_checker, feature_extractor, requires_safety_checker=True)`. The call takes `prompt`, `image`, `height`, `width`, `num_inference_steps` (default 50), `guidance_scale` (default 7.5), `negative_prompt`, `generator`, `output_type`, `cross_attention_kwargs`, and **`adapter_conditioning_scale`** (float, default 1.0), which multiplies the adapter features before they are added to the UNet.
- Verified (installed source): there is no lower or upper bound on `adapter_conditioning_scale`, and no recommended range in the card, the paper, or the docs. The SDXL LCM example in the Diffusers guide uses 0.8.
- Verified (installed source): the SD 1.5 pipeline has **no `adapter_conditioning_factor`**, which only the SDXL pipeline has, so the adapter acts on every step. The original repository's sketch example uses `--cond_tau 0.5`, which applies the adapter only for the first half of the steps. That behavior is not reproducible through this pipeline without a custom loop.
- Verified (installed source and API page): `StableDiffusionAdapterPipeline` subclasses only `DiffusionPipeline`, `StableDiffusionMixin`, and `FromSingleFileMixin`. **It has no `load_lora_weights` or `fuse_lora`**, unlike the ControlNet and SDXL adapter pipelines.
- Verified (installed source): `image` must be a PIL image or a torch tensor. A NumPy array, which the ControlNet backend passes today, falls through `_preprocess_adapter_image` unchanged and then fails on `.to(device)`. A PIL image is resized to `(width, height)` with Lanczos, and a tensor is used as is.
- Verified (installed source): `height` and `width` must be divisible by 8. The adapter computes its features once before the denoising loop and repeats them for the classifier-free guidance batch, so the strokes also condition the unconditional or negative branch.
- Verified (run locally, random weights): a 1-channel `T2IAdapter` with this config has 77.0M parameters, and at 512x360 its features are 45x64, 23x32, 12x16, and 6x8, matching the SD 1.5 UNet. A UNet forward pass with those residuals succeeds, so the backend's existing 512-long-side sizing works.

### Input polarity

- Verified (card): the SD 1.4 sketch adapter in the same family takes "white outlines on a black background", and the examples convert photos with the PiDiNet edge detector, which outputs white edges.
- Verified (example image measured locally): the card's `images/sketch.png` at the pinned revision is 512x640 with a mean of 37.7 of 255 and 13% bright pixels, so it is white strokes on black.
- Verified (original repository, `get_cond_sketch`): a sketch input is scaled to [0, 1] and thresholded at 0.5 with no inversion, so strokes are the bright pixels.
- Inference: the existing `scribble_conditioning`, which makes 0 or 255 white-on-black strokes at 512 px with sides divisible by 8, is already the right input. Only the container changes, a 1-channel PIL `L` image instead of a 3-channel NumPy batch.

### Paper

- Verified (arXiv v2, §1 and §3.3): about 77M parameters and about 300 MB of storage. Features enter the UNet encoder at four scales. Adapters "only need one inference during the diffusion process".
- Verified (§3.4): training samples timesteps from a cubic distribution that favors early, noisy steps, because guidance added in middle and late steps had little effect.
- Verified (§4.1): the paper's sketch adapter was trained on SD 1.4 with COCO 2017 (about 164K images), PiDiNet edges thresholded at 0.5, 512x512, 10 epochs, on 4 V100 GPUs. No training details exist for the `sd15v2` checkpoint in the card, paper, or repository README.
- Verified (Table 1, COCO validation): text plus sketch scored FID 17.36 and CLIP score 0.2666, against 24.68 and 0.2648 for SD text only. The paper has **no numeric comparison with ControlNet** and no user study. These scores are research signals about distribution and text match, not measures of design quality.
- Verified (§4.3.3): adapters trained on one SD version transfer to models fine-tuned from it, such as SD 1.4 to SD 1.5.
- Verified ([Diffusers LCM guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm)): T2I-Adapter "is faster than ControlNet but the results may be slightly worse".

### LCM-LoRA fast mode

- Verified (Diffusers LCM guide): "LCMs are compatible with adapters like LoRA, ControlNet, T2I-Adapter". The only T2I-Adapter example is **SDXL** (`StableDiffusionXLAdapterPipeline`, `LCMScheduler`, `load_lora_weights`, 4 steps, guidance 1.5, `adapter_conditioning_scale=0.8`). No official SD 1.5 adapter example with LCM-LoRA exists, and a web search found no issue thread about the combination.
- Verified (local `models/lcm-lora-sdv1-5`): the LoRA file is in Kohya format (`lora_unet_*`, 834 tensors, UNet only), so it must go through Diffusers' converter.
- Verified (run locally with `StableDiffusionLoraLoaderMixin` classmethods on a randomly initialized SD 1.5 UNet built from the local config): `lora_state_dict` converts the file to 556 `unet.*` tensors with 278 alphas, `load_lora_into_unet` loads them, and `unet.fuse_lora()` changes the weights. This is the workaround for the missing pipeline method.
- Verified (installed source): CFG runs when `guidance_scale > 1` and the UNet has no guidance embedding, which holds for SD 1.5 with the LoRA. So guidance 1.5 doubles the batch and enables the negative prompt, as in the ControlNet backend.
- Inference: the adapter and the LoRA touch different parts of the computation, the adapter adds encoder residuals and the LoRA changes UNet weights, so they should combine. 4-step LCM sampling starts at the noisiest step, where the adapter was trained to matter most. Image quality at 4 steps is **unverified** and needs a real run.

### Devices and precision

- Verified (run locally on the M2 Pro, random weights): the adapter's operations (`PixelUnshuffle`, convolutions, `AvgPool2d` with `ceil_mode`) run on MPS in float16 and float32 with no NaN. float16 differs from CPU float32 by 0.1% relative and float32 matches exactly. The forward pass takes about 0.04 s on CPU.
- Verified (installed source): `from_pipe` casts the new pipeline to float32 unless `dtype` is passed, and casting a model with `.to(dtype)` logs a warning advising `dtype=` in `from_pretrained`.
- Inference: CPU float32 uses only standard operations, and the UNet forward pass with adapter residuals already ran on CPU above. float16 with the real trained weights on MPS is unverified, but the adapter is small and the SD 1.5 float16 path is already measured in [ADR 0018](../decisions/0018-float16-on-the-apple-gpu.md).

## Speed and memory (inference unless marked)

| Added to the loaded SD 1.5 base | ControlNet scribble (current) | T2I-Adapter sketch |
| --- | --- | --- |
| Weights on disk | 1.45 GB fp32 safetensors (verified) | 0.31 GB `.bin` (verified) |
| Memory, float16 / float32 | about 0.72 / 1.45 GB | about 0.15 / 0.31 GB |
| Extra compute | runs on every step, about 40% more per step | runs once per image, about 0.04 s on CPU (verified, random weights) |
| Round time vs. ControlNet | baseline | about 70-80% of ControlNet's generation time |

Against the [T08 baseline](../results/t08-latency-baseline.md) (float16 on the M2 Pro, 2 candidates: 23.3 s in quality mode and 5.2 s in fast mode), expect about 17-19 s and 4-4.5 s. Fixed costs (text encoder, VAE decode) weigh more in fast mode. Measure with `sketchloop benchmark`.

The two backends do not share weights unless they are built from one pipeline with `from_pipe`. The UI holds one generator, so switching backends reloads the base. Loading both at once would duplicate the base (about 2.3 GB in float16). Sharing the UNet through `from_pipe` would also share the fused LCM-LoRA, so both pipelines would have to be in the same mode.

## Options compared

| Option | Diff size | Fast mode | Weights | Notes |
| --- | --- | --- | --- | --- |
| A. Subclass `DiffusersSketchGenerator`, override loading, kwargs, sources, identity | Smallest | LoRA via mixin classmethods | `.bin` with `weights_only=True` | Reuses modes, controls, seeds, runner, records. Recommended. |
| B. Separate `Generator` reusing module helpers | Larger | Same | Same | Cleaner if the two backends diverge later. |
| C. Build `StableDiffusionPipeline`, `load_lora_weights` + `fuse_lora`, then `StableDiffusionAdapterPipeline.from_pipe(..., adapter=, dtype=)` | Medium | Public pipeline API | Same | Uses documented methods only, but builds two pipeline objects and needs `dtype` passed. |
| D. Convert the `.bin` to safetensors in `scripts/download_models.py` | Adds a step | n/a | safetensors after first run | Keeps the "safetensors only" rule of ADR 0009 for loading, but the conversion still unpickles once. |

## Recommendation and confidence

Add a `t2i-adapter` backend as option A, with the LoRA loaded through `StableDiffusionLoraLoaderMixin.lora_state_dict` and `load_lora_into_unet`, then `pipeline.unet.fuse_lora()`. Load the `.bin` directly, since diffusers 0.40 uses `weights_only=True`, and record the weight format. The architect should note this departure from ADR 0009's safetensors-only rule in an ADR.

Confidence is **high** for the repo, revision, license, classes, arguments, input polarity, and loading path, which were checked in source and by running code. Confidence is **medium** for fast-mode image quality and real float16 weights on MPS, and **low** for the speed estimates, which a real run must confirm. No blocker was found. The two things that break a copy-paste of the ControlNet backend are the missing `load_lora_weights` and the NumPy image input.

## Implementation guidance

**Target modules:** `src/sketchloop/backends/t2i_adapter.py` (new), registration in `src/sketchloop/components.py` for the factory and the rerun adapter-name map, the new source in `scripts/download_models.py`, and `examples/sketchloop.toml`. Whether it reads `[generator.diffusers]` options or its own section is an architect decision.

**Pinned versions:** `diffusers==0.40.0`, `torch==2.14.1`, `peft==0.21.2` (needed by `load_lora_into_unet`), and Pillow 12.3.0 installed as a diffusers dependency. If the backend imports PIL directly, declare `pillow` in `pyproject.toml`, or pass a torch tensor built in the runner instead.

```python
T2I_ADAPTER_MODEL = ModelSource(
    folder="t2iadapter_sketch_sd15v2",
    repo_id="TencentARC/t2iadapter_sketch_sd15v2",
    revision="a73a564e35b0956deffd164e1ee463f967c3f952",
    allow_patterns=("config.json", "diffusion_pytorch_model.bin"),
)

# Loading, inside _load_runner (reuse BASE_MODEL and LCM_LORA_MODEL folders).
adapter = diffusers.T2IAdapter.from_pretrained(adapter_folder, dtype=dtype)
pipeline = diffusers.StableDiffusionAdapterPipeline.from_pretrained(
    str(base_folder), adapter=adapter, dtype=dtype,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False,
)
pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
if mode == GenerationMode.FAST:
    loader = diffusers.loaders.StableDiffusionLoraLoaderMixin
    state_dict, alphas = loader.lora_state_dict(str(lcm_folder), weight_name=LCM_LORA_WEIGHTS)
    loader.load_lora_into_unet(state_dict, alphas, unet=pipeline.unet)
    pipeline.unet.fuse_lora()
pipeline.to(device)

# Call kwargs: same conditioning, 1-channel PIL image, renamed scale argument.
conditioning = scribble_conditioning(sketch_payload)  # uint8, 0 or 255, white on black
pipeline_kwargs = {
    "prompt": prompt,
    "image": PIL.Image.fromarray(conditioning),  # 2-D uint8 gives mode "L"
    "width": width,
    "height": height,
    "num_inference_steps": steps,
    "guidance_scale": guidance_scale,
    "adapter_conditioning_scale": controls["conditioning_scale"],
}
```

**Modes:** reuse `MODE_SETTINGS`. Fast mode uses `LCMScheduler`, 4 steps, and guidance 1.5, with the guidance bounds of the [LCM-LoRA note](model-lcm-lora.md). Quality mode uses `UniPCMultistepScheduler`, 20 steps, and guidance 7.5. The card relies on the pipeline defaults of 50 steps and guidance 7.5 with the base's PNDM scheduler, so 20 UniPC steps is an inference to confirm visually. Keep the control name `conditioning_scale` (0-2, default 1.0) so the UI slider and records stay uniform, and map it to `adapter_conditioning_scale`. Values about 0.7-0.9 should give looser variants, which is also an inference.

**Parameters to record:** `adapter_name` such as `sketchloop.t2i-adapter`, the model ID with the base, adapter, and LoRA revisions, as `_identify_backend` already builds, the adapter weight format (`bin`), conditioning polarity and channel count, `conditioning_scale` as `adapter_conditioning_scale`, the fact that the adapter acts on all steps (no `cond_tau` equivalent), and the existing mode, device, dtype, scheduler, resolution, seeds, and safety checker settings. `local_revision` works unchanged, because `snapshot_download` writes the same metadata.

**Pitfalls:**
- Do not call `pipeline.load_lora_weights`, which raises `AttributeError` on this pipeline.
- Do not pass the NumPy batch the ControlNet backend builds, because the pipeline accepts PIL or tensor only.
- Do not pass a 3-channel image, because the adapter expects `in_channels=1`.
- Pass `dtype=` to `from_pretrained` rather than `.to(dtype)`, and pass `dtype` to `from_pipe` if option C is used, since it otherwise casts to float32.
- The base folder's `model_index.json` names `StableDiffusionPipeline`. Loading it with the adapter class is the documented card usage, but the adapter must be passed explicitly.
- Never claim exact replay across devices. The adapter is deterministic, but the SD sampling limits of the [SD 1.5 note](model-sd15.md) still apply.

**Tests (offline, fake backend strategy):** inject a stub `PipelineRunner`, as the ControlNet tests do, and assert that the kwargs contain `adapter_conditioning_scale` and no `controlnet_conditioning_scale`, that `image` is a PIL `L` image of `(width, height)` with white strokes on black from a synthetic black-on-white sketch, and that sides are divisible by 8. Assert the model sources include the adapter, exclude the ControlNet, and include the LoRA only in fast mode, that a missing adapter folder raises `GenerationBackendError` before imports, that `describe()` names the adapter, and that the settings and rerun map resolve `t2i-adapter`. Assert `scripts/download_models.py` lists the new source. One `model`-marked test loads the real weights, generates one 4-step image at 512x360 on the available device, and checks for no NaN or all-black output, with float16 on MPS.

## Open questions for the owner

- Accept loading the pickle `.bin` with `weights_only=True`, or convert it to safetensors once in the download script (option D)?
- Should the second backend share `[generator.diffusers]` settings (mode, precision, models folder) or get its own section?
- After a first real run, which default conditioning scale reads better for physical sketches, 1.0 or about 0.8?
