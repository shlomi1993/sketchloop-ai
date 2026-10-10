# ADR 0009: Diffusers backend, sketch bytes in the generator contract, and the generation extra

Status: accepted

## Context

T03 adds the first real generative backend (R02, R03, R09), chosen in [ADR 0006](0006-first-backend-store-and-ui.md) and described in the [first-model note](../research/first-model-options.md): Stable Diffusion 1.5 with the ControlNet v1.1 scribble model, with an LCM-LoRA fast mode for CPU. The development machines are a Windows laptop without a GPU and an Apple M2 Pro Mac. The [ADR 0005](0005-domain-records-and-generator-contract.md) contract gave `Generator.generate` only the request, whose `ImageRef` names the sketch but carries no bytes, so a real backend had no image to condition on.

## Decision

- **Contract:** `Generator.generate(request, sketch_payload: bytes)` receives the encoded bytes of `request.sketch`. This is the smallest change that gives a backend the image while records keep referencing images only through `ImageRef`. The fake ignores the bytes. The diffusers backend rejects bytes whose SHA-256 differs from the request's sketch.
- **Backend:** `sketchloop.diffusers_backend.DiffusersSketchGenerator`, selected with `sketchloop --backend diffusers` (the default stays `fake`). It loads lazily from the local folders `stable-diffusion-v1-5`, `control_v11p_sd15_scribble`, and `lcm-lora-sdv1-5` under `models/`, downloaded by `scripts/download_models.py` at pinned commits with only safetensors weights and configs. The safety checker is not downloaded or used, and `safety_checker: false` is recorded.
- **Device:** CUDA, then Apple MPS, then CPU. float16 on CUDA, float32 on CPU and MPS, since SD 1.5 in float16 on MPS can produce NaN (black) images.
- **Modes:** `fast` is the default on CPU, with LCM-LoRA fused, `LCMScheduler`, 4 steps, guidance 1.5, and the negative prompt reported unsupported. `quality` is the default on CUDA and MPS, with `UniPCMultistepScheduler`, 20 steps, guidance 7.5, and the negative prompt supported. Both expose `steps`, `guidance_scale`, and `conditioning_scale` controls.
- **Conditioning:** the sketch is resized to 512 px on the long side with both sides divisible by 8, and pixels clearly darker than the paper (the preprocessing stroke threshold) become white strokes on black.
- **Reproducibility:** each candidate uses its own CPU `torch.Generator` seeded with seed + index. Without a requested seed, a base seed is drawn and recorded. Effective settings record mode, device, dtype, scheduler, resolution, seeds, steps, guidance and conditioning scales, and library versions. The model ID names each repository with the commit recorded by its download, or `unknown-revision`.
- **Failures:** missing folders, a missing extra, load errors, and generation errors raise `GenerationBackendError` with one line. There is no fallback to fake output.
- **Dependencies:** the optional `generation` extra pins `diffusers==0.40.0`, `torch==2.14.1`, `transformers==5.18.0`, `accelerate==1.15.0`, `peft==0.21.2` (needed by `load_lora_weights`), `safetensors==0.8.0`, and `huggingface_hub==1.33.0` (diffusers 0.40 requires `huggingface_hub<2`). These installed together on Windows with Python 3.12. The default PyPI torch wheel for Windows and macOS is CPU or MPS only, so no extra index is needed there. Linux users without a GPU can install torch from the PyTorch CPU index first.

## Consequences

Default installs and checks stay light, because the backend imports torch and diffusers only when used. A real-model test is marked `model` and excluded by default (`python -m pytest -m model`). Weights take about 5.9 GB. On the Windows laptop CPU, one fast-mode candidate at 512x360 took about 106 s including the model load. The conditioning image is not yet stored as a run artifact, because `GenerationOutput` payloads are limited to candidates. Determinism across devices and library versions is not guaranteed, as Diffusers documents. On Windows, installing torch into a deeply nested folder can fail without long-path support.
