# First real generation backend

## Question and decision informed

Which sketch-conditioned model should be the first real `Generator` adapter (T03, R02, R03, R06, R09), given a Windows laptop with no GPU and possible later Colab or GPU access? Informs the first-model ADR and the `generation` optional extra.

## Sources (read 2026-10-02)

- PyPI JSON metadata: `diffusers` 0.40.0 (2026-08-20), `torch` 2.14.1 (2026-09-30), `transformers` 5.18.0, `accelerate` 1.15.0. All Apache-2.0 and Python >=3.10.
- Hugging Face model API and cards (revision = commit SHA prefix): [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) `451f4fe16113`, [control_v11p_sd15_scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble) `3564ec7b87dd`, [control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) `8a158f547e03`, [t2iadapter_sketch_sd15v2](https://huggingface.co/TencentARC/t2iadapter_sketch_sd15v2) `a73a564e35b0`, [lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) `cf2fced511db`, [sd-turbo](https://huggingface.co/stabilityai/sd-turbo), [sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo), [sdxs-512-dreamshaper-sketch](https://huggingface.co/IDKiro/sdxs-512-dreamshaper-sketch) `cd3416780873`.
- [Diffusers LCM guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm) and [reproducibility guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/reusing_seeds) (main docs).
- [FastSD CPU](https://github.com/rupeshs/fastsdcpu) README (MIT), CPU benchmarks. [img2img-turbo](https://github.com/GaParmar/img2img-turbo) (MIT, last push 2025-08-01). [SDXS paper](https://arxiv.org/abs/2403.16627). [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Findings

- Verified: SD 1.5 ControlNet scribble expects user-drawn strokes, preprocessed with HED `scribble=True`, and loads through `ControlNetModel` + `StableDiffusionControlNetPipeline`. Weights are CreativeML OpenRAIL-M, with use-based restrictions that must be passed on, so they are not OSI-open. Not gated.
- Verified: T2I-Adapter sketch v2 is Apache-2.0, expects white outlines on black (PidiNet), and uses `T2IAdapter` + `StableDiffusionAdapterPipeline`. Diffusers calls it "even more lightweight than ControlNet... faster but the results may be slightly worse".
- Verified: LCM-LoRA works with ControlNet and T2I-Adapter in 4 steps. With LCM, guidance_scale is best at 1.0-2.0 and "negative prompts don't work". SD-Turbo also ignores `guidance_scale` and `negative_prompt`.
- Verified: SDXL-Turbo is licensed `sai-nc-community` (non-commercial). SD-Turbo points to the Stability community license. SDXS sketch is OpenRAIL++, one-step, Diffusers-loadable.
- Verified: FastSD CPU reports 1-step 512 px at 0.82 s (SDXS), 1.7 s (SD-Turbo), and 2.5 s (SDXL-Turbo) on a Core i7-12700 desktop with OpenVINO and a tiny decoder. It lists LCM-LoRA at about 4 GB RAM and OpenVINO at about 11 GB. These are desktop numbers, not this laptop.
- Verified: img2img-turbo (pix2pix-turbo sketch) is MIT, GPU-only per its README (0.29 s on A6000), custom scripts rather than a Diffusers pipeline.
- Verified: Diffusers recommends a CPU `torch.Generator` for reproducibility and says determinism is "not guaranteed even with an identical seed". Colab free GPUs are "not guaranteed", vary over time, and time out when idle.
- Inference: plain SD 1.5 + ControlNet at 20-30 steps on a laptop CPU is likely tens of seconds to minutes per image. LCM-LoRA at 4 steps should cut this several times. Measure in T03 before committing to batch sizes.
- Inference: ControlNet scribble is the most widely used and documented SD 1.5 sketch condition, and its input type matches camera-captured pencil strokes after inversion and thresholding.

## Options compared

| Option | Conditioning | CPU feasibility | Guidance support | License | Notes |
| --- | --- | --- | --- | --- | --- |
| SD 1.5 + ControlNet scribble + LCM-LoRA | Strong, standard | Likely usable at 4 steps, 512 px | Prompt, conditioning scale, steps, seed. No negative prompt at CFG ~1 | OpenRAIL-M | Same code runs on Colab with LCM off and 20-30 steps |
| SD 1.5 + T2I-Adapter sketch (+ LCM) | Good, slightly weaker | Faster than ControlNet | Same as above | Apache adapter, OpenRAIL-M base | Good fallback with the same base |
| SDXS-512 sketch ControlNet | One-step | Best CPU evidence | Prompt only, little control | OpenRAIL++ | Young project, quality evidence thin |
| SD/SDXL-Turbo img2img | Weak (img2img, not structural) | Fast | No CFG or negative prompt | Non-commercial or community | Poor fit for sketch fidelity |
| pix2pix-turbo sketch | Strong | GPU only | Prompt, gamma | MIT | Not packaged, custom code |
| SDXL / FLUX + ControlNet | Strong | Not on this laptop | Full | Varies | Later Colab or GPU option |

## Recommendation and confidence

First backend: SD 1.5 + ControlNet v1.1 scribble through Diffusers, with LCM-LoRA as a switchable speed mode for CPU. Fallback: swap the ControlNet for T2I-Adapter sketch v2 on the same base, and try SDXS sketch only if CPU latency stays unacceptable. Confidence: medium. The fit is well documented, but laptop latency is unmeasured.

## Implementation guidance

- Module `sketchloop.generation.diffusers_backend` (name for the architect), behind an optional extra such as `generation = ["diffusers==0.40.0", "torch==2.14.1", "transformers==5.18.0", "accelerate==1.15.0"]`. Install CPU torch from the PyTorch CPU index. Verify the set resolves together on Windows with Python 3.11 and 3.12 before writing the ADR.
- Outline: `ControlNetModel.from_pretrained(id, revision=sha)`, `StableDiffusionControlNetPipeline.from_pretrained(base, controlnet=..., revision=sha, safety_checker=None)` while recording that the checker is off, optional `LCMScheduler` + `load_lora_weights(lcm_id, revision=sha)`, then a call with `image`, `num_inference_steps`, `guidance_scale`, `controlnet_conditioning_scale`, `num_images_per_prompt`, and a fresh CPU generator per call.
- Record: model IDs and revisions, scheduler class and config, LCM on or off, steps, guidance scale, conditioning scale, seed per candidate, resolution, dtype, device, library versions, conditioning image `ImageRef`, and stage timing.
- Capabilities must report negative prompt as unsupported when LCM is on with guidance at or below 1, so `validate_request` gives explicit feedback (R03).
- Pitfalls: scribble expects white strokes on black, so invert pencil-on-paper captures. Weights total several GB on first download. Keep `from_pretrained` lazy so imports stay offline-safe.
- Tests: default tests use `FakeGenerator`. Unit-test request-to-pipeline-kwargs mapping with a stub pipeline object injected into the adapter. One marked, explicitly invoked test runs the real model.

## Open questions for the owner

- Is CreativeML OpenRAIL-M acceptable for the thesis and any later lab use?
- What per-round latency is acceptable on the laptop before Colab becomes the default?
- Run on Colab by hosting the whole app there, or only the generator behind a remote adapter?
