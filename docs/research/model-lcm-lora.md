# Model: LCM-LoRA for SD 1.5

Luo et al., *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference*, [arXiv:2310.04378](https://arxiv.org/abs/2310.04378) (v1), and Luo et al., *LCM-LoRA: A Universal Stable-Diffusion Acceleration Module*, [arXiv:2311.05556](https://arxiv.org/abs/2311.05556) (v1). Both use the arXiv non-exclusive license, so the PDFs are not redistributable. Local, git-ignored copies: `docs/articles/local/Latent Consistency Models.pdf` (18 pages) and `docs/articles/local/LCM-LoRA.pdf` (7 pages). Weights: [latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5), revision `cf2fced511dbe7e26c8d1d397e728fbab875db4b` (last modified 2023-11-16). Also read: the [Diffusers LCM guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm) and the [LCM-LoRA blog post](https://huggingface.co/blog/lcm_lora) (2023-11-09). Read 2026-10-09.

## Question and decision informed

Whether a switchable 4-step speed mode makes [SD 1.5](model-sd15.md) + [ControlNet scribble](model-controlnet-scribble.md) usable on the owner's CPU laptop and M2 Pro, and what it costs in controls and quality (R02, R03, R09).

## How it works (plain words)

A normal diffusion model removes noise a little at a time, so it needs 20-50 steps. A consistency model is trained (distilled from a teacher) to jump from any noisy point straight to the clean image, so 2-8 steps suffice. LCM-LoRA stores that distilled skill as a small low-rank add-on (LoRA) for the SD 1.5 UNet. It is loaded on top of the base, swapped in with the `LCMScheduler`, and can be fused or unloaded at runtime.

## Findings

- Verified (LCM pp. 4-6, §4): one-stage guided distillation on an augmented ODE that includes CFG scale ω, with a skipping-step schedule. Training ω range was [2, 14] (p. 8).
- Verified (LCM p. 7, Table 1): at 512 px with ω = 8, FID 35.4 at 1 step, 13.3 at 2, 11.1 at 4, and 11.8 at 8, far ahead of DDIM and DPM-Solver at the same steps. The teacher there was SD 2.1-base, not SD 1.5, and this is full LCM, not LCM-LoRA. The authors note "a noticeable gap" at one step (p. 9).
- Verified (LCM-LoRA p. 4, Table 1, Fig. 2): the SD 1.5 LoRA has 67.5M trainable parameters (versus 0.98B). Distillation used fixed ω = 7.5, and figures use 4-step sampling. The paper shows qualitative results only, with no quantitative benchmark and no ControlNet experiment (pp. 4-5).
- Verified (LCM-LoRA pp. 4-5, Eq. 3, Fig. 3): it can be linearly combined with style LoRAs (λ1 = 0.8, λ2 = 1.0 in the example).
- Verified (card): 2-8 steps, and "either disable guidance_scale or use values between 1.0 and 2.0". License `openrail++`. Weights file 135 MB.
- Verified (Diffusers guide): works with `StableDiffusionControlNetPipeline`. The SD 1.5 example uses `LCMScheduler`, `load_lora_weights`, 4 steps, `guidance_scale=1.5`, `controlnet_conditioning_scale=0.8`, and `cross_attention_kwargs={"scale": 1}`.
- Contradiction: the Diffusers guide says negative prompts "don't work" with LCM-LoRA, while the blog says guidance 1 "ignores negative prompts" and suggests 1-2 to explore them. Inference: in Diffusers, CFG (and so the negative prompt) only applies when `guidance_scale > 1`, which also doubles the cost per step.
- Verified (blog, SDXL 1024 px, 4 versus 25 steps): M1 Mac 6.5 s versus 64 s, Intel i9-10980XE CPU 29 s versus 219 s. These are SDXL numbers, larger than SD 1.5 at 512.
- Inference: expect softer detail, less texture variety, and a narrower style range than 25-step CFG. Seeds still give different candidates.
- Inference: the license of the combined system remains bound by SD 1.5's OpenRAIL-M restrictions.

## Speed and memory (inference)

| SD 1.5 + scribble ControlNet, 512 px, one image | Laptop CPU, fp32 | M2 Pro, MPS, fp16 |
| --- | --- | --- |
| LCM-LoRA, 4 steps, guidance 1.0 | about 10-30 s | about 2-5 s |
| LCM-LoRA, 4 steps, guidance 1.5 (negative prompt active) | about 20-50 s | about 3-8 s |
| No LCM, 25 steps, guidance 7.5 | about 1.5-4 min | about 15-35 s |

Fixed costs (text encoder, VAE decode, first-call warm-up) weigh more at 4 steps. Model loading takes tens of seconds and should happen once per session. RAM grows only by the 135 MB LoRA.

## Implementation guidance

- Speed mode on: `pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)`, then `pipe.load_lora_weights(id, revision=sha)`. Speed mode off: `pipe.unload_lora_weights()` and restore the saved original scheduler config.
- Capabilities: in speed mode, report steps 2-8, guidance 1.0-2.0, and negative prompt supported only when guidance is above 1, so `validate_request` gives explicit feedback (R03).
- Record: LCM-LoRA ID and revision, speed mode on or off, LoRA scale, fused or not, and the scheduler class and config actually used, in addition to the base and ControlNet parameters.
- Tests: a stub pipeline asserts that toggling speed mode swaps scheduler and LoRA and that out-of-range guidance is rejected. A marked real test times one 4-step image.

## Open questions for the owner

- Make speed mode the default on the CPU laptop and off on the M2 Pro?
