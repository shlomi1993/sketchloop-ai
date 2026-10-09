# Model: Stable Diffusion v1.5 (base)

Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) (v2). arXiv non-exclusive license, so the PDF is not redistributable. Local, git-ignored copy: `docs/articles/local/Latent Diffusion Models.pdf` (45 pages). Weights: [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` (last modified 2024-09-07). Read 2026-10-09.

## Question and decision informed

What the T03 base model is, what to expect from it, and what the Diffusers adapter must record (R02, R03, R06, R09). Companion notes: [ControlNet scribble](model-controlnet-scribble.md) and [LCM-LoRA](model-lcm-lora.md). Choice of this stack: [first-model options](first-model-options.md).

## How it works (plain words)

An autoencoder (VAE) squeezes a 512×512 image into a 64×64×4 "latent". A UNet learns to remove noise step by step in that small space, steered by the prompt through cross-attention to a frozen CLIP ViT-L/14 text encoder. Generation starts from random latent noise (fixed by the seed), runs the scheduler for N steps, and decodes the result with the VAE. Classifier-free guidance (CFG) runs the UNet twice per step, with and without the prompt (or with the negative prompt), and pushes away from the second.

## Findings

- Verified (paper pp. 3-4, §3; p. 1 Fig. 1): diffusion runs in a perceptually compressed latent space to cut compute. Conditioning enters through cross-attention (p. 4, §3.3).
- Verified (paper p. 9, §5): "sequential sampling process is still slower than that of GANs", and autoencoder reconstruction "can become a bottleneck" for fine pixel accuracy. Thin pencil detail may soften.
- Verified (card): fine-tuned 595k steps at 512×512 from v1-2, latent downsampling factor 8, CLIP ViT-L/14 text encoder. Limitations: no "perfect photorealism", "cannot render legible text", weak compositionality, faces and people often wrong, English captions mainly, LAION-5B bias with Western defaults, and some memorization.
- Verified (card): the repository is a mirror of the deprecated `runwayml` repository, not gated. License CreativeML OpenRAIL-M.
- Verified (license, §§5-6, Attachment A): use-based restrictions (unlawful use, harming minors, harmful misinformation, harassment, discriminatory or automated legal decisions, medical advice, and others) must be passed on to users of the model or derivatives. The licensor "claims no rights in the Output", and the user is accountable for it.
- Verified (Hugging Face file tree): the UNet alone is 3.44 GB in fp32 and 1.72 GB in fp16 safetensors. The repo also holds `.bin` and full-checkpoint files that must not be downloaded.
- Verified ([Diffusers MPS guide](https://huggingface.co/docs/diffusers/main/en/optimization/mps)): use `pipe.to("mps")` and `enable_attention_slicing()` below 64 GB RAM. Batches "can crash or fail to work reliably", so iterate per image.
- Verified ([Diffusers reproducibility guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/reusing_seeds)): use a fresh CPU `torch.Generator` per call, and determinism is "not guaranteed even with an identical seed". CPU and MPS outputs for the same seed will differ.
- Inference: native size is 512 px (multiples of 8). Much larger sizes tend to duplicate objects. The 77-token CLIP limit truncates long prompts silently.
- Inference: the safety checker blacks out flagged images. With design sketches, false positives are rare but possible. Disabling it must be recorded.

## Speed and memory (inference unless marked)

| Setting, 512 px, base only | Laptop CPU, fp32 | M2 Pro, MPS, fp16 |
| --- | --- | --- |
| 25 steps with CFG (2 UNet passes per step) | about 1-3 min per image | about 10-25 s per image |
| Weights in memory | about 4.5 GB, so 16 GB RAM advised | about 2.3 GB of unified memory |

Evidence: Core ML SD 1.5 takes 21.9 s per image on the M2 GPU (10 cores) in a [Hugging Face benchmark](https://huggingface.co/blog/fast-mac-diffusers) (verified). The M2 Pro GPU has more cores, but PyTorch MPS cannot use the Neural Engine. CPU fp16 is not practical, so use fp32 on CPU. Measure in T03.

## Implementation guidance

- Load with `use_safetensors=True`, `revision=<sha>`, and `variant="fp16"` only on MPS. Pass `safety_checker=None` only as a recorded choice. Newer Diffusers docs use `dtype=` where older ones use `torch_dtype=`, so check the signature in the pinned version (0.41.0 on PyPI when read).
- Record: model ID and revision, weight variant and dtype, device, scheduler class and config, steps, guidance scale, prompt, negative prompt, seed per candidate, width and height, safety checker on or off, and the versions of Diffusers, torch, and transformers.
- Replay limit: same seed, revision, versions, and device give close but not guaranteed identical output. Record the device and never claim exact replay across devices.
- Tests: default tests inject a stub pipeline and assert the kwargs mapping. One marked test loads the real weights.

## Open questions for the owner

- Is CreativeML OpenRAIL-M acceptable for the thesis and lab use (already open in [first-model options](first-model-options.md))?
- Keep the safety checker on by default, or off and recorded?
