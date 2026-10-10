# Model: ControlNet v1.1 scribble for SD 1.5

Zhang, Rao, and Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models*, ICCV 2023, [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) (v3). arXiv non-exclusive license, so the PDF is not redistributable. Local, git-ignored copy: `docs/articles/local/ControlNet.pdf` (12 pages). Weights: [lllyasviel/control_v11p_sd15_scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble), revision `3564ec7b87dd706d5fc760d21c9e90bb160d1e9f` (last modified 2023-05-04). Also read: the [ControlNet v1.1 repository](https://github.com/lllyasviel/ControlNet-v1-1-nightly) and the v1.0 [scribble demo script](https://github.com/lllyasviel/ControlNet/blob/main/gradio_scribble2image.py). Read 2026-10-09.

## Question and decision informed

How the sketch conditions [SD 1.5](model-sd15.md) in T03, what input preprocessing must deliver, and what to record (R01, R02, R03, R06).

## How it works (plain words)

ControlNet makes a trainable copy of the SD UNet's encoder and feeds a control image (here, strokes) into it. The copy's outputs are added back into the frozen SD UNet through "zero convolutions", which start at zero so training cannot damage the base model. At inference, the control image steers layout and shapes while the prompt steers content and style. A conditioning scale sets how strongly the strokes are followed.

## Findings

- Verified (paper pp. 3-4, §3.1-3.2, Fig. 3): the locked SD encoder plus a trainable copy joined by zero convolutions. A small 4-layer encoder maps the 512×512 condition to the 64×64 latent grid (p. 5).
- Verified (paper p. 5, §3.3): 50% of training prompts were replaced with empty strings, so the model can read semantics from the strokes alone. With ambiguous strokes and a vague prompt, "the model tries to interpret input shapes" (p. 8, Fig. 11).
- Verified (paper pp. 5-6, §3.4): CFG interacts with the control signal. The authors weight the control by resolution (CFG-RW) to avoid guidance that is too weak or too strong.
- Verified (paper p. 6, Table 1): in a 12-person ranking against sketch baselines, ControlNet scored highest on quality (4.22) and sketch fidelity (4.28) out of 5. This is a small user study, not proof of design quality.
- Verified (paper p. 4): training with ControlNet costs about 23% more GPU memory and 34% more time per iteration. Inference cost is similar in kind, because the copy runs on every step.
- Verified (paper p. 8, Fig. 12): ControlNets transfer to fine-tuned SD 1.5 community models without retraining.
- Verified (card and v1.1 README): input is "random or user-drawn strokes". Training used synthesized scribbles with aggressive morphological transforms, widths from 1 to 24 px on a 512 canvas. The card example uses `HEDdetector(..., scribble=True)`, `UniPCMultistepScheduler`, and 30 steps. License CreativeML OpenRAIL-M per the card, although the Hub metadata tag says `openrail`.
- Verified (v1.0 scribble demo): user drawings are converted with `detected_map[np.min(img, axis=2) < 127] = 255`, so dark pencil on white paper becomes white strokes on black. Defaults were 512 px, 20 steps, and guidance 9.0. The v1.1 demo also thins, blurs, and rebinarizes the map.
- Verified (Hub tree): fp32 safetensors 1.45 GB, fp16 0.72 GB.
- Inference: camera captures need the T02 preprocessing (crop, deskew, contrast) plus inversion and thresholding, not HED. HED would add a heavy dependency and redraw the person's own lines.
- Inference: typical failures are faint or broken strokes being ignored, dense hatching read as texture, shadows or paper edges read as shapes, and an object that is not the intended one when the prompt does not name it. Conditioning scale 1.0 can feel rigid, and 0.5-0.8 gives looser variants.

## Speed and memory (inference)

ControlNet adds roughly 40% to each UNet step. At 512 px and 25 steps with CFG, expect about 1.5-4 min per image on the laptop CPU and about 15-35 s on M2 Pro MPS. With [LCM-LoRA](model-lcm-lora.md) at 4 steps, see that note. Memory adds about 1.45 GB (fp32) or 0.72 GB (fp16) to the base.

## Implementation guidance

- Preprocessing stage `scribble`: grayscale, threshold at 127 (or Otsu), invert to white on black, optionally dilate thin lines to 2-4 px at 512, and output 3-channel 512×512. Save it as the conditioning `ImageRef` with its step records.
- Load with `ControlNetModel.from_pretrained(id, revision=sha, use_safetensors=True)` and pass it to `StableDiffusionControlNetPipeline`.
- Record: ControlNet ID and revision, conditioning image reference and its preprocessing steps, `controlnet_conditioning_scale`, `guess_mode`, and any start or end control fractions, in addition to the base parameters.
- Tests: unit-test the scribble transform on a synthetic black-on-white drawing (assert polarity and size). A marked real test checks that the output follows a simple shape.

## Open questions for the owner

- Should the person see the conditioning image before generation, to catch inverted or noisy captures?
- Default conditioning scale: strict (1.0) or exploratory (about 0.7)?
