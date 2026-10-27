# Prompt suggestion models for U1

Small local vision-language and captioning models that can look at a processed pencil sketch and help propose Stable Diffusion style prompts, for owner requirement U1 ([ADR 0019](../decisions/0019-offline-prompt-suggestions.md)). Read 2026-10-27 against the installed `transformers==5.18.0` and `torch==2.14.1`, the Hugging Face model cards and Hub API, and the transformers documentation (v5.21.0 at the time of reading).

## Question and decision informed

Which small local model should the `PromptSuggester` use to (a) name the drawn object and (b) offer 3-5 short style prompts, fully offline at run time, on the owner's Apple M2 Pro (MPS, 16 GB shared with SD 1.5 + ControlNet in float16) and on CPU? The input is the processed `sketch.png`: grayscale, 512 px on the long side, dark strokes on light paper, usually one designed object. The answer informs the architect's `PromptSuggester` design, a new entry in `scripts/download_models.py`, and whether any dependency must be added to `pyproject.toml`.

Constraints from the task, not from the proposal: permissive license, safetensors weights at a pinned revision, loadable with the pinned libraries, and no `trust_remote_code`. Automatic prompt improvement is an optional extension in [PROJECT.md](../PROJECT.md) (p. 6, §8), activated by the owner as U1.

## Sources

| Source | Version or revision | Read |
| --- | --- | --- |
| [Florence-2 transformers docs](https://huggingface.co/docs/transformers/model_doc/florence2) | docs v5.21.0, model present in installed 5.18.0 | 2026-10-27 |
| [florence-community/Florence-2-base](https://huggingface.co/florence-community/Florence-2-base) | `00921df66db728a9ceb750f5eca43e5c203a2051`, modified 2025-09-11 | 2026-10-27 |
| [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large) | `4271c66b88cdbc05735372ec13b2360108de5317`, modified 2025-09-11 | 2026-10-27 |
| Xiao et al., *Florence-2: Advancing a unified representation for a variety of vision tasks*, [arXiv:2311.06242](https://arxiv.org/abs/2311.06242), CVPR 2024 | v1 | abstract and card benchmarks only |
| [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | `7e3e67edbbed1bf9888184d9df282b700a323964`, modified 2025-04-08 | 2026-10-27 |
| [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) | `a7da5b986cb59b408707209984f360a5f4ad7e47`, modified 2025-04-08 | 2026-10-27 |
| Marafioti et al., *SmolVLM: Redefining small and efficient multimodal models*, [arXiv:2504.05299](https://arxiv.org/abs/2504.05299) | cited by the card | not read in full |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `2fc06364715b967f1860aea9cf38778875588b17`, modified 2026-03-02 | 2026-10-27 |
| [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | `89644892e4d85e24eaac8bacfd4f463576704203`, modified 2025-10-23 | 2026-10-27 |
| [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) | `353689b859fcf0523410b1806dace5fb46ecdf41`, modified 2025-02-03 | 2026-10-27 |
| [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) | `5d6c926f44e26b07957b0dd315bbedcb4c17a5fe`, modified 2026-09-23 | 2026-10-27 |
| Installed `transformers` 5.18.0 source: `models/auto/*_auto.py`, `processing_utils.py`, `models/florence2`, `models/qwen3_5` | 5.18.0 | 2026-10-27 |

## Findings

### Library support in the pinned transformers

- Verified (installed 5.18.0): `florence2`, `smolvlm`, `idefics3`, `qwen3_vl`, `qwen3_5`, and `blip` are native model types, each mapped in the image-text-to-text auto classes, so none needs `trust_remote_code`.
- Verified (installed 5.18.0): `torchvision` is not installed in the project environment, and every candidate's image processor has a pure PIL fallback (for example `florence2` maps to `CLIPImageProcessorPil` and `qwen3_5` to `Qwen2VLImageProcessorPil`).
- Verified (installed 5.18.0): `Florence2Processor` and `BlipProcessor` have only `image_processor` and `tokenizer` parts. `SmolVLMProcessor` and `Qwen3VLProcessor` (also used by Qwen3.5) also have a `video_processor`, and `AutoVideoProcessor.from_pretrained` raises "requires `torchvision`" when torchvision is missing.
- Inference: `AutoProcessor.from_pretrained` for SmolVLM, Qwen3-VL, and Qwen3.5 will therefore fail in the current environment. Using them needs either a new `torchvision` pin matching torch 2.14.1 or a hand-built processor with `video_processor=None`, which is untested. Florence-2 and BLIP load with what is already pinned.
- Verified (installed 5.18.0): Qwen3.5's linear-attention layers use `use_kernel_func_from_hub_with_fallback`, which falls back to pure torch code. The `kernels` package is not installed, so no hub kernel is fetched. Inference: the torch fallback is slower, and run-time code should still set `HF_HUB_OFFLINE=1` so no future install of `kernels` triggers a download.
- Verified (Moondream2 Hub API): `custom_code` tag and an `auto_map` to `hf_moondream.HfMoondream`, so it needs `trust_remote_code`. It is excluded by the constraints.

### Candidates

- Verified (Hub API): Florence-2-base has 231,546,368 parameters in one F16 `model.safetensors` of 463 MB, license MIT, architecture `Florence2ForConditionalGeneration`. Florence-2-large has 776,710,144 parameters in a 1.55 GB F16 file, also MIT.
- Verified (docs): Florence-2 is task-prompted, not instruction-following. Captioning uses `<CAPTION>`, `<DETAILED_CAPTION>`, or `<MORE_DETAILED_CAPTION>`, decoded with `processor.post_process_generation`. It cannot be asked to "write five style prompts".
- Verified (card): zero-shot COCO caption CIDEr 133.0 (base) and 135.6 (large). These are photo benchmarks with no sketch evidence.
- Contradiction (card): the `florence-community/Florence-2-base` card text says the model is "a continued pretrained version of Florence-2-large model with 4k context length". This looks like text copied from another Microsoft card, while the Hub metadata clearly shows a 0.23B base model. Record the revision SHA rather than relying on the card text.
- Verified (Hub API and card): SmolVLM-256M-Instruct has 0.26B parameters in a 513 MB BF16 file and SmolVLM-500M-Instruct has 0.5B in 1.02 GB, both Apache 2.0, architecture `Idefics3ForConditionalGeneration`. The cards state under 1 GB and 1.23 GB of GPU memory for one image. They are chat models that follow short instructions, and the card warns that output "may not be accurate".
- Verified (Hub API and card): Qwen3.5-0.8B has 873,438,784 parameters in one 1.75 GB BF16 safetensors shard, license Apache 2.0, architecture `Qwen3_5ForConditionalGeneration`, with a vision encoder. It is non-thinking by default. The card recommends temperature 0.7, top-p 0.8, top-k 20, and presence penalty 1.5 for vision tasks, and positions the model for "prototyping, task-specific fine-tuning, and other research".
- Verified (Hub API and card): Qwen3-VL-2B-Instruct has 2.13B parameters in a 4.26 GB BF16 file, Apache 2.0. It is the strongest instruction follower here and the heaviest.
- Verified (Hub API): BLIP-large has 0.47B parameters stored as F32 (1.88 GB file), license BSD-3-Clause. It only captions, optionally continuing a text prefix such as "a pencil sketch of".
- Verified (search, 2026-10-27): no study was found that measures any of these models on pencil sketches or line drawings. A 2024 paper fine-tuned Florence-2 on engineering drawings ([arXiv:2411.03707](https://arxiv.org/abs/2411.03707)), which is not zero-shot evidence. Inference: all of them saw some drawings and clip art in web-scale training, so naming a single clear object such as a chair or lamp is likely to work, while unusual or ambiguous designs will often be named generically ("a drawing of a object") or wrongly. Only a test on the owner's own sketches can settle quality.
- Newer small models exist (MiniCPM-V 4.6 and 4.7, about 1.3-1.6B, and the model types are in 5.18.0), but they are larger than needed and were not evaluated here.

## Options compared

Sizes are verified from the Hub. Memory and latency are inferences for one 512 px sketch with short outputs, not measurements.

| Model | Params | fp16 weights | Expected memory in use | License | Loads with current pins | Can write prompts itself | MPS notes | Latency class, MPS / CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Florence-2-base | 0.23B | 0.46 GB | about 0.7-1 GB | MIT | Yes, native, no torchvision | No, captions only | Native torch ops, float16 should work, untested here | under 1 s / about 1-3 s |
| Florence-2-large | 0.78B | 1.55 GB | about 2 GB | MIT | Yes, native, no torchvision | No, captions only | Same as base | about 1-2 s / about 3-8 s |
| SmolVLM-256M-Instruct | 0.26B | 0.51 GB | under 1 GB (card) | Apache 2.0 | Model yes, `AutoProcessor` needs torchvision | Yes, weakly | Card shows CUDA or CPU with eager attention, MPS not mentioned | about 1-3 s / about 3-8 s |
| SmolVLM-500M-Instruct | 0.5B | 1.02 GB | about 1.2 GB (card) | Apache 2.0 | Model yes, `AutoProcessor` needs torchvision | Yes, modestly | Same as 256M | about 2-4 s / about 5-15 s |
| Qwen3.5-0.8B | 0.87B | 1.75 GB | about 2.5-3 GB | Apache 2.0 | Model yes, `AutoProcessor` needs torchvision | Yes, best of the small ones | New hybrid linear attention uses the slower torch fallback | about 2-5 s / about 10-30 s |
| Qwen3-VL-2B-Instruct | 2.1B | 4.26 GB | about 5-6 GB | Apache 2.0 | Model yes, `AutoProcessor` needs torchvision | Yes, well | Heavy next to SD 1.5 on 16 GB | about 4-10 s / about 30-90 s |
| BLIP-large | 0.47B | 0.94 GB (stored F32 as 1.88 GB) | about 1-2 GB | BSD-3-Clause | Yes, native | No, captions only | Plain ViT and BERT ops | under 1 s / about 1-3 s |
| Moondream2 | 1.9B | 3.85 GB | about 4-5 GB | Apache 2.0 | No, needs `trust_remote_code` | Yes | Excluded | Excluded |

## Recommendation and confidence

- **Recommended:** Florence-2-base from `florence-community/Florence-2-base` at revision `00921df66db728a9ceb750f5eca43e5c203a2051`. It is MIT, under 0.5 GB in float16, native in the pinned transformers, needs no new dependency, and is fast enough on CPU. Use it only to name the object with `<CAPTION>`, and build the 3-5 prompts by combining that object name with a fixed list of style templates. The templates are the same list the always-available fixed suggester uses, so both suggesters share one code path.
- **Fallback:** Florence-2-large at revision `4271c66b88cdbc05735372ec13b2360108de5317`, with identical code and only the repository and revision changed, if base names sketches poorly. It costs about 1.5 GB of memory and a second or two more.
- **Not chosen now:** Qwen3.5-0.8B is the best small option if the owner later wants the model to write whole prompts. It needs a `torchvision` pin, about 2.5-3 GB next to SD 1.5, and output parsing because a 0.8B chat model does not reliably follow a list format. SmolVLM has the same torchvision cost with weaker instruction following.
- **Confidence:** high for license, size, and library fit, all verified. Medium for memory and latency, which are inferred. Low for naming quality on pencil sketches, since no source tests it. Measure on 10-20 of the owner's sketches before declaring U1 done.

## Implementation guidance

### Target module and boundary

A `PromptSuggester` abstract base class next to `Evaluator`, with a fixed-list implementation and a Florence-2 implementation in a module such as `src/sketchloop/suggestions.py` (the architect decides the layout). It takes the processed sketch image (the same `sketch.png` given to the generator) and returns suggestions. It never calls the generator. The UI only puts the chosen suggestion into the prompt box (ADR 0019).

### Download

Add one entry to `scripts/download_models.py` with the repository ID, the pinned revision, and allow patterns `*.json`, `*.txt`, and `model.safetensors`. The repository has no pickle weights, so the patterns stay small. Total size is about 470 MB.

### Loading and captioning (Florence-2)

```python
import os

import torch

from transformers import AutoProcessor, Florence2ForConditionalGeneration

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # No network at run time.
model_folder = "models/florence-2-base"  # Local snapshot from download_models.py.
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
processor = AutoProcessor.from_pretrained(model_folder)
model = Florence2ForConditionalGeneration.from_pretrained(model_folder, dtype=dtype).to(device).eval()


def caption_sketch(image, task: str = "<CAPTION>") -> str:
    # The sketch is grayscale, and the processor expects three channels.
    rgb_image = image.convert("RGB")
    inputs = processor(text=task, images=rgb_image, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=40, num_beams=3, do_sample=False)
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(text, task=task, image_size=rgb_image.size)
    return parsed[task].strip()
```

Verified in the installed source: `.to(device, dtype)` on a `BatchFeature` casts only floating tensors and moves the rest, so token IDs stay integers. Use float32 on CPU, because float16 matmuls on CPU are slow or unsupported for some ops.

### (a) Object name from the caption

Typical captions look like "a drawing of a chair with a curved back". Strip leading framing words with a small rule list, keep the rest as the subject, and cap it at about six words:

```python
FRAMING_PREFIXES = (
    "a black and white drawing of",
    "a black and white sketch of",
    "a pencil drawing of",
    "a pencil sketch of",
    "a line drawing of",
    "a drawing of",
    "a sketch of",
    "an image of",
    "the image shows",
)


def extract_subject(caption: str) -> str:
    # Drop the medium words, since the generator adds its own style.
    subject = caption.lower().rstrip(".")
    for prefix in FRAMING_PREFIXES:
        if subject.startswith(prefix):
            subject = subject[len(prefix):].strip()
            break

    return " ".join(subject.split()[:6]) or "object"
```

Keep the raw caption too, for the record and for display ("The model sees: a chair with a curved back").

### (b) Style prompts from templates

```python
STYLE_TEMPLATES = (
    "{subject}, product design render, studio lighting, white background",
    "{subject}, Scandinavian minimalist design, light oak wood, soft daylight",
    "{subject}, industrial design, brushed steel and black leather, dramatic lighting",
    "{subject}, mid-century modern, walnut and brass, warm interior photo",
    "{subject}, concept art, watercolor and ink, muted colors",
)


def suggest_prompts(subject: str, count: int = 5) -> list[str]:
    return [template.format(subject=subject) for template in STYLE_TEMPLATES[:count]]
```

The templates belong in settings or a data file so the owner can edit them, and the fixed suggester uses the same list with a neutral subject such as "the sketched object". Keep each prompt well under CLIP's 77-token limit for SD 1.5.

### If a chat model is adopted later (Qwen3.5-0.8B)

This needs a `torchvision` version matching torch 2.14.1, checked against PyTorch's compatibility table before pinning. Ask for a fixed format and parse defensively:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": rgb_image},
        {"type": "text", "text": "This is a pencil sketch of one designed object. Line 1: name the object in at most four words. Lines 2-6: five short Stable Diffusion prompts for that object, each in a different visual style, one per line, no numbering."},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
```

Use greedy decoding for repeatable suggestions, strip numbering and empty lines, and fall back to templates when fewer than three usable lines come back.

### Parameters to record (ADR 0019 records)

Suggester kind (fixed or model), model repository and revision SHA, dtype and device, task prompt, generation settings (`max_new_tokens`, `num_beams`, `do_sample`), the raw caption, the extracted subject, the template list version or hash, the offered suggestions, which one the person picked (if any), and suggestion latency and model load time as separate stages like T08.

### Reproducibility

Greedy or beam decoding with fixed inputs is deterministic on one device in principle, but MPS and CPU float16 versus float32 results can differ in the last token choices. Record the device and dtype, and do not claim exact replay across machines.

### Pitfalls

- Pass the processed `sketch.png`, not the raw camera frame, and convert it to RGB.
- Florence-2 decode must use `skip_special_tokens=False` before `post_process_generation`, as in the docs.
- Load the suggester lazily on first use, or at session start, and keep one instance. Loading next to SD 1.5 adds about 0.5 GB on MPS, so check memory pressure, since T08 saw swapping with float32 SD.
- Loading the original `microsoft/Florence-2-base` repository uses remote code. Use only the `florence-community` conversion.
- Similarity of a suggestion to the sketch is a research signal at most. A plausible caption does not mean the design intent was understood, and the person stays in control of the prompt.

### Test ideas

- A fake captioner returning fixed captions, injected into the Florence-2 suggester, tests `extract_subject` and template filling with parametrized captions, including an empty caption, "a drawing of", and a long caption.
- A test that the fixed suggester needs no model and returns the configured count.
- A test that a suggester failure surfaces as a clear error and the page still offers fixed suggestions, without silently substituting output and calling it model output.
- A test that the round record keeps the suggester identity, revision, and picked suggestion, and that older rounds load these as unavailable.
- A `model`-marked test that loads the real Florence-2-base from the models folder, captions a synthetic line drawing, and checks a non-empty string within a time bound. It must be skipped when the weights are absent.

## Open questions for the owner

- Is template-based prompting from a Florence-2 object name enough, or do you want the model itself to write the prompts? The second needs `torchvision` and about 2.5-3 GB more memory with Qwen3.5-0.8B.
- Which 5-8 styles should the default template list contain for your design domain?
- Should suggestions run automatically after each capture, or only when you click a "Suggest" button? A button keeps the stage visible in timings and avoids work you do not use.
