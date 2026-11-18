# Model: CLIP ViT-B/32 and ViT-L/14 as candidate evaluators

Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, PMLR 139, 8748-8763 (2021), [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). Weights: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32), revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (last modified 2024-02-29), and [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14), revision `32bd64288804d66eefd0ccbe215aa642df71cc41` (last modified 2023-09-15). Also read: the two model cards at those revisions, the [OpenAI CLIP repository license](https://github.com/openai/CLIP/blob/main/LICENSE), and the installed `transformers` 5.18.0 source with `torch` 2.14.1 and `huggingface_hub` 1.33.0. Read 2026-11-18.

## Question and decision informed

Can CLIP become the first real `Evaluator`, selectable by name beside `none`, so each round records how close each candidate is to the prompt (R04, R08)? Which repositories, revisions, files, and transformers calls does it need, and what does the score mean?

## How it works (plain words)

CLIP has an image encoder and a text encoder trained together so that an image and a caption that describes it land close together in one shared vector space. The cosine of the angle between an image vector and a text vector is therefore a measure of how well the image matches the words. It says nothing about whether the design is good, original, or buildable.

## Findings

Labels: **Verified** means read in the named source or confirmed by running code. **Inference** is my reasoning. The proposal names CLIP-based similarity filtering only as background from the referenced paper, as summarized in [PROJECT.md](../PROJECT.md).

### Repositories, files, and license

- Verified (Hub API): `openai/clip-vit-base-patch32` holds `config.json`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`, and weights only as `pytorch_model.bin` (605,247,071 bytes), `tf_model.h5`, and `flax_model.msgpack`. **There is no safetensors file on `main`.** About 40 open, unmerged `SFconvertbot` pull requests offer conversions, and none is reviewed by the repository owner.
- Verified (Hub API): `openai/clip-vit-large-patch14` holds the same configs plus `model.safetensors` (1,710,540,580 bytes) next to `pytorch_model.bin`, `tf_model.h5`, and `flax_model.msgpack`.
- Verified (Hub API): neither card has a `license` metadata field. The cards say they are taken from the OpenAI CLIP repository, whose `LICENSE` is MIT (Copyright 2021 OpenAI).
- Verified (cards): "Any deployed use case of the model - whether commercial or not - is currently out of scope." Research use to understand capabilities and limits is the intended use, and the cards warn of bias that depends on how classes or texts are phrased.
- Inference: scoring candidates in a local research tool is a non-deployed research use. Scores must stay a recorded research signal shown beside explicit human selection, never a filter that hides candidates.

### Configs (verified, at the pinned revisions)

| | ViT-B/32 | ViT-L/14 |
| --- | --- | --- |
| Shared embedding size (`projection_dim`) | 512 | 768 |
| Image input | 224 px, patch 32 | 224 px, patch 14 |
| Text limit (`max_position_embeddings`) | 77 tokens | 77 tokens |
| Image preprocessing | resize 224, center crop 224, bicubic, CLIP mean and std | same |
| Weights on disk | 0.61 GB `.bin`, converted to 0.61 GB safetensors | 1.71 GB safetensors |

The center crop means a wide candidate loses its left and right edges before scoring. SD 1.5 candidates at 512x360 lose about 30% of their width.

### transformers 5.18.0 API (verified in the installed source)

- `CLIPModel.from_pretrained(folder, dtype=torch.float32, use_safetensors=True)` and `CLIPProcessor.from_pretrained(folder)` load from a local folder with no network. `CLIPProcessor` is a thin `ProcessorMixin` around the image processor and the tokenizer.
- `CLIPModel.forward(input_ids, pixel_values, attention_mask, ...)` returns `CLIPOutput` with `image_embeds` and `text_embeds` already divided by their norms, and `logits_per_image`, which is the cosine times `logit_scale.exp()` (100 for the released models).
- Changed from transformers 4: `get_image_features` and `get_text_features` now return a `BaseModelOutputWithPooling` whose `pooler_output` holds the projected embedding, not a tensor. Calling the model's forward and reading `image_embeds` and `text_embeds` avoids the difference.
- `processor(text=[prompt], images=images, return_tensors="pt", padding=True, truncation=True)` cuts a prompt longer than 77 tokens, which would otherwise fail on the position embeddings.

### Conversion of ViT-B/32 (verified by running the download script)

`scripts/download_models.py` downloads the pinned `pytorch_model.bin` and converts it once with `torch.load(..., weights_only=True)` and `safetensors.torch.save_file`, as ADR 0022 does for the T2I-Adapter. transformers looks for `model.safetensors`, so the converted file takes that name. The converted file loaded 398 weights with no missing-key warning, and the example sketch scored 0.325 against "a pencil sketch of a chair" and 0.245 against "a photo of a dog", so the weights are the trained ones.

### Measured on the Apple M2 Pro (verified, MPS, float32)

| | ViT-B/32 | ViT-L/14 |
| --- | --- | --- |
| First evaluation, including the model load | 5.5 s (fake backend, 2 candidates) | 3.3-4.9 s (t2i-adapter backend, 2 candidates) |
| Later evaluations | 0.03-0.3 s for 1 candidate | 0.13 s for 2 candidates |
| Process peak resident memory | 0.54 GB with the fake backend | 3.3 GB with the t2i-adapter backend in fast float16 |

The system stayed at 49-74% free memory during the ViT-L/14 runs with no new swapping. ViT-B/32 scores of fake noise candidates were about 0.23, and ViT-L/14 scores of real t2i-adapter chairs were 0.25-0.26 for "modern chair".

## Options compared

| Option | Notes |
| --- | --- |
| A. One `ClipEvaluator` class taking a `ModelSource`, two registry names | Smallest. Both models share the code and differ only in folder, repo, and revision. Recommended. |
| B. `open_clip` or the original `clip` package | Adds a dependency for the same weights. transformers is already installed for Florence-2. |
| C. Pin an unmerged `SFconvertbot` pull request for ViT-B/32 safetensors | Avoids conversion but trusts an unreviewed ref. Rejected. |

## Recommendation and confidence

Use option A with float32 on the generator's device, a lazy load on the first evaluation, and scores as the raw cosine from -1 to 1 rather than CLIP's scaled logits, so values compare across rounds and models are not softmaxed against each other. Record the model ID, revision, and meaning with each round. Confidence is **high** for repositories, revisions, files, and the API, which were checked against the Hub and the installed source and by running both models, and **medium** for the meaning of the numbers on sketch-conditioned product images, which CLIP was not trained to judge.

## Implementation guidance

Implemented in `src/sketchloop/evaluators/clip.py` and registered as `clip-b32` and `clip-l14` in `src/sketchloop/components.py`.

```python
CLIP_B32_MODEL = ModelSource(folder="clip-vit-base-patch32", repo_id="openai/clip-vit-base-patch32",
                             revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                             allow_patterns=("*.json", "*.txt", "pytorch_model.bin"))
CLIP_L14_MODEL = ModelSource(folder="clip-vit-large-patch14", repo_id="openai/clip-vit-large-patch14",
                             revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
                             allow_patterns=("*.json", "*.txt", "model.safetensors"))

outputs = model(**processor(text=[prompt], images=images, return_tensors="pt", padding=True, truncation=True).to(device))
scores = outputs.image_embeds @ outputs.text_embeds[0]  # cosine, both already normalized
```

**Pitfalls:** the first round's `evaluation` time includes the model load, since evaluators report no separate load time. A zero embedding gives an undefined score, which the evaluator refuses with `EvaluationError` because round records accept only finite scores. The center crop ignores the edges of wide candidates.

## Open questions

- Whether the score should compare against the sketch as well as the prompt, as the referenced paper's filtering might, is a research choice for later.
- Whether evaluators should report their load time separately, like generators, so the first round's evaluation time is comparable.
- CPU timing on the Windows laptop is unmeasured.
