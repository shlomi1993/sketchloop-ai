# Next-stroke suggestions and follow detection for T09

How the system can suggest how to continue a partial pencil sketch toward a stated goal, and how it can tell whether the person then drew what was suggested. This note covers requirements S01 and S02 in [PROJECT.md](../PROJECT.md), the idea from the [first supervisor meeting](../meetings/2026-11-supervisor-meeting-1.md), and [ADR 0021](../decisions/0021-guided-sketching-and-user-study.md). It was read and prototyped on 2026-11-18 with the installed `torch==2.14.1`, `diffusers==0.40.0`, `transformers==5.18.0`, `opencv-python==5.0.0.93`, and `numpy==2.5.5`, on the owner's Apple M2 Pro (16 GB, MPS) with the weights already in `models/`.

## Question and decision informed

1. **Suggestions (S01):** given a partial sketch from the webcam and a goal such as "a car", which local method can propose a few new strokes, in a few seconds on the M2 Pro, as an overlay that never replaces the sketch?
2. **Follow detection (S02):** given the suggested strokes and later captures of the same paper, how should the system decide followed, not followed, or undetermined, and which raw values should it record so a threshold can be tuned later against hand labels?

The answers inform the T09 ADR, the `architect`'s design of a suggestion component and its records, and the `backend-developer`'s implementation. Constraints from the task, not from the proposal: fully local and offline at run time, no paid service, suggestions on a button press only, and ideally feasible on the Windows laptop CPU or else served from the M2 Pro.

## Sources

| Source | Version or date | Read |
| --- | --- | --- |
| Lin et al., *Inkspire: Supporting design exploration with generative AI through analogical sketching*, CHI 2025, [doi:10.1145/3706598.3713397](https://doi.org/10.1145/3706598.3713397), [arXiv:2501.18588](https://arxiv.org/abs/2501.18588) | arXiv v1 HTML, sections 4.3, 5, 6 | 2026-11-18 |
| Lee, Zitnick, and Cohen, *ShadowDraw: Real-time user guidance for freehand drawing*, ACM TOG 30(4), 2011, [doi:10.1145/2010324.1964922](https://doi.org/10.1145/2010324.1964922) | Crossref record and project page, abstract only | 2026-11-18 |
| Vinker et al., *SketchAgent: Language-driven sequential sketch generation*, CVPR 2025, [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Vinker_SketchAgent_Language-Driven_Sequential_Sketch_Generation_CVPR_2025_paper.html), [code](https://github.com/yael-vinker/SketchAgent) (MIT) | repository README and LICENSE | 2026-11-18 |
| Ha and Eck, *A neural representation of sketch drawings* (SketchRNN), ICLR 2018, [arXiv:1704.03477](https://arxiv.org/abs/1704.03477), [Magenta.js sketch models](https://github.com/magenta/magenta-js/tree/master/sketch), [Magenta repository](https://github.com/magenta/magenta) (Apache 2.0, archived 2026-01-06) | arXiv v4, README model list | 2026-11-18 |
| [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) (CC BY 4.0) | README and `categories.txt` | 2026-11-18 |
| Chin et al., *AutoSketch: VLM-assisted style-aware vector sketch completion*, [arXiv:2502.06860](https://arxiv.org/abs/2502.06860) | v3, abstract | 2026-11-18 |
| Tiwari, Biswas, and Lladós, *SketchGPT*, ICDAR 2024, [arXiv:2405.03099](https://arxiv.org/abs/2405.03099) | v1, abstract | 2026-11-18 |
| *StrokeFusion*, [arXiv:2503.23752](https://arxiv.org/abs/2503.23752), and Jhaveri and Shah, *SketchMamba*, [arXiv:2607.23580](https://arxiv.org/abs/2607.23580) | abstracts, via search summaries | 2026-11-18 |
| Chan, Durand, and Isola, *Learning to generate line drawings that convey geometry and semantics*, CVPR 2022, [arXiv:2203.12691](https://arxiv.org/abs/2203.12691), [code](https://github.com/carolineec/informative-drawings) (MIT) | LICENSE file | 2026-11-18 |
| [controlnet_aux](https://github.com/huggingface/controlnet_aux) (Apache 2.0) and the [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) weights | PyPI 0.0.10 (2025-05-08), Annotators revision `982e7edaec38759d914a963c48c4726685de7d96`, license tag "other" | 2026-11-18 |
| [lllyasviel/control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) | revision `8a158f547e031c5b8fbca19ead09a74767ff4db0` (2023-05-04), tag `openrail` | 2026-11-18 |
| Evangelidis and Psarakis, *Parametric image alignment using enhanced correlation coefficient maximization*, IEEE TPAMI 30(10), 1858-1865, 2008, [doi:10.1109/TPAMI.2008.113](https://doi.org/10.1109/TPAMI.2008.113), and the `cv2.findTransformECC` docstring in the installed OpenCV 5.0.0 | installed 5.0.0 | 2026-11-18 |
| Borgefors, *Hierarchical chamfer matching: A parametric edge matching algorithm*, IEEE TPAMI 10(6), 849-865, 1988, [doi:10.1109/34.9107](https://doi.org/10.1109/34.9107) | citation records | 2026-11-18 |
| Installed sources: `diffusers` `StableDiffusionControlNetImg2ImgPipeline`, `transformers` `models/florence2/processing_florence2.py` | 0.40.0, 5.18.0 | 2026-11-18 |
| Existing notes: [ControlNet scribble](model-controlnet-scribble.md), [LCM-LoRA](model-lcm-lora.md), [T2I-Adapter sketch](model-t2i-adapter-sketch.md), [prompt suggestion models](prompt-suggestion-models.md), [CLIP](model-clip.md) | | 2026-11-18 |

## Findings

Labels: **Verified** means read in the named source or confirmed by running code here. **Measured** means timed in the prototype. **Inference** is my reasoning. Nothing here comes from the original proposal, which lists continuous suggestions and alternative ways of presenting suggestions only as extensions (see [PROJECT.md](../PROJECT.md)). The guided-sketching requirement comes from the supervisor meeting.

### Prior work on guiding a drawing

- Verified (Inkspire, section 4.3): Inkspire shows a "sketch scaffolding" under the designer's canvas, made by abstracting each AI-generated design into lines. Its Design2Sketch step intersects segment boundaries from Segment Anything with HED soft edges. The authors report that Canny edges, HED alone, manga line extraction, and a model trained on sketch-image pairs added texture lines, broke key lines, or added shading. It uses Stable Diffusion with ControlNet, a fixed seed, and GPT-4 for analogies, so it is not local.
- Verified (Inkspire, sections 5 and 6): in a within-subjects study with 12 designers against a ControlNet baseline, participants reported more exploration (5.83 vs 3.83) and inspiration (5.92 vs 4.00), drew fewer strokes with longer pauses, and said the scaffolding helped them plan next strokes. Design quality did not differ significantly (p = 0.06). This is the closest published precedent for S01, but it shows the whole abstracted design, not only the missing strokes.
- Verified (ShadowDraw abstract): ShadowDraw blends edge images retrieved from a database into a "shadow" under the user's strokes, updated as they draw, and its user study found more realistically proportioned drawings. Inference: it is the classic precedent for showing guidance under freehand strokes, and its database retrieval is not practical here.
- Inference: neither work infers acceptance from later strokes, so S02 has no direct precedent in these sources. Inkspire logs interactions but measures support with questionnaires.

### Part A: suggestion methods

**(1) Generate and diff.** Generate a full image with the existing sketch-conditioned backend, convert it to lines, and keep only lines away from the existing strokes.

- Measured: with the photo-style prompt "{goal}, product photo, white background", the scribble ControlNet, and 4 LCM steps, Canny edges of the output were dominated by wall and floor textures for the car and by double outlines for the airplane. Usable new strokes appeared in about 4 of 12 test cases, mostly chair legs. This matches Inkspire's finding about Canny.
- Verified (controlnet_aux and Annotators): better line extractors exist (HED, PiDiNet, the informative-drawings "lineart" model, TEED). The package is Apache 2.0 at 0.0.10, but its weights come from `lllyasviel/Annotators`, whose license tag is "other", and the informative-drawings code is MIT. Each extractor is a new dependency and a new download. Inkspire's extractor adds Segment Anything on top.
- Inference: a better extractor would clean the lines, but a photo prompt still spends the generation on materials and background that are then thrown away.

**(2) Generate directly in the sketch domain.** Prompt for a black line drawing so the output is already strokes.

- Measured, text-to-image from noise: "a simple black line drawing of {goal}, clean thin black outlines on white paper, coloring book page, minimal" with the negative "shading, color, gray, texture, photo, shadow, hatching", 4 LCM steps, and the scribble ControlNet often produced vertical bars, solid fills, or dark frames. About 4 of 12 cases were usable.
- Measured, **image-to-image from the person's own sketch** with `StableDiffusionControlNetImg2ImgPipeline`: the sketch (black on white) is the starting image, the same sketch inverted is the scribble control image, and the same line-drawing prompt is used. With strength 0.8 to 0.85, 5 scheduled LCM steps (4 run), guidance 1.5, and conditioning scale 0.8, the output stayed a clean line drawing on white paper that kept the existing lines in place and added parts in a matching line style. It produced chair legs from a seat and back, a roof, cabin, or wheels from a rectangle with the goal "a car, side view", and wings and a tail from a fuselage. About 9 of 12 cases had at least one plausible new part. This is my visual judgment on a small set, not a quality measurement.
- Measured, strength: at 0.6 and 0.7 almost nothing was added. At 1.0 the existing lines were lost and the output behaved like option 2 from noise. 0.8 to 0.9 is the useful range.
- Measured, ControlNet: without the ControlNet, the same img2img was about 25% faster (1.7 s instead of 2.2 s at 512x296) but drifted. The redrawn object no longer sat on the person's lines, so the diff produced duplicates of existing strokes or missed parts. The ControlNet is what keeps the new lines registered to the old ones.
- Measured, goal control: the same rectangle gave a cabin and wheels for "a car", a hull line and a mast for "a ship", window dividers for "a bus", and less coherent strokes for "a motorcycle", which a rectangle does not suit. The goal prompt steers the suggestion, but the start shape limits what is plausible.
- Measured, framing: a tight crop around the strokes (35% margin, giving 512x296) made the model fit the whole object into the frame, cut legs and wheels at the edge, and gave worse results than the paper-shaped 512x384 canvas. The canvas should be the paper, not the crop.
- Not tested: inpainting with a mask around the existing strokes. Inference: `StableDiffusionControlNetInpaintPipeline` would only guarantee that existing pixels stay unchanged, which the diff already handles, and a dedicated inpainting ControlNet or UNet would be a new download. The [T2I-Adapter backend](model-t2i-adapter-sketch.md) cannot be used here as is, because diffusers 0.40 has no img2img pipeline for the SD 1.5 adapter (verified: no such class is exported).

**(3) Dedicated sketch completion or stroke prediction models.**

- Verified (Magenta.js README): about 100 pretrained SketchRNN models exist, including chair, cruise_ship, speedboat, truck, and bus, but **not car, airplane, or motorbike**, three of the four study tasks. Training them needs the Quick, Draw! data (CC BY 4.0, which has all four classes) and TensorFlow, and the Magenta Python repository was archived on 2026-01-06. SketchRNN works on vector strokes as offsets, so a pencil photo would first need vectorizing, and its style is a 20-second doodle.
- Verified (abstracts): SketchGPT says code and weights "will be released", SketchMamba (58 Quick, Draw! classes, about 1.55M parameters) and StrokeFusion give no confirmed release, and AutoSketch links no code. All work on vector sketches.
- Verified (SketchAgent repository): the code is MIT, but it "requires an Anthropic API key", and other models are a to-do item. It is excluded by the no-cloud constraint. Inference: a local VLM of the size that fits beside SD 1.5 would not follow its sketching language reliably.
- Inference: none of these is usable for the study timeline without training or a cloud service.

**(4) Part-level guidance from a vision-language model.**

- Verified (installed `transformers` 5.18.0): the already-downloaded Florence-2-base supports `<CAPTION_TO_PHRASE_GROUNDING>`, `<OPEN_VOCABULARY_DETECTION>`, and `<OD>`, so it can return boxes for named parts with no new model.
- Measured: grounding a per-task part list ("wheel. windshield. roof. door. headlight.") on the generated line drawing took 0.3 to 1.2 s on MPS. It labeled the car's roof and two wheels and the chair legs correctly, but called every airplane part "cockpit" and often returned a box covering the whole image. Labels are useful as record metadata, not reliable enough to decide anything.
- Inference: naming missing parts from the partial sketch alone ("add wheels") would need an instruction-following VLM such as Qwen3-VL-2B, which needs `torchvision` and about 4.3 GB (see [prompt suggestion models](prompt-suggestion-models.md)), and text alone is not the visual suggestion S01 asks for. It could complement option 2 later.

### Part B: follow detection

Simulated with 240 trials on four real suggestions from the prototype (car, two chairs, and an airplane), each choosing one suggested part. The "person" either redrew it closely (shift about 3 px, 50-100% of it) or loosely (shift about 10 px, scale ±20%, rotation ±10°, 40-100% of it), drew other strokes elsewhere, drew a different shape where the part was, or drew nothing. Each "capture" was rendered as a photo with uneven light, soft pencil, sensor noise, JPEG at quality 75, a random camera shift of up to 4 px per corner, and a hand-like blob in 30% of trials.

- Verified (OpenCV 5.0.0 docstring): `findTransformECC(template, input, warp, motionType, criteria, inputMask, gaussFiltSize)` supports translation, Euclidean, affine, and homography motion, needs a rough initial alignment, and throws when it does not converge.
- Measured: ECC on raw grayscale failed to converge or scored about 0.58 because lighting differed between captures. On a lighting-independent "darkness" map, meaning how much darker each pixel is than a 31 px median of its surroundings, blurred with sigma 2, ECC converged in every test, with correlation 0.57 to 0.74, or 0.48 to 0.70 with a hand in view.
- Measured: a hand detector that thresholds the global median flagged the drawing itself as an occluder, because blurred strokes form large dark bands, and masking them made ECC fail. Estimating the paper level as a wide local maximum (101 px dilation then blur), thresholding pixels 30 levels darker, and keeping only blobs that survive a 25 px opening found every simulated hand and no false occluders.
- Measured, per-part scores with new strokes = current strokes minus existing strokes grown by 4 px: coverage at tolerance *t* is the share of the suggested part's skeleton within *t* px of new strokes. Precision at *t* is the share of new strokes near the part, within its bounding box grown by 32 px, that lie within *t* px of it. Chamfer is the mean capped distance from the part to the new strokes.

| Comparison (AUC, 1.0 is perfect) | coverage 8 px | coverage 16 px | precision 8 px | precision 16 px | chamfer |
| --- | --- | --- | --- | --- | --- |
| Close follow vs drawn elsewhere or a different shape in place | 0.79 | 0.76 | 0.92 | 0.92 | 0.77 |
| Loose follow vs the same negatives | 0.70 | 0.73 | 0.81 | 0.86 | 0.71 |
| Any follow vs drawn elsewhere | | 0.98 | | 0.98 | |
| Any follow vs a different shape in the same place | | 0.52 | | 0.82 | |

- Measured, a rule of `coverage_16 >= 0.35 and precision_16 >= 0.4` (undetermined when ECC < 0.3, more than half the part is occluded, or fewer than 60 new pixels): close follows 69 followed, 7 not, and 4 undetermined of 80. Loose follows 33, 4, and 3 of 40. Drawn elsewhere was 40 of 40 not followed. A different shape in the same place counted as followed 29 of 40 times. A stricter rule, `coverage_16 >= 0.3 and precision_8 >= 0.75`, cut that to 5 of 40 but missed 18 of 80 close and 16 of 40 loose follows.
- Inference: location is easy and shape is hard. The core open question is whether "followed" means drawing the suggested part where it was shown or copying its shape. A fixed pixel tolerance is also unfair to small parts, since nearly any strokes near a small wheel cover it at 16 px. Tolerances should scale with the part's size.
- Measured: scoring one capture against one suggestion took a median of 111 ms and at most 0.5 s on the CPU, mostly ECC.
- Inference: the person copies from a screen, not by tracing, so loose follows will be the common case. The pilot should hand-label real cases before a rule is fixed.

### Timings on the Apple M2 Pro

MPS, float16, with ordinary apps open and 74% of memory free before the run. Recommended setup: `StableDiffusionControlNetImg2ImgPipeline`, LCM-LoRA fused, strength 0.85, 5 scheduled steps (4 run), guidance 1.5 so the negative prompt applies, conditioning scale 0.8, one image per press.

| Stage | Time |
| --- | --- |
| Load SD 1.5, ControlNet scribble, and LCM-LoRA (once per session) | 17.7-24.6 s |
| Load Florence-2-base for part labels (once) | 1.2 s |
| First call at a new canvas size (shape warm-up) | 1.9-4.7 s extra, with outliers of 8-12 s |
| Generate one suggestion at 512x384 | 2.9-3.5 s |
| Generate one suggestion at 512x512 | 4.6 s |
| Generate four at 512x296, batched / one by one | 7.8 s total (1.9 s each) / 2.2 s each |
| Extract lines, diff, and group parts | 5-40 ms |
| Label parts with Florence-2 grounding | 0.3-1.2 s |
| Draw the overlay | under 35 ms |
| Score follow detection per capture (CPU) | median 111 ms, max 0.5 s |
| Process peak resident memory, both models loaded | 3.6 GB |

### Timings on a CPU

Measured on the same M2 Pro's CPU in float32 (8 performance and 4 efficiency cores, 8 torch threads), with the same settings:

| Stage | Time |
| --- | --- |
| Load the pipeline | 11.2 s |
| First call at a new size | 17.6-18.2 s |
| Generate one suggestion at 512x384 | 15.2-18.2 s |
| Generate one suggestion at 512x512 | 18.2-21.9 s |
| Label parts with Florence-2 | 1.5-6.0 s |
| Process peak resident memory | 9.6 GB |

The Windows laptop has not been measured. [STATUS.md](../STATUS.md) records about 35-60 s of sampling for one 4-step fast-mode image on its CPU. A suggestion runs the same 4 steps with guidance above 1 at a similar size, so expect about 35-60 s per suggestion there, which is an inference. That is not "a few seconds". Guidance 1.0, which drops the negative prompt and halves the batch, or a smaller canvas would shorten it, but neither was tested for quality. For the study, either the laptop condition accepts a wait of about a minute, or suggestions run on the M2 Pro (see open question 5).

## Options compared

| Method | Suggestion quality in the prototype | Goal control | Latency on M2 Pro | Memory beside SD 1.5 | New dependency or weights | License | Effort here |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Generate a photo, Canny, diff | Mixed, texture and background lines (about 4 of 12 usable) | Good | about 2.8 s | none | none | SD 1.5 OpenRAIL-M | Small |
| 1b. Same with HED, PiDiNet, or lineart extractor | Not tested, cleaner lines expected, Inkspire still needed SAM | Good | about 3-4 s (inference) | about 0.1-2.5 GB | `controlnet_aux` and Annotators weights, or SAM | Apache 2.0 code, weights tagged "other" | Medium |
| 2a. Line drawing from noise, threshold, diff | Unstable at 4 steps, bars and fills (about 4 of 12) | Good | about 2.8 s | none | none | SD 1.5 OpenRAIL-M | Small |
| **2b. Line drawing by img2img from the sketch with scribble ControlNet, diff** | **Best, clean strokes in the sketch's style (about 9 of 12)** | **Good, limited by the start shape** | **2.9-3.5 s at 512x384** | **none, same weights** | **none** | **SD 1.5 OpenRAIL-M, LCM-LoRA openrail++** | **Small** |
| 2c. Inpainting around the strokes | Not tested, no benefit over the diff expected | Good | about 3 s | 0-1.5 GB | inpainting ControlNet or UNet if dedicated | OpenRAIL | Medium |
| 3. SketchRNN and successors | Doodle style, three of four study classes missing | Class only | fast | small | TensorFlow, training, vectorizer | Apache 2.0, data CC BY 4.0 | Large |
| 3b. SketchAgent | Strong in its paper | Good | network bound | none | Anthropic API | MIT code, cloud | Excluded |
| 4. VLM part naming and grounding | Florence-2 labels right on car and chair, wrong on airplane | Per-task part list | 0.3-1.2 s | about 0.5 GB | none for Florence-2 | MIT | Small as labels only |

## Recommendation and confidence

**Suggestions:** use option 2b. On a button press, take the current paper-frame sketch, run the existing SD 1.5 + scribble ControlNet + LCM-LoRA weights as an img2img line drawing from the person's own sketch, extract dark lines, remove everything within a few pixels of the existing strokes, and group what remains into up to four parts, shown in distinct colors over the live sketch. It needs no new dependency or download, takes about 3 s on the M2 Pro, and is the only option whose output already looks like strokes in the person's style. Florence-2 grounding may add part names to the record, but it should not drive any decision.

**Follow detection:** compare later captures with the capture taken when the suggestion was shown, in the same paper frame, refine the alignment with ECC homography on darkness maps, mask large dark blobs as occluders, take new strokes as the difference, and score each suggested part with coverage and precision at several tolerances plus chamfer distances. Record all of them for every capture in the time window. Start with the rule `coverage_16 >= 0.35 and precision_16 >= 0.4` as provisional and tune it against hand labels from the pilot.

**Confidence:** high that 2b runs locally in about 3 s with existing weights, since it was measured. Medium on suggestion usefulness, which I judged visually on 12 cases from 4 synthetic or edited sketches, not on real participants' sketches. Medium that the follow scores separate "drew it here" from "drew elsewhere", shown on simulated captures only. Low on separating a copied shape from a different shape drawn in the same place, and on the threshold itself, which needs real hand-labeled data.

## Implementation guidance

### Where it fits

- **Capture and preprocessing:** suggestions and follow detection need a stable canvas, the **paper frame**, not the current `sketch.png`. The current `preprocess_sketch` crops to the strokes (`crop_to_drawing`), so its coordinates change as the drawing grows. Add a preprocessing output that stops after `isolate_paper` and `extract_strokes`, warped to a fixed size with the paper's aspect ratio, for example 512x362 for A4. When the paper's corners are not found, fall back to ECC against the canvas taken at suggestion time, and record that the corners were missing.
- **Suggestion (new component):** a `StrokeSuggester` abstract base class beside `Generator`, in its own module, with a `ControlNetStrokeSuggester` and a labeled fake. It reuses the diffusers loading code. Whether it shares the loaded pipeline with the generator through `from_pipe` is an architect decision. Sharing saves about 2.3 GB but ties both to the same LCM-fused UNet, which is fine in fast mode.
- **Evaluation (new component):** a `FollowDetector` abstract base class with an OpenCV-only implementation. It never loads a model, so default tests stay offline.
- **Orchestration:** `AppController` keeps the open suggestions with their baseline canvas and scores each saved capture against them. The decision is made when the window ends.
- **UI:** a "Suggest next strokes" button, and the overlay drawn on the paper-frame preview the page already refreshes, which is the simplest first step. Warping the parts back into the live camera image with the inverse paper homography can come later.
- **Experiments:** suggestion events, per-capture follow scores, and the final inference are stored in the round or session records. The `architect` designs the schema.

### Minimal outline (pinned versions as installed: torch 2.14.1, diffusers 0.40.0, opencv-python 5.0.0.93, numpy 2.5.5)

```python
# Load once, reusing the existing ModelSource folders and revisions from backends/diffusers.py.
controlnet = diffusers.ControlNetModel.from_pretrained(str(models / CONTROLNET_MODEL.folder), dtype=dtype)
pipeline = diffusers.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    str(models / BASE_MODEL.folder), controlnet=controlnet, dtype=dtype,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False,
)
pipeline.scheduler = diffusers.LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_lora_weights(str(models / LCM_LORA_MODEL.folder), weight_name=LCM_LORA_WEIGHTS)
pipeline.fuse_lora()
pipeline.to(device)

# One suggestion: the sketch is a uint8 paper-frame canvas, black strokes on white, sides divisible by 8.
strokes = (canvas < 128).astype(np.uint8) * 255
start_image = np.repeat(canvas[None, :, :, None], 3, axis=3).astype(np.float32) / 255
control_image = np.repeat(strokes[None, :, :, None], 3, axis=3).astype(np.float32) / 255
image = pipeline(
    prompt=f"a simple black line drawing of {goal}, clean thin black outlines on white paper, coloring book page, minimal",
    negative_prompt="shading, color, gray, texture, photo, shadow, hatching",
    image=start_image, control_image=control_image, strength=0.85, num_inference_steps=5, guidance_scale=1.5,
    controlnet_conditioning_scale=0.8, width=width, height=height,
    generator=torch.Generator("cpu").manual_seed(seed), output_type="np",
).images[0]

# Lines are pixels clearly darker than the local paper. New strokes are lines away from the existing strokes.
gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
lines = ((cv2.medianBlur(gray, 31).astype(np.int16) - gray > 40) | (gray < 90)).astype(np.uint8) * 255
near_existing = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
new_strokes = cv2.bitwise_and(lines, cv2.bitwise_not(near_existing))
# Drop specks under 25 px, link strokes within 10 px into parts, keep parts of at least 150 px, largest four first.
```

Follow detection, per capture:

```python
def darkness(gray):  # Lighting-independent stroke map for ECC.
    return cv2.GaussianBlur(np.clip(cv2.medianBlur(gray, 31).astype(np.float32) - gray, 0, 255), (0, 0), 2)

paper_level = cv2.GaussianBlur(cv2.dilate(current, np.ones((101, 101), np.uint8)), (0, 0), 20)
dark_blobs = cv2.morphologyEx(((paper_level.astype(np.float32) - current) > 30).astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
# Keep blobs over 1% of the image as occluders, grow them by 10 px, and pass the rest as the ECC mask.
correlation, warp = cv2.findTransformECC(darkness(baseline), darkness(current), np.eye(3, dtype=np.float32),
                                         cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                         valid_mask, 5)
aligned = cv2.warpPerspective(current, warp, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
# New strokes = strokes(aligned) minus strokes(baseline) grown by 4 px, minus occluders, minus specks under 20 px.
# Per part: coverage_t, precision_t for t in (4, 8, 16, 24) px, chamfer both ways capped at 64 px, as defined above.
```

### Parameters to record

For each suggestion event:
- Its ID, the round or session it belongs to, the wall-clock time it was shown, and the goal prompt as typed.
- The baseline canvas: its file, SHA-256, size, paper corners or the fallback used, and preprocessing steps.
- The suggester identity: kind, model IDs with revisions as `_identify_backend` builds them, device, dtype, scheduler, LoRA fused, and torch threads.
- The settings actually used: the full prompt and negative prompt, strength, scheduled and executed steps, guidance, conditioning scale, seed, and canvas size.
- The generated line drawing as a file, because it explains the suggestion and cannot be regenerated exactly on another device.
- Extraction settings: line thresholds, existing-stroke tolerance, speck size, link distance, minimum part size, and maximum parts.
- Each part: index, mask file, bounding box, pixel count, overlay color, and an optional label with the labeler's ID, revision, and vocabulary.
- Stage timings: load, warm-up, generation, extraction, labeling, and overlay.

For each capture scored against an open suggestion:
- Seconds since the suggestion was shown, the capture file and SHA-256, ECC correlation and warp, and the occluded share of the image and of each part.
- New-stroke pixels overall and near each part, coverage and precision at each tolerance, both chamfer distances, and the part's size, so a size-scaled tolerance can be computed later.

For the inference: the method name and version, the rule and thresholds, the time window and how it ended (next suggestion, task end, or timeout), which capture decided it, and the result per part and per suggestion. Because captures and parts are stored, the inference can be recomputed with any later rule.

### Reproducibility

The seed, settings, and weights make a suggestion repeatable on the same device and library versions, as for the generator. Do not claim exact replay across devices or dtypes, and keep the stored line drawing as the reference. MPS shape compilation makes the first call at a new canvas size several seconds slower, so a fixed canvas size and a warm-up call at session start keep timings comparable. Follow detection is deterministic given the stored captures.

### Pitfalls

- Do not run suggestions on the cropped `sketch.png`. Its coordinates move between captures, and a tight crop also leaves no room for the model to add parts.
- Do not drop parts that touch the canvas edge. Clip them instead, since legs and wheels often reach the edge.
- Strength below 0.8 adds almost nothing, and 1.0 discards the existing lines.
- Keep the ControlNet. Without it the redrawn object drifts away from the person's lines.
- Guidance must stay above 1 for the negative prompt to apply, which doubles the batch. The current `fast` mode in the generator marks negative prompts as unsupported, so the suggester needs its own settings.
- Stray long straight lines and frame lines appear in some outputs. Consider dropping parts that are nearly straight and span most of the canvas.
- Similarity scores such as coverage, chamfer, or CLIP are research signals. A followed suggestion is not evidence that the design improved.
- A hand over the paper masks part of the drawing. Decide on the latest unoccluded capture in the window rather than the last capture.

### Test ideas

- **Fake suggester:** returns deterministic parts, for example a circle under the bounding box of the existing strokes, labeled as fake in its identity, so the UI, records, and follow detection are tested without weights.
- **Suggester unit tests with a stub runner:** as the diffusers backend tests do, assert the pipeline kwargs (start image polarity, control image polarity, strength, steps, guidance, conditioning scale, sizes divisible by 8) and that the diff removes lines within the tolerance of existing strokes on synthetic images.
- **Part grouping:** synthetic line images with known separate shapes give the expected number, order, and boxes of parts.
- **Follow detection:** port the simulation as parametrized tests: a closely redrawn part is followed, strokes elsewhere are not followed, no new strokes is undetermined, a hand blob is masked and not counted as strokes, and a shifted capture still aligns. Keep a seeded random generator so the tests are deterministic.
- **Records:** a session with a fake suggestion and three captures round-trips through the store, and the inference can be recomputed from the stored files with a different threshold.
- **Model test, marked:** load the real weights, suggest for a synthetic rectangle with the goal "a car, side view" at 512x384, and check for non-empty new strokes within a time bound.

## Open questions for the owner

1. **What counts as following:** drawing the suggested part where it was shown, in any shape, or copying its shape? The simulation shows the location test is reliable and the shape test is not. The study's definition decides the rule.
2. **What the person sees per press:** one suggestion with up to four colored parts, with another press for another seed, or several alternatives at once? One per press keeps latency near 3 s and makes acceptance easier to attribute.
3. **Time window:** until the next suggestion request or the task's end, or a fixed limit such as 60 or 120 s? Recording every capture leaves this open, but the pilot protocol should fix it.
4. **No drawing at all:** if the person draws nothing after a suggestion, is it not followed, as the meeting example suggests, or undetermined?
5. **Windows laptop:** see the CPU timings. Either accept slow suggestions there, or run suggestions on the M2 Pro. The browser can use the webcam only on `localhost` or HTTPS ([MDN secure contexts](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts)), so a laptop browser on a plain-HTTP LAN page of the M2 Pro cannot use its camera. The options are HTTPS with a local certificate, the camera attached to the M2 Pro, or a remote suggestion service called from the laptop, which is a larger change.
6. **Part names:** should each task have a fixed part list for the Florence-2 labels (for a car: wheel, windshield, roof, door, headlight), or should labels be dropped until they are reliable?
7. **Hand labels:** who labels pilot suggestions as followed or not, and how many, so the threshold can be fixed before the five participants?
