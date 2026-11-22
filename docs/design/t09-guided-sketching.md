# T09 design: next-stroke suggestions, drawing captures, and follow detection

Status: accepted by the owner for the T09 build, with the records unit and experiment names in [ADR 0025](../decisions/0025-drawings-as-the-unit-of-guided-sketching.md) (accepted). Applies [ADR 0024](../decisions/0024-next-stroke-suggestions-and-follow-detection.md) and serves S01, S02, and S04 in [PROJECT.md](../PROJECT.md), with record changes under R06, the replaceable component under R08, and stage timings under R09. Runnable increment, from [ROADMAP.md](../ROADMAP.md): `sketchloop ui` suggests next strokes, and `sketchloop show` lists each suggestion and whether it was followed. Nothing below exists in `src/` yet. Every class, function, and file named here is proposed.

Impact: High. It adds a component boundary (`stroke_suggestion` with its `stroke_suggesters` package), a follow-detection module, a drawing orchestrator, a new record file and folder in every session that uses them, experiment names for session folders, and page controls. It adds one optional field, `name`, to `session.json`, changes no existing record field, keeps `SCHEMA_VERSION` at 1, and leaves `SketchSession.run_round`, `round.json`, `rerun`, and the generator contract as they are. The affected boundaries are `preprocessing`, `generation` (one extracted helper), `components`, `settings`, orchestration (`session`, a new `drawing`, `app`), `experiments`, `cli.show`, `cli.serve`, and `ui`.

## Decisions this design applies

From [ADR 0024](../decisions/0024-next-stroke-suggestions-and-follow-detection.md), unchanged:

1. A press runs `StableDiffusionControlNetImg2ImgPipeline` on the paper-frame sketch with the scribble ControlNet and LCM-LoRA at strength about 0.85, keeps dark lines away from existing strokes, and groups them into up to four colored parts shown over the live sketch.
2. One suggestion per press, and each part is scored separately.
3. Followed means the same part in the same place, judged by coverage and precision of the new strokes against the part.
4. A suggestion's window runs from when it is shown until the next suggestion or the end of the task. Nothing drawn gives undetermined.
5. Every capture's alignment and per-part scores at several tolerances are recorded. The provisional rule is `coverage_16 >= 0.35 and precision_16 >= 0.4`.
6. Part names are metadata only, tasks have no fixed part lists, and the suggestion component is replaceable through the settings.
7. The study runs on the M2 Pro, and Windows and Linux must still install and run it without a speed requirement.

The owner's answers to the design's questions:

1. **Drawing unit:** approved, so [ADR 0025](../decisions/0025-drawings-as-the-unit-of-guided-sketching.md) is accepted.
2. **Captures:** every 2 s, with interval frames stored as JPEG at quality 90 and the frames at each suggestion and at finish as PNG.
3. **Paper:** A4 in landscape, giving a 512x360 canvas, kept as a setting.
4. **No goal:** the system is never told the task. The task is given to the participant, who writes the prompt and makes the sketch. There is no goal input, and a suggestion is conditioned only on whatever the participant typed in the existing prompt box, which may be empty.
5. **Experiment names:** a whole experiment's results are saved under a name the experimenter enters, such as `participant1_car`, which is the only study label the system stores.
6. **Defaults accepted:** the suggestion-level result rule below, part labels waiting until after item 33, and the extra memory of a separate suggestion pipeline.

Where this design narrows or defers something in ADR 0024, it says so in [Conflicts and deferrals](#conflicts-and-deferrals).

## Current state

- `SketchSession` (`session.py`) runs numbered rounds and writes `session.json` lazily before round 1 through `ExperimentStore.save_session`. Each round is a generation iteration with candidates, written as `round-<n>/round.json`. `load_session` reads only `round-<n>` folders and ignores any other folder.
- `AppController` (`app.py`) holds one generator, the `SketchSession`, and a `PromptSuggester` from `suggestion.py`, which suggests **style prompts**, not strokes. The page keeps the newest webcam frame every `ui.capture_interval_seconds` (0.5 s) and refreshes a preview of `sketch.png` on a `gr.Timer`.
- `preprocess_sketch` crops to the strokes (`crop_to_drawing`), so its coordinates move as the drawing grows. `isolate_paper` straightens the sheet by its corners, or by its edges when a corner is hidden, and whitens everything outside it, but warps to the paper's measured size, which differs between captures.
- `DiffusersSketchGenerator` loads its pipeline lazily on the first `generate`. A session that never generates never loads it. `ModelSource` constants, `select_device`, `TorchPrecision`, `describe_torch_runtime`, `local_revision`, `encode_rgb_png`, `PipelineRunner`, and `DiffusersRunner` are module-level and reusable. `scripts/download_models.py` already fetches SD 1.5, the scribble ControlNet, and LCM-LoRA, which are all the real suggester needs.

## Options

### Which unit suggestions and captures belong to

| Option | Description | Tradeoffs |
| --- | --- | --- |
| A | Inside rounds, as lists in `round.json` | A round is one generation iteration with a request, candidates, and lineage checks. Suggestions and captures happen between rounds, a no-AI study drawing has no round at all, and a drawing with 400 captures would spread over unrelated rounds. |
| B | On the session, in `session.json` or a session-level timeline file | `session.json` is written once and describes the environment. In T10 a participant's session holds several tasks, so one timeline would mix tasks and conditions and need splitting later. |
| **C, chosen** | A new **drawing** unit, `drawing-<n>/drawing.json` beside `round-<n>/`, holding one drawing from its start to its end with its captures, suggestions, follow scores, and decisions | Matches what S04 calls "each task ends with a final drawing", so T10 builds on drawings and experiment names without moving anything. Older sessions have no drawing folders, and older code ignores them. Costs one new record type and loader. |

### How a drawing's record is written

| Option | Tradeoffs |
| --- | --- |
| **One `drawing.json`, rewritten atomically after each change, recommended** | The pattern `round.json` already uses, with one loader and one set of invariants. A 15-minute drawing at 2 s is about 450 captures and under 1 MB of JSON, so each rewrite takes milliseconds. |
| An append-only `captures.jsonl` plus a small `drawing.json` | Constant cost per write, but a second format, a partial last line to tolerate on load, and invariants split across two files. Worth it only if rewrites become measurably slow. |
| One JSON file per capture | Each written once, but hundreds of small files, each needing its own schema version. |

### The suggestion pipeline and the generator's weights

| Option | Memory with both loaded | Tradeoffs |
| --- | --- | --- |
| **A separate img2img pipeline owned by the suggester, recommended** | About 2.3 GB more in float16 than sharing, only when the person also generates in the same process. The suggester alone peaked at 3.6 GB with Florence-2 on the M2 Pro and 9.6 GB on a CPU in float32. | The suggester never depends on the generator, so it works beside the fake backend, the T2I-Adapter backend, or quality mode. In the study nobody needs to press Generate, and the lazy generator then never loads. |
| Share through `StableDiffusionControlNetImg2ImgPipeline.from_pipe(generator_pipeline)` | Saves about 2.3 GB | Couples the suggester to one backend's private pipeline object and its LCM-fused UNet, so it fails with the fake backend, quality mode, or T2I-Adapter, and needs a new accessor on the generator. Revisit only if memory becomes a measured problem. |

### When the page saves captures

| Option | Tradeoffs |
| --- | --- |
| **Every `drawing.save_interval_seconds` (default 2 s) while a drawing is open, plus one at each suggestion press and at finish, recommended** | Simple, predictable, and keeps the hand-over-paper frames, which follow detection needs to mask anyway. Storage is set by the interval (see [Storage estimate](#storage-estimate)). |
| Only when the frame changes beyond a threshold | Fewer files, but [ADR 0019](../decisions/0019-offline-prompt-suggestions.md) found camera noise made change detection unreliable, and a missed faint stroke cannot be recovered. |

### Where the experiment name lives

| Option | Tradeoffs |
| --- | --- |
| **The session, chosen** | The session folder is what a person finds in `runs/`, so a named session is `runs/participant1_car/` and `sketchloop show runs/participant1_car` works by name. Uniqueness in the runs folder is one exclusive folder creation. The rounds the participant generates during the experiment land in the same folder as the drawing. `session.json` already holds the environment and hardware that describe the whole experiment. |
| The drawing | A name such as `drawing-1-participant1_car` would be found only by searching every session folder, uniqueness in the runs folder would need that search on every start, and rounds generated during the experiment would sit in a folder named by time. |

## Contracts

### Stroke suggestion component (`sketchloop.stroke_suggestion`, new)

The name says strokes everywhere, so it is not confused with `PromptSuggester` in `suggestion.py`. Records live next to the interface, as `GenerationOutput` lives in `generation.py`.

```python
MAX_PARTS: Final = 4
PART_COLORS: Final = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")  # Okabe-Ito colors by part index, readable with color blindness


class StrokeSuggestionError(SketchLoopError, RuntimeError): ...  # Cannot load the suggestion model, or it fails while suggesting


@dataclass(frozen=True, slots=True, kw_only=True)
class StrokeSuggestionRequest:
    canvas: ImageRef  # Paper-frame strokes the suggester receives, black on white, mode L, sides divisible by 8
    prompt: str  # Prompt box text exactly as the participant typed it, possibly empty
    seed: int | None = None  # Seed, or None for a random one that the result records
    controls: Mapping[str, ControlValue] = field(default_factory=dict)  # Requested controls, empty for the defaults
    max_parts: int = MAX_PARTS  # Most parts to return, from 1 to MAX_PARTS


@dataclass(frozen=True, slots=True, kw_only=True)
class SuggestedPart:
    index: int  # 0-based position, largest part first
    mask: ImageRef  # Mode L PNG at the canvas size, 255 on the part's strokes
    box: tuple[int, int, int, int]  # Left, top, width, and height in canvas pixels
    pixel_count: int  # Pixels in the mask, at least 1
    skeleton_pixels: int  # Pixels in the mask's one-pixel skeleton, so a size-scaled tolerance can be computed later
    label: str | None = None  # Part name from a labeler, always None in T09


@dataclass(frozen=True, slots=True, kw_only=True)
class StrokeSuggestion:
    suggester: BackendIdentity  # Adapter, version, execution kind, and model IDs with revisions
    effective: EffectiveSettings  # Full prompt, negative prompt, and every setting and extraction parameter used
    seed: int | Unavailable  # Seed actually used
    line_drawing: ImageRef | Unavailable  # The model's whole output, kept because it cannot be regenerated exactly elsewhere
    parts: tuple[SuggestedPart, ...]  # 0 to max_parts new parts, largest first
    timings_seconds: Mapping[str, float]  # generation and extraction, without any model load


@dataclass(frozen=True, slots=True, kw_only=True)
class StrokeSuggestionOutput:
    suggestion: StrokeSuggestion  # The suggestion record
    payloads: Mapping[str, bytes]  # Bytes of the line drawing and each mask by path, checked against their checksums
    load_seconds: float | None = None  # Seconds this call spent loading models, None when it loaded none


class StrokeSuggester(ABC):
    @property
    @abstractmethod
    def controls(self) -> tuple[ControlSpec, ...]: ...  # Controls with bounds and defaults, such as strength and steps

    @abstractmethod
    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput: ...

    @abstractmethod
    def describe(self) -> str: ...  # One sentence naming the method, model, device, and precision

    def load(self, width: int, height: int) -> None: ...  # Load and warm up at the canvas size, which suggesters without a model skip
```

Invariants, checked in `__post_init__` with `require`:

- The request prompt is any text, including empty, and is stored unchanged. The seed is None or an int of at least 0, `max_parts` is from 1 to `MAX_PARTS`, and the canvas is mode L with both sides divisible by 8.
- Part indexes are 0 to n-1 in order, n is at most the request's `max_parts`, every mask has the same size, boxes lie inside it, and counts are at least 1.
- `StrokeSuggestionOutput` requires one payload per mask and per line drawing path, each matching its checksum, as `GenerationOutput` does. `GuidedDrawing` also checks that every mask has the request canvas's size, since the output does not hold the request.
- Paths are bare file names, `line-drawing.png` and `part-<index + 1>.png`. `GuidedDrawing` places them under `suggestions/suggestion-<k>/`, as `SketchSession` places candidates under the round folder.
- `suggest` raises `UnsupportedConfigurationError` for unknown or out-of-range controls before loading anything. `generation.py` gains `control_issues(controls, specs) -> tuple[ValidationIssue, ...]`, extracted unchanged from the control loop of `request_issues`, which then calls it.
- No parts is a valid result. It is shown as "No new strokes suggested" and recorded like any suggestion.

### Fake suggester (`sketchloop.stroke_suggesters.fake`, new)

`FakeStrokeSuggester(StrokeSuggester)` with `ADAPTER_NAME = "sketchloop.fake-strokes"` and `execution=ExecutionKind.FAKE`, so `suggester.is_fake` labels it on the page and in `show`. It finds the bounding box of the dark pixels in the canvas and returns two parts: a ring centred below the box and a short horizontal line to its right, both clipped to the canvas. With no strokes it centres them on the canvas. Positions depend only on the canvas, so tests know exactly where the parts are. The seed and prompt are recorded and otherwise unused. `line_drawing` is the canvas with both parts drawn on it. Its controls are empty.

### Real suggester (`sketchloop.stroke_suggesters.controlnet`, new)

`ControlNetStrokeSuggester(StrokeSuggester)` with `ADAPTER_NAME = "sketchloop.controlnet-strokes"`, `ADAPTER_VERSION = "1"`, and constructor `(models_dir: Path = Path("models"), *, device: TorchDevice | None = None, precision: TorchPrecision | None = None, runner: PipelineRunner | None = None)`.

- **Loading:** lazy on first `load` or `suggest`, under a `threading.Lock` like `FlorenceSuggester`. It reuses `BASE_MODEL`, `CONTROLNET_MODEL`, `LCM_LORA_MODEL`, `LCM_LORA_WEIGHTS`, `select_device`, `describe_torch_runtime`, `local_revision`, `encode_rgb_png`, and `DiffusersRunner` from `backends/diffusers.py`. It builds `StableDiffusionControlNetImg2ImgPipeline`, switches to `LCMScheduler`, loads and fuses the LCM-LoRA, and moves to the device. The missing-folder check in `DiffusersSketchGenerator._load_runner` moves into a module function `require_model_folders(models_dir, sources) -> None` in `backends/diffusers.py`, which both call, with the same message. Precision defaults like the generator: float16 on a GPU, float32 on the CPU, which refuses float16.
- **Warm-up:** `load(width, height)` loads, then runs one suggestion on a blank canvas of that size and discards it, because the first call at a new size on MPS took 2 to 12 s longer.
- **Controls and defaults, from the research note:** `strength` 0.85 (0.5 to 1.0), `steps` 5 (1 to 20, scheduled, with about 4 run), `guidance_scale` 1.5 (1.0 to 5.0, which must stay above 1 for the negative prompt to apply), and `conditioning_scale` 0.8 (0 to 2).
- **Prompt:** with a non-blank participant prompt, the research template `"a simple black line drawing of {prompt}, clean thin black outlines on white paper, coloring book page, minimal"`, with `{prompt}` stripped of surrounding spaces. With a blank prompt, the neutral `"a simple black line drawing, clean thin black outlines on white paper, coloring book page, minimal"`. The negative prompt is always `"shading, color, gray, texture, photo, shadow, hatching"`.
- **What is recorded about the text:** `request.prompt` keeps the participant's text unchanged, `effective.prompt` and `effective.negative_prompt` keep the exact text passed to the pipeline, and `effective.controls` adds `prompt_template` (`with_prompt` or `neutral`), `prompt_tokens` counted with the pipeline's tokenizer, and `prompt_truncated`, true when the text exceeded the 77-token limit and its end was cut. Long style prompts are the participant's choice, so the cut is recorded rather than prevented.
- **Call:** the canvas (black on white) as the start image, the canvas inverted as the control image, `width` and `height` equal to the canvas size, and a CPU `torch.Generator` seeded per call through `DiffusersRunner`.
- **Extraction (`sketchloop.stroke_suggesters.line_parts`, new, NumPy and OpenCV only):** `extract_new_lines(line_drawing_gray, canvas_strokes, params) -> np.ndarray` keeps pixels darker than the local median by 40 or below 90, minus the existing strokes grown by 13 px. It also removes, in a 9 px band, every straight segment that `cv2.HoughLinesP` finds on the whole drawing's lines, grown by 3 px, that is at least 0.4 of the canvas's longer side with gaps up to 0.04 of it and has at most half its length within reach of the existing strokes. This drops ruled paper, floors, and horizons before the existing strokes cut them into short pieces, while a leg that continues an existing post stays. `group_parts(new_lines, params, max_parts) -> list[np.ndarray]` drops specks under 25 px, links strokes within 10 px, keeps groups of at least 150 px, drops a group that is nearly straight and spans most of the canvas, clips at the canvas edge without dropping parts that touch it, and returns the largest `max_parts`. Every threshold is a field of a frozen `LinePartParams` dataclass and is written into `effective.controls` with an `extract_` prefix.
- **Effective settings** add `device`, `dtype`, `scheduler`, `lcm_lora_fused`, `width`, `height`, `executed_steps`, and `torch_threads`. `suggester.model_id` joins the three repository IDs with their local revisions, as `_identify_backend` does.
- **Failures:** a missing folder or a load error raises `StrokeSuggestionError` with the folder and "Run python scripts/download_models.py and retry". A pipeline error raises `StrokeSuggestionError("Next-stroke suggestion failed: <first line>. Free memory and retry.")`.

### Registry and settings (`sketchloop.components`, `sketchloop.settings`)

```toml
[stroke_suggester]
name = "fake"            # fake or controlnet

[stroke_suggester.controlnet]
precision = "float16"    # optional, float16 on a GPU and float32 on the CPU by default

[drawing]
canvas_width = 512       # paper frame width in pixels, divisible by 8
canvas_height = 360      # 512 x 360 is A4 or Letter in landscape, rounded to the latent grid
save_interval_seconds = 2.0
```

- `STROKE_SUGGESTERS: Final[dict[str, StrokeSuggesterFactory]]` holds `"fake"` and `"controlnet"`, built with `settings.models.dir` and the precision. `make_stroke_suggester(settings)` creates one, and `check_component_names` checks `stroke_suggester.name` and `stroke_suggester.controlnet.precision`.
- The default is `"fake"`, like `generator.backend`, so default checks need no weights. The study's settings file names `"controlnet"` explicitly, which also documents the method used.
- `MINIMUMS` and `check_range` refuse canvas sides that are not multiples of 8 or are outside 256 to 1024, and a save interval of 0.5 s or less, since frames arrive only every `ui.capture_interval_seconds`.
- `scripts/download_models.py` needs no change, because it already fetches all three models.

### Paper-frame canvas (`sketchloop.preprocessing`)

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class PaperCanvas:
    image: ImageRef  # Grayscale paper frame at the canvas size, path "canvas.png" until the caller places it
    payload: bytes  # PNG bytes of the image
    steps: tuple[PreprocessingStep, ...]  # grayscale, rotate, isolate_paper, crop_to_paper when used, resize_to_canvas


def make_paper_canvas(payload: bytes, rotation: int, width: int, height: int) -> PaperCanvas: ...
def extract_canvas_strokes(canvas_payload: bytes) -> PreprocessedSketch: ...  # Black strokes on white, path "strokes.png"
```

- `make_paper_canvas` reuses `decode_grayscale`, `rotate_image`, and `isolate_paper` unchanged. With the corners found, the isolated image is already the warped sheet. Without them, it crops to the bounding box of the paper mask, recorded as `crop_to_paper`. Either way it resizes to exactly `width` x `height` with area interpolation, recorded as `resize_to_canvas`. The canvas keeps every stroke on the paper. It never calls `extract_strokes`, whose reach filter would hide strokes drawn away from the main drawing, and never calls `crop_to_drawing`.
- A sheet of another aspect ratio is stretched to the canvas. Every capture of a drawing is stretched the same way, so alignment is unaffected, and the model sees a slightly stretched sketch.
- `extract_canvas_strokes` marks pixels darker than the local paper with `mark_darker_pixels`, drops specks under the minimum stroke area, and thickens them with `thicken_strokes`. It is the suggester's input and is stored with the suggestion.
- Later captures go through the same function at the same size, so the corner warp maps them into the same frame. Follow detection then refines the remaining shift with ECC against the suggestion's baseline canvas.
- An image that cannot be decoded raises `InvalidSketchError`. A frame without a sheet still gives a canvas, with `paper_found: false` in the steps.

### Follow detection (`sketchloop.follow`, new, NumPy and OpenCV only)

```python
TOLERANCES_PIXELS: Final = (4, 8, 16, 24)
CHAMFER_CAP_PIXELS: Final = 64


class FollowResult(StrEnum):
    FOLLOWED = "followed"
    NOT_FOLLOWED = "not_followed"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True, kw_only=True)
class Alignment:
    converged: bool  # Whether ECC converged
    correlation: float | Unavailable  # ECC correlation, unavailable when it did not converge
    warp: tuple[tuple[float, float, float], ...] | None  # 3x3 homography from the capture to the baseline, row by row
    occluded_share: float  # Share of the canvas masked as an occluder, such as a hand


@dataclass(frozen=True, slots=True, kw_only=True)
class PartScores:
    part: int  # Part index
    occluded_share: float  # Share of the part's pixels under an occluder
    new_pixels_near: int  # New-stroke pixels inside the part's box grown by 32 px
    coverage: Mapping[int, float]  # Share of the part's skeleton within t px of new strokes, by tolerance t
    precision: Mapping[int, float | Unavailable]  # Share of nearby new strokes within t px of the part, unavailable with none near
    chamfer_part_to_new: float | Unavailable  # Mean distance from the skeleton to new strokes, capped at 64 px
    chamfer_new_to_part: float | Unavailable  # Mean distance from nearby new strokes to the part, capped at 64 px


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureScores:
    suggestion: int  # Number of the suggestion whose window the capture fell in
    seconds_since_shown: float  # Seconds from when the suggestion was shown to the capture
    alignment: Alignment  # Alignment to the suggestion's baseline canvas
    new_stroke_pixels: int | None  # New-stroke pixels on the whole canvas, None when alignment failed
    parts: tuple[PartScores, ...]  # One per part in index order, empty when alignment failed


@dataclass(frozen=True, slots=True, kw_only=True)
class FollowRule:
    coverage_tolerance_pixels: int = 16
    min_coverage: float = 0.35
    precision_tolerance_pixels: int = 16
    min_precision: float = 0.4
    min_correlation: float = 0.3
    max_part_occluded_share: float = 0.5
    min_new_stroke_pixels: int = 60


@dataclass(frozen=True, slots=True, kw_only=True)
class PartDecision:
    part: int  # Part index
    result: FollowResult  # The part's result
    deciding_capture: int | None  # Capture whose scores decided it, None when no capture was usable
    reason: str | None  # Why it is undetermined, such as "nothing new drawn", otherwise None


@dataclass(frozen=True, slots=True, kw_only=True)
class FollowDecision:
    method: str  # Detector name, such as "aligned-strokes"
    method_version: str  # Detector version, bumped whenever its scores change
    rule: FollowRule  # Rule that produced the results
    result: FollowResult  # The suggestion's result
    parts: tuple[PartDecision, ...]  # One per part in index order


class FollowDetector(ABC):
    @property
    @abstractmethod
    def method(self) -> tuple[str, str]: ...  # Name and version

    @abstractmethod
    def describe_params(self) -> Mapping[str, ControlValue]: ...  # Every threshold, recorded once per drawing

    @abstractmethod
    def score(self, baseline_canvas: bytes, capture_canvas: bytes, part_masks: Sequence[bytes]) -> tuple[Alignment, int | None, tuple[PartScores, ...]]: ...  # noqa: E501


class AlignedStrokeFollowDetector(FollowDetector): ...  # method ("aligned-strokes", "1")


def decide_follow(window: Sequence[tuple[int, CaptureScores]], part_count: int, rule: FollowRule, method: tuple[str, str]) -> FollowDecision: ...  # noqa: E501
```

`AlignedStrokeFollowDetector.score`, item 30 then item 31, follows the research note:

1. **Darkness map:** `make_darkness_map(gray)` is the 31 px median minus the image, clipped at 0 and blurred with sigma 2.
2. **Occluders:** `find_occluders(gray)` takes the paper level as a 101 px dilation blurred with sigma 20, marks pixels 30 levels darker, opens them with a 25 px ellipse, keeps blobs over 1% of the canvas, and grows them by 10 px.
3. **Alignment:** `align_to_baseline(baseline, capture, baseline_mask, capture_mask)` runs `cv2.findTransformECCWithMask` with `MOTION_HOMOGRAPHY` on the two darkness maps, starting from the identity, with 100 iterations, eps 1e-5, and Gaussian size 5. `cv2.error` gives `converged=False`, no warp, and no part scores. It never raises. Since version 3 of the detector, `choose_alignment_pixels` limits the baseline mask to pixels within 16 px of its visible strokes and removes from the capture mask its strokes more than 12 px from every baseline stroke, grown by 8 px, so new strokes cannot drag the warp. A baseline with fewer than 1000 visible stroke pixels uses the whole visible canvas instead, and `aligned_on` records which was used. Both masks always leave out occluders and filled areas.
4. **New strokes:** strokes in the aligned capture, minus the baseline's strokes grown by 4 px, minus occluders, minus specks under 20 px. Strokes are darkness above `LOCAL_STROKE_CONTRAST`.
5. **Per part:** coverage and precision at every tolerance in `TOLERANCES_PIXELS`, both chamfer distances, the part's occluded share, and the nearby new pixels, as typed above.

`decide_follow` is pure, so a stored drawing can be decided again with any rule:

- For each part, a capture in the window is usable when it converged, its correlation reaches `min_correlation`, and the part's occluded share is at most `max_part_occluded_share`.
- No usable capture gives undetermined with "no usable capture".
- Otherwise the latest usable capture decides, since new strokes are measured against the baseline and accumulate. Fewer than `min_new_stroke_pixels` on the whole canvas gives undetermined with "nothing new drawn". Coverage at the coverage tolerance of at least `min_coverage`, and a numeric precision at the precision tolerance of at least `min_precision`, gives followed. Anything else gives not followed.
- **Suggestion result, accepted by the owner:** followed when any part is followed, otherwise not followed when any part is not followed, otherwise undetermined. A suggestion with no parts is undetermined with "no parts suggested".
- `PROVISIONAL_RULE = FollowRule()` is the rule in force. Changing it after the pilot is a code change recorded in an ADR, and older decisions keep the rule they were made with.

### Drawing orchestration (`sketchloop.drawing`, new, and `sketchloop.session`)

`SketchSession` gains `ensure_session_record()`, which is today's `_write_session_record` made public, a `drawings: list[GuidedDrawing]`, and:

```python
def start_drawing(self, suggester: StrokeSuggester | None, detector: FollowDetector, *, canvas_width: int,
                  canvas_height: int) -> GuidedDrawing: ...  # None means suggestions are off, as in T10's no-AI condition
```

It refuses to start while another drawing of the session is open, numbers drawings from 1, and saves `session.json` first.

### Experiment names (`sketchloop.experiments`, `sketchloop.session`)

```python
EXPERIMENT_NAME_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
WINDOWS_DEVICE_NAMES: Final = frozenset({"con", "prn", "aux", "nul", *(f"com{digit}" for digit in range(1, 10)),
                                         *(f"lpt{digit}" for digit in range(1, 10))})


class ExperimentNameError(SketchLoopError, ValueError): ...  # The name breaks the naming rules
class ExperimentNameTakenError(StoreError): ...  # A folder of that name already exists in the runs folder


def check_experiment_name(name: str) -> str: ...  # Returns the name with surrounding spaces stripped, or raises ExperimentNameError
```

- **Rules:** 1 to 64 characters of ASCII letters, digits, `_`, and `-`, starting with a letter or digit, and not a Windows device name in any case. This excludes path separators, dots and so `..`, spaces, drive letters, and anything a folder name on Windows, macOS, or Linux would refuse. One line names the rule: `Experiment name must be 1 to 64 letters, digits, "_" or "-", starting with a letter or digit. Use a code such as participant1_car.`
- **Uniqueness:** `ExperimentStore` gains `claim_session(session_id: str) -> None`. `FolderStore` refuses a name equal, ignoring case, to any existing entry of the runs folder, so the check behaves the same on case-insensitive macOS and Windows and on Linux, then creates the folder with `mkdir(exist_ok=False)`, which also catches a race. `MemoryStore` checks its own sessions and claims. Both raise `ExperimentNameTakenError("An experiment named participant1_car already exists in the runs folder. Choose another name.")`.
- **Session:** `SketchSession.__init__` gains `name: str | None = None`. A named session checks the name, uses it as `session_id` and so as its folder name, and calls `store.claim_session` in the constructor before anything else is saved. An unnamed session keeps today's time-based folder name.
- **Record:** `SessionRecord` gains `name: str | None  # Experiment name the experimenter entered, also the folder name, None for a time-based folder`. Its invariant is that a name passes the rules and equals `session_id`. `session.json` writes `"name": "participant1_car"` after `session_id`, and older sessions load it as null, so no migration is needed.
- **Privacy:** the name is the only study label the system stores. The page hints that it must be a code and never a real name. Names stay in the ignored `runs/` folder, and any later export must not copy them, because they may encode a participant code and a task.
- **Nothing else encodes the task:** apart from the experiment name the experimenter types and the prompt the participant types, no field in `session.json`, `round.json`, or `drawing.json` and no page text holds a task or goal, and the suggester is never told one.

```python
class CaptureTrigger(StrEnum):
    INTERVAL = "interval"
    SUGGESTION = "suggestion"
    FINISH = "finish"


class DrawingEnd(StrEnum):
    FINISHED = "finished"
    NEW_SESSION = "new_session"


class WindowEnd(StrEnum):
    NEXT_SUGGESTION = "next_suggestion"
    FINISHED = "finished"
    NEW_SESSION = "new_session"


class GuidedDrawing:
    def record_capture(self, captured: CapturedSketch, *, source: SketchSource, rotation: int,
                       frame_age_seconds: float | None = None, trigger: CaptureTrigger = CaptureTrigger.INTERVAL) -> CaptureRecord: ...
    def suggest(self, captured: CapturedSketch, *, source: SketchSource, rotation: int, prompt: str, seed: int | None = None,
                controls: Mapping[str, ControlValue] | None = None, frame_age_seconds: float | None = None) -> SuggestionRecord: ...
    def finish(self, captured: CapturedSketch | None, *, source: SketchSource | None = None, rotation: int = 0,
               frame_age_seconds: float | None = None, end: DrawingEnd = DrawingEnd.FINISHED) -> DrawingRecord: ...
    def compose_preview(self, canvas: PaperCanvas) -> bytes: ...  # RGB PNG of the canvas with the open suggestion's overlay
    @property
    def record(self) -> DrawingRecord: ...
    @property
    def open_suggestion(self) -> SuggestionRecord | None: ...
```

- **Capture:** make the canvas, then, when a suggestion's window is open, score it against that suggestion's baseline canvas. Save the raw frame and canvas files, then rewrite `drawing.json`. Times are `at`, wall-clock UTC, and `seconds_since_start` from a monotonic clock started with the drawing.
- **Suggest, in this order:**
  1. Take a capture with trigger `suggestion`, scored against the open suggestion if any.
  2. Refuse a canvas with no strokes with `InvalidSketchError("No strokes found on the paper. Draw something, then ask again.")`. The capture stays, and no suggestion is recorded.
  3. Run the suggester on `extract_canvas_strokes(canvas)`.
  4. Place its files, draw the overlay, and save.
  5. Only now end the previous suggestion's window with `next_suggestion` at that capture and record its `decide_follow` result. A failed press therefore shows nothing new and leaves the previous window open.
- **Failure:** any exception from step 3 or 4 records a failed suggestion with its stage and message, saves `drawing.json`, and re-raises, as `run_round` does.
- **Finish:** take a capture with trigger `finish` when a frame is given, which becomes `final_capture`. End the open window with `finished` or `new_session`, record its decision, set `ended_at` and `end`, and save. A finished drawing refuses further calls.
- **Overlay:** an RGBA PNG at the canvas size, with each part's mask grown by 2 px in its `PART_COLORS` color and the rest transparent. It is stored once per suggestion as `overlay.png`, which is what the person saw. `compose_preview` blends it at 80% over the given canvas.
- **Thread safety:** a `threading.Lock` guards every method, because the page's capture timer and the Suggest button run in different queues.

### Records (`sketchloop.experiments.drawings`, new)

The types mirror the JSON below, as frozen keyword-only dataclasses with `require` checks: `CaptureRecord`, `StageFailure(stage, error, message)`, `SuggestionWindow(ended_at, end, last_capture)`, `SuggestionRecord`, and `DrawingRecord`. The module also holds `drawing_to_dict`, `drawing_from_dict`, `DRAWING_FILE = "drawing.json"`, and the stage enum `SuggestionStage` with `canvas`, `model_load`, `generation`, `extraction`, `overlay`, and `saving`. It reuses `encode_value`, `decode_value`, `image_from_dict`, and the `BackendIdentity` and `EffectiveSettings` parts of `result_to_dict` and `result_from_dict`, which are split into `backend_to_dict` and `effective_to_dict` and their inverses.

Folder layout, inside the session folder beside `round-<n>/`:

```
drawing-1/
  drawing.json
  captures/capture-0001-raw.png      lossless for suggestion and finish captures
  captures/capture-0002-raw.jpg      JPEG quality 90 for interval captures (owner decision 2)
  captures/capture-0001.png          paper-frame canvas, grayscale, at the canvas size
  suggestions/suggestion-1/strokes.png        the suggester's input
  suggestions/suggestion-1/line-drawing.png
  suggestions/suggestion-1/part-1.png ... part-4.png
  suggestions/suggestion-1/overlay.png
```

`drawing-<n>/drawing.json`, with `...` marking a full `ImageRef`, step list, or repeated entry:

```json
{"schema_version": 1, "drawing": 1, "drawing_id": "5be1...", "started_at": "2026-11-21T10:02:00+00:00",
 "ended_at": "2026-11-21T10:14:31+00:00", "end": "finished",
 "canvas": {"width": 512, "height": 360},
 "save_interval_seconds": 2.0,
 "suggestions_enabled": true,
 "suggester": "ControlNetStrokeSuggester",
 "follow_detector": {"method": "aligned-strokes", "version": "1", "params": {"stroke_contrast": 12, "...": "..."}},
 "rule": {"coverage_tolerance_pixels": 16, "min_coverage": 0.35, "precision_tolerance_pixels": 16, "min_precision": 0.4,
          "min_correlation": 0.3, "max_part_occluded_share": 0.5, "min_new_stroke_pixels": 60},
 "captures": [
   {"number": 7, "trigger": "interval", "at": "...", "seconds_since_start": 12.04, "source": "webcam",
    "frame_age_seconds": 0.21, "rotation": 180, "raw": {"path": "captures/capture-0007-raw.jpg", "...": "..."},
    "canvas": {"path": "captures/capture-0007.png", "...": "..."}, "preprocessing": ["..."],
    "timings_seconds": {"capture": 0.01, "canvas": 0.04, "scoring": 0.11, "saving": 0.02},
    "follow": {"suggestion": 1, "seconds_since_shown": 4.0,
               "alignment": {"converged": true, "correlation": 0.68, "warp": [[1.0, 0.0, 1.2], [0.0, 1.0, -0.4], [0.0, 0.0, 1.0]],
                             "occluded_share": 0.04},
               "new_stroke_pixels": 812,
               "parts": [{"part": 0, "occluded_share": 0.0, "new_pixels_near": 233,
                          "coverage": {"4": 0.21, "8": 0.33, "16": 0.48, "24": 0.55},
                          "precision": {"4": 0.30, "8": 0.52, "16": 0.71, "24": 0.80},
                          "chamfer_part_to_new": 11.2, "chamfer_new_to_part": 6.4}]},
    "failure": null}],
 "suggestions": [
   {"number": 1, "id": "c0a4...", "status": "shown", "requested_at": "...", "shown_at": "...", "baseline_capture": 5,
    "request": {"canvas": {"path": "suggestions/suggestion-1/strokes.png", "...": "..."}, "prompt": "a red sports car",
                "seed": null, "controls": {}, "max_parts": 4},
    "result": {"suggester": {"adapter": "sketchloop.controlnet-strokes", "adapter_version": "1", "execution": "in_process",
                             "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe16113 + ..."},
               "effective": {"prompt": "a simple black line drawing of a red sports car, clean thin ...", "negative_prompt": "shading, ...",
                             "controls": {"prompt_template": "with_prompt", "prompt_tokens": 24, "prompt_truncated": false,
                                          "strength": 0.85, "steps": 5, "executed_steps": 4, "guidance_scale": 1.5, "...": "..."}},
               "seed": 1234, "line_drawing": {"path": "suggestions/suggestion-1/line-drawing.png", "...": "..."},
               "parts": [{"index": 0, "mask": {"path": "suggestions/suggestion-1/part-1.png", "...": "..."},
                          "box": [120, 250, 80, 60], "pixel_count": 1430, "skeleton_pixels": 260, "color": "#0072B2",
                          "label": null}]},
    "overlay": {"path": "suggestions/suggestion-1/overlay.png", "...": "..."},
    "timings_seconds": {"canvas": 0.04, "model_load": 21.3, "generation": 3.1, "extraction": 0.02, "overlay": 0.01, "saving": 0.05},
    "window": {"ended_at": "...", "end": "next_suggestion", "last_capture": 26},
    "decision": {"method": "aligned-strokes", "method_version": "1", "rule": {"...": "..."}, "result": "followed",
                 "parts": [{"part": 0, "result": "followed", "deciding_capture": 25, "reason": null}]},
    "failure": null}],
 "final_capture": 361}
```

Field rules:

- **Requested and effective stay apart:** `request.controls` is what was asked, and `result.effective` and `result.seed` are what the suggester used.
- **Open values:** `window` and `decision` are null while a window is open. `ended_at`, `end`, and `final_capture` are null until the drawing finishes. A server stopped mid-drawing therefore leaves an unfinished drawing, which `show` reports as such.
- **A failed suggestion** has `status: "failed"`, a `failure` with its stage, null `shown_at`, `result`, `overlay`, `window`, and `decision`, and a `request` that is null only when the canvas failed.
- **A failed capture** has `failure` with stage `canvas` or `saving`, null `canvas`, and null `follow`. An alignment that did not converge is not a failure. It is recorded in `follow.alignment`.
- **No-AI drawings:** `suggestions_enabled: false` and `suggester: null` mark a drawing with suggestions off, which T10 uses for that condition. T09's page always starts drawings with suggestions on.
- **Labels and colors:** `label` is null in T09. `color` is the overlay color by index from `PART_COLORS`, held in `SuggestionRecord.part_colors` and written into each part. It is a presentation choice, recorded because it is what the person saw.
- **Selections stay separate:** no field of `drawing.json` is a selection. Rounds keep explicit selections, and inferred acceptance lives only here.

`DrawingRecord` invariants, checked on construction and so on load:

- **Numbering and baselines:** capture and suggestion numbers run from 1 without gaps. Every shown suggestion's `baseline_capture` exists, has trigger `suggestion` and a canvas, and was taken before `shown_at`.
- **Windows:** windows do not overlap. Each capture's `follow.suggestion` names the suggestion whose window held it, and at most one window is open.
- **Decisions:** a decision exists exactly when a window has ended. It has one part decision per part, and each `deciding_capture` lies inside that window.
- **Finishing:** a finished drawing has no open window, and its `final_capture`, when set, has trigger `finish`.
- **Sizes:** every canvas, strokes image, mask, and overlay has the canvas size.

### Store (`sketchloop.experiments.stores`)

`ExperimentStore` gains three abstract methods, implemented by `FolderStore` and `MemoryStore`. `claim_session` is described under [Experiment names](#experiment-names-sketchloopexperiments-sketchloopsession).

```python
def claim_session(self, session_id: str) -> None: ...  # Reserve a named session's folder, or raise ExperimentNameTakenError
def locate_drawing(self, session_id: str, number: int) -> Path | None: ...
def save_drawing(self, session_id: str, record: DrawingRecord, files: Mapping[str, bytes]) -> None: ...  # files first, then drawing.json
```

They follow the same rules as `save_round`, with one-line `StoreError` messages relative to the runs folder. If `drawing.json` fails to save after its files did, `GuidedDrawing` keeps the new entry in memory, and the next save writes it. If a file fails to save, the entry is dropped and the error goes to the page.

### Loading (`sketchloop.experiments.loading`, `records`)

- `SavedSession` gains `drawings: tuple[DrawingRecord, ...] = ()`. A session saved before T09 has no `drawing-<n>` folders and loads with none, so no migration is needed.
- `load_session` lists `drawing-<n>` folders with the same gap check as rounds, reads each `drawing.json` through `convert_record`, and verifies every recorded image's checksum and that its path stays inside the drawing folder. Missing or changed files go to `problems`, and a folder without `drawing.json` is reported as interrupted.
- `rerun` and `compare_rounds` are unchanged. Drawings are inspected, not rerun.

### AppController operations (`sketchloop.app`)

The constructor gains `stroke_suggester: StrokeSuggester | None = None`, defaulting to `FakeStrokeSuggester()`, `follow_detector: FollowDetector | None = None`, defaulting to `AlignedStrokeFollowDetector()`, `canvas_size: tuple[int, int] = (512, 360)`, and `save_interval_seconds: float = 2.0`. The existing `suggester` argument stays the prompt suggester.

| Operation | What the page gets |
| --- | --- |
| `start_drawing(name: str) -> DrawingView` | Checks the name, finishes any open drawing with `DrawingEnd.NEW_SESSION`, replaces the session with a new named `SketchSession` that reuses the loaded generator, as `start_new_session` does, and starts drawing 1 in it with the configured suggester. An invalid or taken name raises before anything is saved, and the current session stays. |
| `suggest_strokes(sketch, source, prompt, *, rotation, seed=None, frame_age_seconds=None) -> SuggestionView` | Runs `GuidedDrawing.suggest` with the prompt box text as typed, which may be empty. `SuggestionView` holds the drawing and suggestion numbers, the part count, the colors, whether it is fake, the generation seconds, and a one-line status such as "Suggestion 2: 3 parts in 3.1 s". Refuses when no drawing is open or the sketch is missing, before reading anything. |
| `record_drawing_capture(sketch, source, *, rotation, frame_age_seconds=None) -> CaptureView \| None` | Returns None without saving when no drawing is open, the sketch is missing, or the last saved capture is less than `0.8 * save_interval_seconds` old, so several tabs do not multiply captures. Otherwise saves and returns the capture number and the open suggestion's latest scores, for an optional status line. |
| `preview_guided_sketch(sketch, rotation) -> bytes \| None` | The current frame's paper canvas with the open suggestion's overlay, or None when no drawing is open, so the page keeps today's preview. Corner warp only, without ECC, since it runs every 0.5 s. |
| `finish_drawing(sketch, source, *, rotation, frame_age_seconds=None) -> DrawingView` | Final capture, closes the window, and records the decision. `DrawingView` holds the number, capture and suggestion counts, and one line per suggestion result. |
| `latest_drawing() -> DrawingView \| None` | Restores the view on page reload. |
| `describe_stroke_suggester() -> str` | The suggester's sentence for the page header. |
| `start_new_session()` | Now first finishes an open drawing with `DrawingEnd.NEW_SESSION`, without a final capture. |

Rounds generated after `start_drawing` are saved in the named session too, so the whole experiment is in one folder. T10 builds on this: it can compose or check names from the experimenter's codes and the order of conditions, starts the no-AI condition with `suggester=None`, and its blinded export reads each named session's final capture and maps the name to a random code, with the key kept local as S05 requires. None of this needs a change to the fields above.

### Page (`sketchloop.ui`, `static/`)

- **Experiment controls:** an "Experiment name" text box with the hint "Use a code such as participant1_car. Never use a real name.", and a "Start drawing" button enabled when the name is not blank. Its one-line error shows under the box when the name is invalid or taken.
- **Drawing controls:** a "Suggest next strokes" button enabled while a drawing is open and a fresh frame exists, which sends the existing prompt box's text, empty or not, a "Finish drawing" button enabled while a drawing is open, and a status line naming the experiment and the drawing state. There is no goal box, and nothing on the page names a task. Labels follow the no-parentheses rule for UI labels.
- **Queues:** Suggest runs on `concurrency_id="session"` with Generate, so the two models never run at once. The capture timer runs on its own `concurrency_id="drawing-capture"`.
- **Timers:** a `gr.Timer(save_interval_seconds)` calls `record_drawing_capture` with the newest fresh frame. The existing preview timer calls `preview_guided_sketch` first and falls back to today's preview when it returns None.
- **Lifetime:** captures stop while no tab is open, because Gradio timers run in the browser.
- **Startup:** `cli/serve.py` creates the stroke suggester from the settings and warms it up in a background thread at the canvas size, like `warm_up_suggester`, printing the time it took or why suggestions are unavailable.

### Printing (`sketchloop.cli.show`)

After the rounds and before the integrity section, `show` prints each drawing:

```
Experiment: participant1_car
Drawing 1: finished after 12 min 31 s, 361 captures, canvas 512x360, final drawing captures/capture-0361.png
Suggester: sketchloop.controlnet-strokes (model IDs with revisions), follow method aligned-strokes 1
Rule: coverage_16 >= 0.35 and precision_16 >= 0.4, undetermined below correlation 0.3, half a part covered, or 60 new pixels
 #  shown    prompt             parts  seed  time   window                        result        parts
 1  +00:42   a red sports car   3      1234  3.1 s  44 s, 22 captures, next       followed      1 followed, 2 not, 3 undetermined
 2  +01:26   (empty)            failed during generation: Next-stroke suggestion failed: ...
```

- The experiment line appears only for a named session, and the header of every named session also prints it. An empty prompt prints as `(empty)`, and a truncated one is marked.
- A fake suggester gets the same warning line as a fake backend.
- An unfinished drawing prints "not finished", and decides its open window with `decide_follow` and the drawing's rule, labelled "computed by show, not recorded".
- A suggestion with no parts prints "no new strokes". Per-capture scores are not printed, since they stay in `drawing.json` for analysis.

## Failure paths

| Case | Behavior |
| --- | --- |
| Invalid experiment name | `ExperimentNameError` with the rule in one line. Nothing is created, and the current session stays. |
| Experiment name taken | `ExperimentNameTakenError` naming the existing experiment. Nothing is created. |
| Server stops during a named experiment | The folder keeps every saved capture, and its drawing loads as not finished. Resuming it is out of scope, so the experimenter starts a new name, such as `participant1_car_2`, and both stay visible. |
| Weights missing | The startup warm-up prints `Next-stroke suggestions are unavailable: Model folder models/... is missing. Run python scripts/download_models.py and retry.` A press records a failed suggestion at stage `model_load` and shows the same line. Captures continue. |
| Suggestion fails or runs out of memory | Failed suggestion at stage `generation`, with the message on the page. The previous window stays open, and the baseline capture is kept. |
| Unsupported or out-of-range control | `UnsupportedConfigurationError` listing every issue, recorded as a failed suggestion at stage `generation`, as rounds do. |
| No strokes on the canvas | `InvalidSketchError` before the model runs. The capture is kept, and no suggestion is recorded. |
| Paper corners not found | The canvas falls back to the paper's bounding box, records `corners_found: false`, and relies on ECC. With no paper found at all, `paper_found: false` is recorded, and alignment usually fails, which makes those captures unusable rather than wrong. |
| Alignment does not converge or correlates weakly | Recorded in `follow.alignment`, with no part scores when it did not converge. `decide_follow` skips the capture. |
| A hand covers the paper | Occluders are masked from alignment and new strokes, and each part records its occluded share. A part more than half covered is skipped for that capture, and the latest unoccluded capture decides. |
| Stale or missing webcam frame | `record_drawing_capture` returns None and saves nothing. Gaps show in the capture times. |
| Save fails | `StoreError` with one line. A failed `drawing.json` is retried on the next save, and a failed file drops that entry. |
| Server stops mid-drawing | `drawing.json` keeps every saved capture, and the drawing loads as not finished with its last window open. |

## Storage estimate

At a 2 s interval, a 15-minute drawing has about 450 captures. Lossless PNG frames at 640x480 are about 0.5 MB each, which is about 230 MB per drawing and about 9 GB for 40 drawings. JPEG at quality 90 for interval captures is about 60 KB, and with PNG canvases of about 100 KB this gives about 75 MB per drawing and about 3 GB for 40. The streamed webcam frames already arrive as JPEG, so re-encoding loses little. These are estimates to check in the pilot. Everything stays under the ignored `runs/` folder, as [PRIVACY.md](../PRIVACY.md) requires for participant data.

## Conflicts and deferrals

- **Part labels, deferred:** ADR 0024 records Florence-2 part names as metadata, but it also says tasks have no part lists, and grounding needs a vocabulary. Items 24 to 33 do not include labeling. `label` is reserved and null in T09, and a labeler can be added after item 33 once the owner says where its vocabulary comes from.
- **One open window:** the research note's "AppController keeps the open suggestions" is narrowed by ADR 0024's window rule to at most one open suggestion, which this design enforces.
- **A failed press:** ADR 0024 ends a window at "the next suggestion". This design reads that as the next suggestion *shown*, so a failed press does not end the previous window.
- **The suggestion-level result:** ADR 0024 and S02 ask for followed, not followed, or undetermined per suggestion, while scoring per part. The owner accepted the aggregation above, and it can be recomputed from the stored part decisions.
- **No stated goal:** S01 in PROJECT.md and ADR 0024 speak of suggestions "toward the stated goal". Following the owner, the only statement of a goal is the participant's own prompt, and the suggester runs with a neutral line-drawing prompt when it is empty. Whether neutral suggestions are useful was not tested in the research note, and the model test and pilot check it. PROJECT.md S01 can be reworded at handoff.
- **T10 wording:** the T10 row of ROADMAP.md says the experimenter enters the participant code, task, and condition. Under the owner's answer the only stored study label is the experiment name, so T10's design reconciles that row before it is built.
- **No conflict with existing records:** every change is additive, older sessions load unchanged, and a pre-T09 `sketchloop` ignores `drawing-<n>` folders.

## File plan

Each slice is one commit, with its tests, and updates [the call graph](../call-graph.html), `docs/ARCHITECTURE.md` where a boundary row changes, and traceability. B is the `backend-developer` and F the `frontend-developer`.

| Slice | Item | Who | Files, in order |
| --- | --- | --- | --- |
| 1. Suggester interface | 24 | B | `generation.py`: extract `control_issues`. `stroke_suggestion.py`: constants, error, the four records, `StrokeSuggester`. |
| 2. Fake suggester and settings | 25 | B | `stroke_suggesters/__init__.py`, `stroke_suggesters/fake.py`. `settings.py`: `StrokeSuggesterSettings`, `ControlNetSuggesterSettings`, `DrawingSettings`, and their ranges. `components.py`: `STROKE_SUGGESTERS`, `make_stroke_suggester`, and the name checks. `examples/sketchloop.toml`. |
| 3. Paper canvas | 28, 30 | B | `preprocessing.py`: `PaperCanvas`, `make_paper_canvas`, `extract_canvas_strokes`. |
| 4. Real suggester | 26 | B | `backends/diffusers.py`: `require_model_folders`. `stroke_suggesters/line_parts.py`. `stroke_suggesters/controlnet.py`. Register `"controlnet"`. |
| 5. Experiment names, drawing records, and suggest | 27, 28 | B | `experiments/records.py`: `check_experiment_name`, the two name errors, and `SessionRecord.name`. `experiments/serialization.py`: the `name` field, and split the backend and effective helpers. `experiments/drawings.py`. `experiments/stores.py`: `claim_session`, `locate_drawing`, `save_drawing`. `drawing.py`: `GuidedDrawing` with `suggest`, the overlay, `compose_preview`, and `finish` without follow. `session.py`: the `name` argument, `ensure_session_record`, `start_drawing`. `app.py`: `start_drawing`, `suggest_strokes`, `preview_guided_sketch`, `finish_drawing`, and the views. `cli/serve.py`: create and warm up the suggester. |
| 6. Name box, buttons, and overlay | 27, 28 | F | `ui.py` and `static/`: the Experiment name box with its privacy hint, the Start drawing, Suggest, and Finish buttons, the status line, and the guided preview through the existing timer. |
| 7. Interval captures | 29 | B, then F | `drawing.py`: `record_capture` with the JPEG and PNG choice. `capture.py`: `encode_frame_jpeg(frame, quality)`. `app.py`: `record_drawing_capture` with the rate limit. Then `ui.py`: the capture timer and queue. |
| 8. Alignment | 30 | B | `follow.py`: `Alignment`, `make_darkness_map`, `find_occluders`, `align_to_baseline`, and the new-stroke mask. |
| 9. Scoring | 31 | B | `follow.py`: `PartScores`, `CaptureScores`, `FollowDetector`, `AlignedStrokeFollowDetector`. `drawing.py`: score each capture in an open window. `experiments/drawings.py`: the `follow` field. |
| 10. Decisions | 32 | B | `follow.py`: `FollowRule`, `PROVISIONAL_RULE`, `PartDecision`, `FollowDecision`, `decide_follow`. `drawing.py`: close windows and record decisions. `experiments/drawings.py`: `window` and `decision` with their invariants. |
| 11. Loading and `show` | 33 | B | `experiments/records.py`: `SavedSession.drawings`. `experiments/loading.py`: drawing folders and checks. `cli/show.py`: `print_drawing_record` and the experiment name in the header. Then `docs/EXPERIMENTS.md`, `docs/STATUS.md`, and the README usage. |

Slices 5 and 6 can be built in parallel from this contract. Slice 6 tests against a controller built with the fake suggester.

## Tests for the qa-engineer

Default checks stay offline: no camera, GPU, weights, or network. Synthetic canvases are drawn with OpenCV in fixtures in `tests/conftest.py`, including a seeded "photo" renderer that adds uneven light, noise, JPEG at quality 75, a camera shift of up to 4 px per corner, and an optional hand blob, ported from the research simulation.

1. **Interface, R08:** each record refuses its invalid cases (`max_parts` of 0 or 5, canvas sides not divisible by 8, gaps in part indexes, payload checksum mismatch, mask size mismatch). An empty or spaces-only prompt is accepted and kept unchanged. `control_issues` gives the same issues as before through `request_issues`, and a test generator and test suggester share it.
2. **Fake suggester, S01, R08:** the same canvas gives the same parts and bytes. The parts lie below and right of the strokes' box and inside the canvas. A blank canvas still gives parts. `is_fake` is true. A settings file naming `fake` or `controlnet` builds the right class, and a bad name or a canvas side of 500 fails with one line naming the key.
3. **Canvas, S04:** a synthetic sheet photographed at a tilt, shifted between captures, gives canvases of exactly the configured size whose strokes land within 3 px of each other. A missing corner records `corners_found: false` and still has the size. Strokes far from the main drawing survive, unlike `extract_strokes`.
4. **Real suggester with a stub runner, S01:** a typed prompt gives the research template and an empty or spaces-only prompt the neutral one, each recorded with `prompt_template`, `prompt_tokens`, and `prompt_truncated`, which is true for a 100-word prompt with a stub tokenizer. The pipeline receives the start image black on white, the control image inverted, strength 0.85, steps 5, guidance 1.5, conditioning scale 0.8, and the canvas size. Line extraction removes lines within 13 px of existing strokes. `group_parts` on synthetic images with known shapes gives the expected count, order, and boxes, keeps a part touching the edge, and drops a long straight frame line. A missing folder raises `StrokeSuggestionError` naming the download script before importing torch.
5. **Drawing and records, S01, S04, R06:** with `MemoryStore` and the fake suggester, three captures then a suggestion then two captures then finish give consecutive numbers, a baseline with trigger `suggestion`, a final capture with trigger `finish`, an overlay of canvas size, and a `drawing.json` that round-trips through `drawing_to_dict` and `drawing_from_dict` unchanged. Parametrized invalid records fail on load: overlapping windows, a decision without a window, a `follow.suggestion` outside its window, a capture gap, and a path with `..`.
6. **Failure paths, R06:** a suggester stub that raises records a failed suggestion with stage `generation`, keeps the previous window open, and re-raises. A blank canvas raises `InvalidSketchError` and records the capture but no suggestion. A store whose `drawing.json` write fails once keeps the capture and writes it on the next save. A second `start_drawing` while one is open is refused.
7. **Alignment, S02:** a capture shifted by up to 4 px per corner converges with correlation above 0.3 and, after warping, its strokes lie within 2 px of the baseline. A hand blob is found as an occluder and is not counted as new strokes. A blank capture does not converge and returns no part scores, without raising.
8. **Scoring and decisions, S02:** in the ported simulation with a fixed seed, a closely redrawn part is followed, strokes drawn elsewhere are not followed, no new strokes is undetermined, a part under a hand in the last capture is decided by the previous unoccluded capture, and no usable capture is undetermined. `decide_follow` on stored scores with a stricter rule changes the result without touching any image, which shows the decision can be recomputed.
9. **Windows, S02:** suggestion 2 ends suggestion 1's window at suggestion 2's baseline capture. Finish ends the last one with `finished`. A new session ends it with `new_session`. A failed suggestion 2 leaves suggestion 1 open.
10. **Controller and page, S01, R07:** `record_drawing_capture` returns None with no drawing open and when called twice within the interval. `suggest_strokes` accepts an empty prompt and refuses when no drawing is open. `start_drawing` creates `runs/<name>/` with `session.json` holding the name, and a round generated afterwards lands in the same folder. `preview_guided_sketch` returns None without a drawing and an image of canvas size with one. Page handler tests, without a browser: Start drawing is disabled with a blank name, Suggest and Finish are disabled without a drawing, and the page has no goal component.
11. **Experiment names, S05, R06 (parametrized):** `participant1_car` and `P-03` pass. An empty name, `../x`, `a/b`, `a\\b`, `C:x`, `.hidden`, `has space`, `nul`, `COM1`, and 65 characters each raise `ExperimentNameError` with one line and create nothing. A name already used, or the same name in other letter case, raises `ExperimentNameTakenError`, and the existing folder is untouched. A `session.json` without `name` loads it as None, and a name that differs from `session_id` fails on load.
12. **Loading and `show`, S02, R06, R09:** a session with a round and a drawing loads both. A session saved before T09, with no drawing folders, loads with `drawings == ()`. A changed mask file is a problem, not an error. `show` prints the suggestion table, the fake warning, "not finished" with "computed by show, not recorded" for an open window, and a failed suggestion's stage.
13. **Model test, marked `model`, S01, R09:** on the M2 Pro, load the real suggester, warm up at 512x360, and suggest for a synthetic rectangle once with the prompt "a car, side view" and once with an empty prompt. Expect at least one part, masks of canvas size, and a recorded generation time under 10 s after warm-up. Record the load, warm-up, and generation times, and the process's peak memory with and without the generator loaded.

Checks that need a person, before T09 is called done:

- The overlay lines up with the live webcam sketch at the default rotation.
- A real hand over the paper is masked.
- Captures from a 10-minute drawing take the expected disk space.
- Several suggestions on the owner's own sketches look plausible, with a typed prompt and with an empty one.

Hand-labeling pilot cases to tune the rule belongs to the T10 pilot.

## Out of scope

- Part labels.
- Warping the overlay back into the raw camera image.
- Several alternatives per press.
- Sharing weights with the generator.
- Suggestion replay or rerun.
- Per-capture scores in `show`, and exports.
- A settings key for the rule.
- Change-triggered captures.
- Further study features, such as composing names from codes, conditions, and the blinded export, which are T10.
- Resuming a named experiment after the server stops.
- Linking rounds to the drawing they happened in. `round.json` may gain an optional `drawing` number later without other changes.
- Streaming or automatic suggestions, which [ADR 0021](../decisions/0021-guided-sketching-and-user-study.md) rejected.

## Open questions for the owner

The owner answered the design's earlier questions, listed under [Decisions this design applies](#decisions-this-design-applies). Two remain for T10:

1. Should the "Suggest a prompt" button stay available during experiments? It names the sketched object with Florence-2, so a participant who uses it would send a system-derived prompt, and the records cannot yet tell typed from suggested prompts (U1).
2. How should T10 reconcile its roadmap row, which has the experimenter enter the task, with the experiment name being the only stored study label?
