# T10 design: study mode, the condition record, prompt suggestion records, and the blinded export

Status: accepted by the owner for the T10 build, with [ADR 0026](../decisions/0026-study-mode-condition-and-blinded-export.md) (accepted) and the answers listed below. Serves S03, S04, and S05 in [PROJECT.md](../PROJECT.md), with the record changes under R06, the page under R07, and the replaceable prompt suggester under R08. It also finishes U1 in [ROADMAP.md](../ROADMAP.md), which [ADR 0019](../decisions/0019-offline-prompt-suggestions.md) left without records. Runnable increment: `sketchloop ui --study` runs one experiment at a time on a participant page, and `sketchloop study export` writes the final drawings under random codes for the experts. Nothing below exists in `src/` yet. Every class, function, field, and file named here as new is proposed.

Impact: Medium. It adds no new component boundary and no dependency. It adds optional fields to `drawing.json` and `round.json` within schema version 1, one records module for prompt suggestions, one export module, a `study` subcommand, `[study]` and `[prompt_suggester]` settings tables, an `identity` property on the `PromptSuggester` base class, and a second page layout. The affected boundaries are `suggestion` (the prompt suggester contract), `components`, orchestration (`drawing`, `session`, `app`), `experiments` (`prompts`, `drawings`, `records`, `serialization`, `loading`, a new `study_export`), `settings`, `cli` (`arguments`, `serve`, `show`, a new `study`), and `ui` with `static/`.

## Decisions this design applies

The owner's decisions for T10, final:

1. **Prompt suggestions stay and are recorded:** "Suggest a prompt" stays available in the AI condition. Every suggested prompt is recorded with the suggester's identity, the drawing or round it belongs to, its text, and when it was shown, and each stroke suggestion or generation records whether its prompt was typed, suggested, or a suggestion the participant edited. This finishes U1. In the no-AI condition "Suggest a prompt" is hidden too, since it is AI help, so the two conditions differ only in AI help.
2. **One page:** `sketchloop ui --study`, or a settings key, serves the study page. Participants see the live sketch view, the prompt box, Suggest next strokes and Suggest a prompt in the AI condition only, and the drawing status. Generate, the candidate gallery, selection, the round status, sliders, seed, "Start a new session", and the experiment name are hidden. Experimenter controls sit in a collapsed panel at the bottom. The header must not reveal the condition.
3. **No-AI condition:** the same page and capture, with the Suggest next strokes button removed, not disabled. The experimenter chooses the condition when starting each experiment, and the condition is recorded on the drawing, where [ADR 0025](../decisions/0025-drawings-as-the-unit-of-guided-sketching.md) already reserves `suggestions_enabled` and a study object.
4. **Blinded export:** one final-drawing PNG per experiment named by a random code, a key file mapping codes to experiment names and conditions kept in ignored local storage, missing final drawings reported, a configurable output folder, and a whole runs folder as input.
5. **Configured, not fixed:** any tasks, any order of tasks and conditions, and free prompts. The system never knows the task, and the experiment name is the only study label.

The owner's answers to this design's questions:

1. **The experts' image:** the recorded 512x360 paper canvas is fine for the pilot. Warping the final raw frame again at a larger size stays a noted option.
2. **Final prompt:** each drawing records `final_prompt` and `final_prompt_origin` at Finish.
3. **Prompt suggester setting:** a `[prompt_suggester]` settings table chooses the prompt suggester independently of the image backend, and [ADR 0019](../decisions/0019-offline-prompt-suggestions.md) is amended to say so.
4. **Placeholder:** "Optional. Describe your drawing in a few words" is approved.
5. **Wording:** the proposed S03 wording in PROJECT.md and the T10 row in ROADMAP.md are applied.

This design also answers the two open questions the [T09 design](t09-guided-sketching.md#open-questions-for-the-owner) left: "Suggest a prompt" stays and is recorded, and the experimenter enters a name and a condition, never a task.

## Current state

- **Drawings:** `AppController.start_drawing(name)` claims `runs/<name>/` through a new named `SketchSession` and opens drawing 1 with the configured stroke suggester. `SketchSession.start_drawing(None, ...)` already gives a drawing with `suggestions_enabled: false` and `suggester: null`, which no page uses yet. `GuidedDrawing.finish` keeps the frame at finish as `final_capture`, a PNG raw frame plus its paper canvas at the drawing's canvas size, 512x360 by default.
- **Prompt suggestions:** `AppController.suggest_prompt` returns one string. It runs the `PromptSuggester` on the preprocessed sketch when the drawing changed and otherwise returns the next entry of the previous list, and nothing is recorded. `PromptSuggester` has `suggest` and `warm_up`, and no identity. `cli/serve.py` picks `FlorenceSuggester` next to a real backend and `TemplateSuggester` next to the fake, as amended in ADR 0019.
- **Page:** one layout. The header prints the backend and stroke suggester descriptions. The Experiment accordion, open by default, holds the name, Start drawing, Finish drawing, and the capture notice. The prompt placeholder names an oak lounge chair. The status line names the experiment.
- **Records:** `DrawingRecord` has no study field, no prompt suggestions, and no prompt origin. `RoundRecord` has neither. `decode_value` treats any JSON object as an unavailable marker, so a new object-valued field needs its own decoder.
- **Generator:** creating the diffusers backend imports torch but loads no model, so a study page that never generates never loads SD 1.5 for generation.

## Options

### Where prompt suggestions are recorded

| Option | Tradeoffs |
| --- | --- |
| **A, recommended: in the unit they belong to.** In `drawing.json` while a drawing is open, and in the next `round.json` otherwise | Matches the owner's "the drawing or round it belongs to". A study drawing stays one self-contained record, as ADR 0025 decided, and the link from a stroke suggestion to its prompt suggestion is an invariant `DrawingRecord` checks on load. Outside a drawing, suggestions wait in memory for the next round, so those shown before the page closes without a round are lost. That happens only on the generation page, never in the study. |
| B: a session-level `prompt-suggestions.json`, saved on every press | Nothing is ever lost, and rounds and drawings only reference IDs. It is a second record file with its own loader, the link becomes a cross-file check, and a session with no presses cannot be told from an older one without another marker. |
| C: drawings in `drawing.json`, everything else in a session file | Two places for the same event, the worst of both. |

### How the system knows whether a prompt was suggested

| Option | Tradeoffs |
| --- | --- |
| **The page keeps the ID of the suggestion it wrote into its prompt box, and sends it with each use, recommended** | Each tab knows what its own box received, a reload empties the box and the ID together, and the controller checks that the ID belongs to the drawing, as a selection must name the shown iteration. Comparing the used text with the suggestion's text then gives typed, suggested, or edited. |
| The controller remembers the last suggestion it returned | No page change, but a reload or a second tab makes a typed prompt look edited. |
| The page reports every keystroke | Exact, but it records typing behavior nobody asked for, adds JavaScript, and adds noise to every capture. |

### What the experts receive

| Option | Tradeoffs |
| --- | --- |
| **The final capture's recorded paper canvas, re-encoded without metadata, chosen** | Exactly what the system recorded and scored, already straightened and limited to the sheet, the same pipeline in both conditions, and no new processing. At 512x360 it is small, and an area hidden by a hand at finish is filled with paper. |
| The final raw PNG frame warped again at a larger size, such as 1024x720, with `make_paper_canvas`, a noted option for later | More detail when the webcam delivers more than about 512 px across the sheet, which is unverified. It is a new derived image, and its steps must go in the key. It can be added later as an option without changing records. |
| The raw camera frame | Shows hands, desk, and room, is not straightened, and may reveal more than the drawing. Rejected. |

### How the export is laid out and keyed

| Option | Tradeoffs |
| --- | --- |
| **A new folder per export holding `drawings/` to hand over, and `key.csv` and `export.json` beside it that never leave the machine, recommended** | The handover is one folder, the key opens in a spreadsheet to join expert scores by code, `export.json` records how the export was made, and a second export never overwrites the first. |
| The key in a separate configurable folder | Safer if someone zips the whole export folder, but two paths to configure and keep together. |
| One JSON key only | Versioned like the other records, but joining expert scores needs a script. |

### Study configuration

| Option | Tradeoffs |
| --- | --- |
| **Free per experiment, recommended:** the experimenter types the name and picks the condition at each start | Serves any tasks and any order with no file to maintain, and the system never sees a task. The experimenter can mistype a name or pick the wrong condition, which the page reduces by clearing both after each start. |
| A plan file of names and conditions in order, which the panel offers one at a time | Fewer typing errors across 40 sessions, but a new file format whose names may encode tasks, and order changes mid-study need edits. Can be added later without record changes. |

## Contracts

### Prompt suggester identity (`sketchloop.suggestion`)

`PromptSuggester` gains one abstract property, so records can name what suggested:

```python
class PromptSuggester(ABC):
    @property
    @abstractmethod
    def identity(self) -> BackendIdentity: ...  # Adapter, version, execution kind, and model ID with revision or unavailable
```

- `TemplateSuggester`: `BackendIdentity(adapter="sketchloop.prompt-templates", adapter_version="1", execution=ExecutionKind.IN_PROCESS, model_id=Unavailable("fixed style templates, no model"))`. It is a real, always-available suggester, not a test fake, so it is not labeled fake.
- `FlorenceSuggester`: `adapter="sketchloop.florence-prompts"`, `adapter_version="1"`, `execution=ExecutionKind.IN_PROCESS`, and `model_id` joining `FLORENCE_MODEL.repo_id` with the local folder's revision through `local_revision`, as the stroke suggester names its models.
- `adapter_version` is bumped whenever `STYLE_TEMPLATES`, `FRAMING_PREFIXES`, or the caption settings change, since the same sketch would then give other text.
- `suggest` keeps returning `tuple[str, ...]`. The sketch the suggester saw is not stored, and a drawing records the latest capture instead.

### Prompt suggestion records (`sketchloop.experiments.prompts`, new)

```python
ORIGIN_NOT_RECORDED: Final = "not recorded before prompt origins were"


class PromptOriginKind(StrEnum):
    TYPED = "typed"  # No suggestion was in the box, or the prompt was blank
    SUGGESTED = "suggested"  # The prompt equals the suggestion written into the box
    EDITED = "edited"  # A suggestion was written into the box and the prompt differs from it


class PromptSuggestionStatus(StrEnum):
    SHOWN = "shown"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptOrigin:
    kind: PromptOriginKind  # typed, suggested, or edited
    prompt_suggestion: str | None  # ID of the suggestion the page wrote into its prompt box, None when typed


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSuggestionFailure:
    error: str  # Error class name
    message: str  # First line of the error message, with paths redacted


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSuggestionRecord:
    number: int  # Position in the drawing's or round's list, from 1
    id: str  # Unique ID the page sends back with the prompt this suggestion was written into
    status: PromptSuggestionStatus  # shown, or failed with a failure record
    requested_at: datetime  # Timezone-aware time of the press
    shown_at: datetime | None  # Timezone-aware time the text reached the page, None when failed
    suggester: BackendIdentity  # The prompt suggester's identity
    text: str | None  # Suggested prompt written into the box, None when failed
    list_position: int | None  # 0-based position in the suggester's list, such as the style template, None when failed
    ran_suggester: bool  # True when this press ran the suggester, false when it took the next entry for an unchanged sketch
    suggester_seconds: float | None  # Seconds the suggester ran, None when it did not run
    after_capture: int | None  # Latest capture of the drawing when shown, None outside a drawing or before its first capture
    failure: PromptSuggestionFailure | None  # Why it failed, None when shown


def classify_prompt_origin(prompt: str, suggestion: PromptSuggestionRecord | None) -> PromptOrigin: ...
def prompt_origin_to_dict(origin: PromptOrigin | Unavailable) -> dict[str, object]: ...
def prompt_origin_from_dict(data: Mapping[str, object] | None) -> PromptOrigin | Unavailable: ...  # Missing gives ORIGIN_NOT_RECORDED
def prompt_suggestion_to_dict(record: PromptSuggestionRecord) -> dict[str, object]: ...
def prompt_suggestion_from_dict(data: Mapping[str, object]) -> PromptSuggestionRecord: ...
```

Invariants, checked in `__post_init__` with `require`:

- A `PromptOrigin` names a suggestion exactly when its kind is not `typed`.
- A shown record has `shown_at` at or after `requested_at`, non-empty `text`, a `list_position` of at least 0, and no failure. A failed one has a failure and null `shown_at`, `text`, and `list_position`.
- `suggester_seconds` is a finite number of at least 0 exactly when `ran_suggester` is true.
- `classify_prompt_origin` is pure. With no suggestion, or a prompt that is blank after stripping spaces, it gives `typed` with no ID. A prompt equal to the suggestion's text, both stripped of surrounding spaces, gives `suggested`. Anything else gives `edited`. Clearing the box and typing new words therefore counts as edited, and both texts are stored, so an analysis can measure how much was changed.
- `prompt_origin_from_dict` checks for the `unavailable` key itself, because `decode_value` would misread any object.

### Study condition (`sketchloop.experiments.drawings`)

```python
class StudyCondition(StrEnum):
    AI = "ai"  # Next-stroke and prompt suggestions are offered
    NO_AI = "no_ai"  # The same page and capture without any AI help


@dataclass(frozen=True, slots=True, kw_only=True)
class DrawingStudy:
    condition: StudyCondition  # Condition the experimenter chose when starting the drawing
```

`DrawingStudy` holds one field today and is the study object ADR 0025 reserved. No task, participant, or order field is added, since the experiment name is the only study label.

### Drawing record additions (`sketchloop.experiments.drawings`)

`DrawingRecord` gains, all with defaults so older drawings load:

```python
study: DrawingStudy | None = None  # Condition of a study drawing, None outside study mode and in older drawings
prompt_suggestions: tuple[PromptSuggestionRecord, ...] | None = None  # Prompt suggestions shown during the drawing, None in older drawings
final_prompt: str | Unavailable | None = Unavailable(FINAL_PROMPT_NOT_RECORDED)  # Prompt box text at finish, None while open or ended without the page
final_prompt_origin: PromptOrigin | Unavailable | None = Unavailable(ORIGIN_NOT_RECORDED)  # How the final prompt came about, None when final_prompt is None
```

`SuggestionRecord` gains `prompt_origin: PromptOrigin | Unavailable = Unavailable(ORIGIN_NOT_RECORDED)  # How the prompt sent with the press came about`.

`suggestions_enabled` keeps its name and now means that the drawing offers AI help, both next-stroke and prompt suggestions. A T09 drawing with `true` offered both on the page, so its meaning does not change for saved data.

New invariants:

- **Condition:** a study drawing with condition `ai` has suggestions on, and one with `no_ai` has them off. A non-study drawing is unconstrained, as in T09.
- **No AI help without suggestions:** a drawing with suggestions off has no prompt suggestions, as it already has no stroke suggestions.
- **Prompt suggestions:** numbered from 1 without gaps, IDs unique within the drawing, each `after_capture` at most the drawing's capture count, and `requested_at` times not decreasing.
- **Origins are checkable:** a shown stroke suggestion's `prompt_origin`, when it names a prompt suggestion, names a shown one of this drawing whose `shown_at` precedes the stroke suggestion's `requested_at`, and `classify_prompt_origin(request.prompt, that suggestion).kind` equals the recorded kind. The same holds for `final_prompt_origin` against `final_prompt` and `ended_at`.
- **Final prompt:** `final_prompt` is None while the drawing is open, and `final_prompt_origin` is None exactly when `final_prompt` is None.

`drawing.json` adds four keys and one per suggestion. The example abbreviates as the T09 design does:

```json
{"schema_version": 1, "drawing": 1, "...": "...",
 "suggestions_enabled": true, "suggester": "ControlNetStrokeSuggester",
 "study": {"condition": "ai"},
 "captures": ["..."],
 "suggestions": [
   {"number": 1, "...": "...", "request": {"prompt": "a sports car, product design render, studio lighting, white background", "...": "..."},
    "prompt_origin": {"kind": "suggested", "prompt_suggestion": "9f1c..."}}],
 "prompt_suggestions": [
   {"number": 1, "id": "9f1c...", "status": "shown", "requested_at": "...", "shown_at": "...",
    "suggester": {"adapter": "sketchloop.florence-prompts", "adapter_version": "1", "execution": "in_process",
                  "model_id": "florence-community/Florence-2-base@00921df66db7"},
    "text": "a sports car, product design render, studio lighting, white background", "list_position": 0,
    "ran_suggester": true, "suggester_seconds": 0.31, "after_capture": 41, "failure": null}],
 "final_capture": 361,
 "final_prompt": "a sports car, low and wide",
 "final_prompt_origin": {"kind": "edited", "prompt_suggestion": "9f1c..."}}
```

A no-AI drawing writes `"suggestions_enabled": false`, `"suggester": null`, `"study": {"condition": "no_ai"}`, `"prompt_suggestions": []`, and a `final_prompt` with origin `typed`. A drawing started outside study mode writes `"study": null` and `"prompt_suggestions": []`. Loading a drawing without these keys gives `study` None, `prompt_suggestions` None, and the two final-prompt fields and every suggestion's `prompt_origin` unavailable. A new drawing always writes `prompt_suggestions`, so an empty list means none were shown and a missing key means none were recorded.

### Round record additions (`sketchloop.experiments.records`, `serialization`)

`RoundRecord` gains:

```python
prompt_suggestions: tuple[PromptSuggestionRecord, ...] | None = None  # Suggestions shown outside any drawing since the previous round, None in older rounds
prompt_origin: PromptOrigin | Unavailable = Unavailable(ORIGIN_NOT_RECORDED)  # How the round's prompt came about
```

`round.json` writes them as top-level `prompt_suggestions` and `prompt_origin` after `evaluation`, so a round that failed before its request still records them. A round's origin may name a suggestion in an earlier round or in a drawing of the same session, so the round checks only an ID found in its own list. `load_session` resolves every recorded origin ID across the session's rounds and drawings and adds a problem line, not an error, when one is not found.

### Orchestration (`sketchloop.drawing`, `sketchloop.session`)

```python
# GuidedDrawing
def record_prompt_suggestion(self, record: PromptSuggestionRecord) -> PromptSuggestionRecord: ...  # Numbers it, sets after_capture, saves
def find_prompt_suggestion(self, suggestion_id: str) -> PromptSuggestionRecord | None: ...
def suggest(self, captured, *, source, rotation, prompt, seed=None, controls=None, frame_age_seconds=None,
            prompt_suggestion_id: str | None = None) -> SuggestionRecord: ...
def finish(self, captured, *, source=None, rotation=0, frame_age_seconds=None, end=DrawingEnd.FINISHED,
           prompt: str | None = None, prompt_suggestion_id: str | None = None) -> DrawingRecord: ...

# SketchSession
def start_drawing(self, suggester, *, canvas_width, canvas_height, save_interval_seconds, follow_detector=None,
                  study: DrawingStudy | None = None) -> GuidedDrawing: ...
def record_prompt_suggestion(self, record: PromptSuggestionRecord) -> PromptSuggestionRecord: ...
def run_round(self, ..., prompt_suggestion_id: str | None = None) -> RoundOutcome: ...
```

- **Recording a prompt suggestion:** `SketchSession.record_prompt_suggestion` hands it to the open drawing, which refuses it with `SketchLoopError("Suggestions are off in this drawing.")` when suggestions are off, and otherwise appends it under its lock and saves `drawing.json`. A failed save removes the entry again and re-raises, so the record never says shown for text the page did not show. Without an open drawing, the session keeps it in memory and writes it into the next round, then forgets it.
- **Resolving an origin:** `suggest` and `finish` look the ID up in their own drawing only, and `run_round` in the session's pending list, rounds, and drawings. An ID that is not found gives `Unavailable("the page named a prompt suggestion outside this drawing")` or `"... outside this session"`, and the press goes ahead, so a stale tab never blocks a participant. Otherwise the origin is `classify_prompt_origin(prompt, suggestion)`.
- **Starting a study drawing:** `start_drawing(None, ..., study=DrawingStudy(condition=StudyCondition.NO_AI))` gives suggestions off, and `StudyCondition.AI` requires a suggester. Mismatches raise `SketchLoopError` before anything is saved.
- **Finishing:** `finish` records `final_prompt` as given, an empty string included, and its origin. Ending with `DrawingEnd.NEW_SESSION` records None for both.

### AppController operations (`sketchloop.app`)

| Operation | Change |
| --- | --- |
| `start_drawing(name: str, condition: StudyCondition \| None = None) -> DrawingView` | With a condition, starts a study drawing with that condition, passing no stroke suggester for `no_ai`. Without one, unchanged. It also forgets the prompt suggester's cached list, so the first press in a new experiment runs the model. |
| `suggest_prompt(sketch, rotation, use_raw) -> PromptSuggestionView` | Returns a view instead of a string. Runs the suggester outside any lock, times it, then records a shown or failed `PromptSuggestionRecord` through the session. A failure is recorded and re-raised as one line. Refuses before running anything when the open drawing has suggestions off. |
| `suggest_strokes(..., prompt_suggestion_id: str \| None = None)` | Passes the ID to `GuidedDrawing.suggest`. |
| `run_round(..., prompt_suggestion_id: str \| None = None)` | Passes the ID to `SketchSession.run_round`. |
| `finish_drawing(..., prompt: str \| None = None, prompt_suggestion_id: str \| None = None)` | Passes both to `GuidedDrawing.finish`. |
| `describe_prompt_suggester() -> str` | One sentence for the experimenter panel, such as "Prompt suggestions: Florence-2-base on mps." or "Prompt suggestions: fixed style templates, no model." |

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSuggestionView:
    suggestion_id: str  # ID the page keeps with its prompt box and sends back with each use
    text: str  # Suggested prompt to write into the box
```

`DrawingView` gains `condition: StudyCondition | None  # Study condition, None outside study mode`, `suggestions_enabled: bool  # Whether the page offers AI help in this drawing`, and `started_at: datetime  # Local start time for the experimenter's status line`.

The controller holds no study flag. The page decides what to show, and the controller enforces only the condition, which is what the records must guarantee.

### Settings and commands (`sketchloop.settings`, `sketchloop.cli`)

```toml
[study]
enabled = false                     # serve the study page with sketchloop ui, also --study and --no-study
export_dir = "data/study-exports"   # where sketchloop study export creates a new folder, relative to this file

[prompt_suggester]
name = "templates"                  # templates, which needs no model, or florence, the local Florence-2-base model
```

- `PromptSuggesterSettings(name: str = "templates")` joins `Settings` as `prompt_suggester`. `components` gains `PROMPT_SUGGESTERS: Final[dict[str, PromptSuggesterFactory]]` with `"templates"` and `"florence"`, built with `settings.models.dir`, and `make_prompt_suggester(settings)`, and `check_component_names` checks the name. `cli/serve.py` builds the prompt suggester through it instead of from the backend, so the study can run the fake or no image backend with Florence-2 prompts, or a real backend with templates. The default is `templates`, like `stroke_suggester.name = "fake"`, so default checks need no weights, and the study's settings file names `florence` explicitly. `examples/sketchloop.toml` shows `florence` next to its real backend. A `florence` setting whose model folder is missing keeps today's behavior: the background warm-up prints why prompt suggestions are unavailable, and each press records a failed suggestion.

- `StudySettings(enabled: bool = False, export_dir: Path = Path("data/study-exports"))` joins `Settings` as `study`. `data/` is already ignored by Git.
- `sketchloop ui` gains `--study/--no-study` (`BooleanOptionalAction`, settings `study.enabled`). A bare `sketchloop` follows the settings file, so a study machine can set `enabled = true` once.
- In study mode, `serve` prints one line saying the page is in study mode, and one yellow line for each stand-in: the fake stroke suggester and the template prompt suggester. The page's experimenter panel shows the same warnings.
- `sketchloop study export [--runs-dir DIR] [--output DIR] [--match PATTERN] [--config PATH]` and `sketchloop study summary [--runs-dir DIR] [--match PATTERN] [--config PATH]` are a new `study` subcommand with two actions in `cli/study.py`. `--match` is a glob on experiment names, compared ignoring case, `*` by default, so `--match "p3_*"` gives one participant's experiments when names start with the participant code.

### Blinded export (`sketchloop.experiments.study_export`, new, standard library and OpenCV only)

```python
EXPORT_CODE_ALPHABET: Final = "23456789ABCDEFGHJKMNPQRSTVWXYZ"  # No 0, 1, I, L, O, or U, so codes read aloud without confusion
EXPORT_CODE_LENGTH: Final = 6
DRAWINGS_FOLDER: Final = "drawings"
KEY_FILE: Final = "key.csv"
EXPORT_FILE: Final = "export.json"
KEY_COLUMNS: Final = ("code", "experiment", "drawing", "condition", "status", "reason", "final_capture", "source", "source_sha256",
                      "exported_sha256")


class ExportError(SketchLoopError, RuntimeError): ...  # Nothing to export, or the output folder cannot be used or written


class ExportStatus(StrEnum):
    EXPORTED = "exported"
    MISSING = "missing"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExportEntry:
    code: str | None  # Random code naming the exported file, None when missing
    experiment: str  # Experiment name, which is the session folder's name
    drawing: int  # Drawing number in the session
    condition: StudyCondition  # Condition recorded on the drawing
    status: ExportStatus  # exported or missing
    reason: str | None  # Why it is missing, None when exported
    final_capture: int | None  # Number of the final capture, None when there is none
    source: str | None  # Final canvas path relative to the runs folder, None when there is none
    source_sha256: str | None  # Recorded checksum of the final canvas, None when there is none
    exported_sha256: str | None  # Checksum of the exported PNG, None when missing


@dataclass(frozen=True, slots=True, kw_only=True)
class StudyExport:
    folder: Path  # The export folder that was created
    entries: tuple[ExportEntry, ...]  # One per study drawing found, sorted by experiment and drawing
    skipped: tuple[str, ...]  # One line per runs-folder entry that could not be loaded


def make_export_code() -> str: ...  # secrets.choice over the alphabet, never derived from a name, time, or order
def find_study_drawings(runs_dir: Path, match: str = "*") -> tuple[list[tuple[SavedSession, DrawingRecord]], list[str]]: ...
def export_study_drawings(runs_dir: Path, output: Path, *, match: str = "*", make_code: Callable[[], str] = make_export_code) -> StudyExport: ...  # noqa: E501
```

Behavior:

1. **Find:** every entry of the runs folder with a `session.json` is loaded with `load_session`. A load error adds a `skipped` line with the folder name and the one-line error and goes on. A drawing is a study drawing when `study` is set, and only named sessions whose name matches `--match` count. Time-named sessions, T09 drawings without `study`, and folders without `session.json` are passed over silently, and their count is printed.
2. **Decide each drawing:** exported when it ended `finished`, has a `final_capture` whose canvas exists, and the canvas file matches its recorded checksum. Otherwise missing, with one of these reasons: "not finished: the server stopped during it", "ended by a new session", "finished without a final frame", "final frame has no canvas: <failure message>", or "final canvas is missing or changed".
3. **Code:** each exported drawing gets a fresh code from `make_code`, drawn again on a collision within the export. Codes are new on every export, since nothing links two exports.
4. **Write:** into a sibling temporary folder `<output>.partial-<random>`, the canvases are decoded and re-encoded with `cv2.imencode(".png")` into `drawings/<code>.png` in code order, so neither file names, file order, nor metadata carry the experiment or condition. Then `key.csv` in the column order above, sorted by experiment and drawing and including missing rows, and `export.json`. Finally the temporary folder is renamed to `output`. A failure removes the temporary folder and raises `ExportError` with one line, so a partial export is never left looking complete.
5. **Refuse before writing:** a runs folder that does not exist, no study drawing found, or no drawing exportable, with the missing ones listed. An output folder that exists and is not empty, or lies inside the runs folder, where it would look like a session and block that name.

`export.json`, local only:

```json
{"schema_version": 1, "created_at": "2026-12-10T10:15:00+00:00", "sketchloop_version": "0.1.0",
 "image": "final paper canvas, grayscale PNG re-encoded without metadata", "match": "*",
 "exported": 38, "missing": 2, "skipped": ["old-session: session.json is invalid: ..."]}
```

`cli/study.py` adds the default output `<study.export_dir>/export-<UTC yyyymmdd-HHMMSS>`, and refuses an output folder that `git check-ignore` reports as not ignored inside a Git working tree, with "Export folder <path> is not ignored by Git, so participant data could be committed. Choose a folder under data/ or another ignored folder." When Git is missing or the folder is outside a working tree, the check is skipped with one printed line. The command prints the counts, each missing drawing's experiment and reason, the skipped lines, and "Hand over only the drawings folder. key.csv and export.json link codes to experiments and stay on this machine."

### Study summary (`cli/study.py`)

`sketchloop study summary` prints one row per study drawing: experiment, condition, state, length, captures, stroke suggestions shown and failed, recorded results as followed, not followed, and undetermined, prompt suggestions shown, how the prompts sent with stroke suggestions came about as typed, suggested, and edited counts, and whether a final drawing exists. An open window is decided as `show` does and marked as computed. It reads only, writes nothing, and prints names, so it is run on the experimenter's machine. A CSV option is deferred until the pilot shows it is needed.

### Printing (`sketchloop.cli.show`)

For each drawing, `show` adds a "Condition: with AI" or "without AI" line for a study drawing, a "prompt" column in the suggestion table showing `typed`, `suggested 2`, or `edited 2` with the prompt suggestion's number, one line per prompt suggestion with its offset, text, list position, whether the model ran, and its time or failure, and the final prompt with its origin. Older drawings print "not recorded" for each missing value. Rounds add their prompt origin and any prompt suggestions in the same form.

## Study page (`sketchloop.ui`, `static/`)

### Participant area

```
+----------------------------------------------------------------------------------------+
| SketchLoop                                                                              |
+------------------------------------------------------+---------------------------------+
|                                                      | Prompt                          |
|   Your sketch                                        | [ Optional. Describe your       |
|   paper canvas with the suggestion overlay,          |   drawing in a few words      ] |
|   large, fixed height                                |                                 |
|                                                      | [ Suggest a prompt ]   AI only  |
|                                                      | [ Suggest next strokes ] AI only|
|                                                      |                                 |
|                                                      | Drawing status line             |
+------------------------------------------------------+---------------------------------+
| > Experimenter                                          collapsed accordion            |
+----------------------------------------------------------------------------------------+
```

- **Header:** the title only. The backend, stroke suggester, and prompt suggester descriptions and the shared-session note move to the experimenter panel, so nothing above the drawing differs between conditions or names a model.
- **Sketch view:** the paper canvas with the open suggestion's overlay, from `preview_guided_sketch`, at a fixed height while a drawing is open. Before a start and after a finish it is empty, with no fallback to the generation preview. Its label is "Your sketch" in both conditions, without "with suggested strokes", so the label is the same in both.
- **Prompt:** label "Prompt", placeholder "Optional. Describe your drawing in a few words", with no example object, since the current oak chair example would prime every participant. Cleared at each start.
- **Buttons:** Suggest a prompt and Suggest next strokes exist only while an AI drawing is open, made visible or invisible by the start, finish, and page-load outputs, so they are removed, not disabled, in the no-AI condition and between experiments. They sit in a right column of fixed width below the prompt, so removing them moves nothing else. Suggest next strokes keeps today's waiting line and re-enabling.
- **Status line, the same words in both conditions apart from suggestion results:** before a start, "Waiting for the experimenter to start." While open, "Drawing. Tell the experimenter when you are done." After a suggestion, its line, such as "Suggestion 2: 3 colored parts", with the fake-suggester warning when it is fake. After finish, "Finished. Thank you." The experiment name and condition never appear here.
- **Hidden in study mode and not built:** Generate, candidates, seed, sliders, the gallery, the choice, the round status, Start a new session, the session history, the source choice, rotation, the raw option, and the generation preview. Components that are not built cannot fire an event.
- **Readability:** a `study` CSS class on the page raises the status text size and keeps the sketch view tall, through theme variables so light and dark modes both work.

### Experimenter panel

A `gr.Accordion("Experimenter", open=False)` at the bottom holding:

- **Start row:** Experiment name with today's hint, a Condition radio with "With AI" and "Without AI" and no default value, Start drawing, enabled only with a name and a condition and no open drawing, and Finish drawing, enabled while a drawing is open. Finish sends the prompt box text and the kept prompt suggestion ID.
- **After a start:** the name and condition are cleared, so the next experiment needs a fresh choice and a condition cannot carry over, and the accordion closes itself.
- **Experimenter status:** "participant1_car, without AI, started 10:02, drawing open", from `DrawingView`. It shows the start time, not a running clock.
- **Camera:** the source choice, rotation, and the live webcam view, moved here because only the experimenter adjusts them.
- **System:** the backend, stroke suggester, and prompt suggester sentences, and a warning line for each stand-in.
- **Saving notice:** today's capture notice. While it is shown, the accordion's label changes to "Experimenter: saving failed", so a failure is visible while the panel is closed without alarming the participant with a message in their area.

`SketchPage` gains `study: bool`, and `launch_app` passes it from the settings. The study layout is a separate build method that reuses today's handlers and cards where they fit. The prompt box gets a per-tab `gr.State` holding the ID from the latest `PromptSuggestionView`, set by Suggest a prompt, cleared by Start and by Start a new session, and sent with Suggest next strokes, Generate, and Finish on both pages.

### Why this layout

- **The drawing stays the task:** one large view mirrors the paper, and the only controls are the prompt and, with AI, two buttons, so a participant who glances up finds the same place every time.
- **The conditions look alike:** both pages have the same header, view, size, prompt, and status words. Only the two AI buttons and the overlay they produce differ, which is exactly the AI help being compared.
- **No time pressure or model talk:** participants see no clock, counter, model name, or experiment name.
- **Experimenter errors are the main risk to validity:** an unset condition, a reused name, or a carried-over condition would corrupt a drawing's label, so Start needs both, clears both, and the record and the panel status show the condition.
- **The experimenter finishes:** Finish sits in the panel, so a participant cannot end a drawing by accident, and saying "done" is part of the protocol.

The frontend-developer must confirm in a real browser whether the webcam keeps streaming while the accordion is closed. If Gradio stops it, the webcam moves to a small "Camera" view under the sketch view in the participant area, and the rest of the panel stays collapsed.

## Failure paths

| Case | Behavior |
| --- | --- |
| Start without a name or condition | Start stays disabled. A library call with an AI condition and no suggester, or `no_ai` with one, raises `SketchLoopError` before anything is saved. |
| Name invalid or taken | Unchanged from T09: one line, nothing created, and the name and condition stay for correction. |
| Suggest a prompt in a no-AI drawing | The button does not exist, and a direct call raises "Suggestions are off in this drawing." with nothing recorded. |
| Prompt suggester fails, such as missing weights | A failed `PromptSuggestionRecord` is saved with the error, the page shows one line, and the prompt box and kept ID stay as they were. |
| Saving a prompt suggestion fails | The entry is removed again and a one-line `StoreError` is shown, so the text is not written into the box and the record never claims it was shown. |
| Unknown prompt suggestion ID, from a stale tab | The origin is recorded as unavailable with the reason, and the suggestion, round, or finish goes ahead. |
| Prompt suggestions outside a drawing, then the page closes without a round | Lost, on the generation page only. Documented as a limit. |
| Fake stroke suggester or template prompts in study mode | Allowed for dry runs, with warnings in the terminal and the panel, and the suggester identity recorded in every suggestion. |
| Webcam frame stale at Finish | Unchanged: Finish refuses, so a drawing never ends without its final capture unless the session is replaced. |
| Export: runs folder missing, nothing to export, output not empty, inside the runs folder, or not ignored by Git | `ExportError` or a CLI refusal with one line, and nothing written. |
| Export: a session fails to load | A skipped line, and the export goes on. |
| Export: a drawing has no usable final canvas | A missing row in `key.csv` with its reason, printed, and no file. |
| Export: a write fails midway | The temporary folder is removed and `ExportError` names the step. |

## Privacy

- **Names are codes:** the page keeps the hint that names must be codes and never real names, and experiment names should not encode the condition, since the condition is recorded separately and the name is printed in the panel and by `show` and `summary`.
- **The export carries no label:** the drawings folder holds only `<code>.png` files written in code order and re-encoded without metadata, with random codes from `secrets`. The images come from the same pipeline in both conditions, and no overlay is ever drawn on the canvas.
- **The key stays local:** `key.csv` and `export.json` sit beside `drawings/` in the export folder, which defaults to the ignored `data/` folder, and the command refuses a folder Git does not ignore. Handing the drawings to the supervisor is the owner's action outside the repository.
- **Participant data stays in ignored storage:** captures, records, and exports stay under `runs/` and `data/`, as [PRIVACY.md](../PRIVACY.md) and S05 require. Test fixtures use synthetic names such as `p1_car`, and nothing in the repository holds real participant codes.

## Conflicts and deferrals

- **S03 in PROJECT.md** said a study session records "the design task". Under owner decision 5 and ADR 0025 the system never stores a task, and the name, which the experimenter may compose from a participant code and a task, is the only label. With the owner's approval, S03 now reads: "A study drawing records the condition, with or without AI, in an experiment named by a pseudonymous code the experimenter chooses. Without AI, the same capture and records run with all AI help turned off."
- **The T10 row in ROADMAP.md** said `sketchloop study` runs a participant through the tasks. Owner decision 2 makes it `sketchloop ui --study`, one experiment per start, with the name and condition only, and the row's runnable increment now reads "`sketchloop ui --study` runs each experiment on a participant page, and `sketchloop study export` produces the drawings for the experts".
- **U1 and NEXT-STEPS item 52** placed prompt suggestion records after the study. Owner decision 1 moves them into T10, as this design does, and the U1 row in ROADMAP.md says so.
- **ADR 0019, amended in the UI session,** chose the prompt suggester with the backend, which would have forced a real backend into the study's settings only to get Florence-2. The owner chose a `[prompt_suggester]` settings table instead, and ADR 0019 now carries a T10 amendment saying so.
- **ADR 0021** says the no-AI condition has suggestions turned off. Owner decision 1 widens this to all AI help, so `suggestions_enabled` now covers prompt suggestions too. No saved meaning changes, because the T09 page offered both whenever it was true.
- **ADR 0025** reserved `suggestions_enabled` and a study object for T10. This design uses both as reserved and moves nothing.
- **The final prompt** was not in the owner's first list. The owner accepted it, because the prompt box exists in both conditions while only the AI condition records prompts through its suggestions, so without it the no-AI drawings would hold no prompt at all.
- **Deferred:** a plan file of names and conditions, a CSV from `summary`, reusing codes across exports, exporting the re-warped raw frame, storing the sketch the prompt suggester saw, linking rounds to the drawing they happened in, and resuming an experiment after the server stops.

## File plan

Each slice is one commit with its tests, and updates [the call graph](../call-graph.html), `docs/EXPERIMENTS.md` where records change, and traceability. B is the `backend-developer` and F the `frontend-developer`. The `ux-reviewer` reviews slices F1 and F2 before they are committed.

| Slice | Items | Who | Files, in order |
| --- | --- | --- | --- |
| B1. Prompt suggestion records | U1 | B | `suggestion.py`: the `identity` property on `PromptSuggester`, `TemplateSuggester`, and `FlorenceSuggester`. `experiments/prompts.py`: the enums, `PromptOrigin`, `PromptSuggestionFailure`, `PromptSuggestionRecord`, `classify_prompt_origin`, and the dict helpers. `experiments/__init__.py`: exports. |
| B2. Drawing study fields | 37, 38, U1 | B | `experiments/drawings.py`: `StudyCondition`, `DrawingStudy`, the four `DrawingRecord` fields, `SuggestionRecord.prompt_origin`, their serialization and invariants. `drawing.py`: `record_prompt_suggestion`, `find_prompt_suggestion`, and the `prompt_suggestion_id` and `prompt` arguments. `session.py`: `start_drawing(study=...)` and `record_prompt_suggestion`. |
| B3. Round prompt fields | U1 | B | `experiments/records.py` and `serialization.py`: the two `RoundRecord` fields. `session.py`: the pending list and `run_round(prompt_suggestion_id=...)`. `experiments/loading.py`: the session-wide origin check. |
| B4. Controller | 37, 38, 39, U1 | B | `app.py`: `PromptSuggestionView`, `DrawingView` fields, `start_drawing(name, condition)`, `suggest_prompt` recording and refusal, the ID arguments, and `describe_prompt_suggester`. |
| F1. Prompt suggestion ID on the generation page | U1 | F | `ui.py`: the per-tab `gr.State`, writing `PromptSuggestionView.text`, and sending the ID with Suggest next strokes, Generate, and Finish. |
| B5. Settings, serve, and show | 34, 35, 36, U1 | B | `settings.py`: `StudySettings` and `PromptSuggesterSettings`. `components.py`: `PROMPT_SUGGESTERS`, `make_prompt_suggester`, and the name check. `cli/arguments.py`: `--study/--no-study`. `cli/serve.py`: the prompt suggester from settings, the study lines, and passing the flag. `cli/show.py`: condition, prompt column, prompt suggestions, final prompt. `examples/sketchloop.toml`: the `[study]` and `[prompt_suggester]` tables. Then the U1 row and `docs/STATUS.md`. |
| F2. Study page | 37, 38, 39, 40 | F | `ui.py`: `SketchPage(study=...)`, the study build method, the participant area, the experimenter panel with the condition radio, the visibility outputs, the auto-closing panel, the saving label, and the neutral texts. `static/ui.css`: the `study` class. `launch_app(study=...)`. |
| B6. Blinded export | 41, 42 | B | `experiments/study_export.py`. `cli/arguments.py`: the `study` subcommand with `export`. `cli/study.py`: the default folder, the Git check, and the printout. `cli/__init__.py`: dispatch. |
| B7. Summary | 43 | B | `cli/study.py`: `summary`, reusing `decide_suggestion` for open windows. |
| Pilot | 44, 46 | Owner | Run the protocol below, then fix what it finds as its own slices. |

B6 depends only on B2 and can be built in parallel with B3 to F2. F1 can start once B4's signatures exist, and F2 after B5.

## Tests for the qa-engineer

Default checks stay offline: no camera, GPU, weights, or network. The Git check test skips when Git is missing. Experiment names in fixtures are synthetic codes.

1. **Prompt origin, U1, R06 (parametrized):** `classify_prompt_origin` gives `typed` with no suggestion, `typed` for a blank or spaces-only prompt even with a suggestion, `suggested` for equal text with different surrounding spaces, and `edited` for one changed word and for completely different text. A `PromptOrigin` of kind `typed` with an ID, or `edited` without one, is refused.
2. **Prompt suggestion records, U1, R06:** a shown record without text, a failed one with text, `suggester_seconds` with `ran_suggester` false, and a negative `list_position` are refused. Records round-trip through the dict helpers unchanged. `prompt_origin_from_dict` turns a missing value into the not-recorded reason and does not misread an origin object as unavailable.
3. **Identity, R08:** `TemplateSuggester().identity` is in-process with an unavailable model ID and not fake. `FlorenceSuggester` with a stub captioner and a fake models folder names the repository and revision. A test subclass without `identity` cannot be instantiated.
4. **Drawing condition, S03, R06:** with `MemoryStore`, a `no_ai` study drawing has suggestions off, no suggester, an empty prompt suggestion list, and refuses `suggest` and `record_prompt_suggestion`. An `ai` study drawing without a suggester, or `no_ai` with one, is refused before anything is saved. Records with `study.condition: no_ai` and `suggestions_enabled: true`, or with prompt suggestions while suggestions are off, fail on load.
5. **Drawing prompt links, U1, S04:** a prompt suggestion, then a stroke suggestion with its ID and the same text, records `suggested`, and with one word changed records `edited`, both round-tripping through `drawing.json`. An ID from another drawing records an unavailable origin and the suggestion is still shown. A record whose origin kind disagrees with the stored texts, or names a prompt suggestion shown after the stroke suggestion, fails on load. A failed save of a prompt suggestion leaves no entry in memory or on disk.
6. **Final prompt, S04:** finishing with a prompt and an ID records both, finishing with an empty prompt records `""` and `typed`, and ending by a new session records None for both. An open drawing with a final prompt fails on load.
7. **Backward compatibility, R06:** a T09 `drawing.json` and a T08 `round.json` without the new keys load with `study` None, `prompt_suggestions` None, and unavailable origins and final prompt, and `show` prints "not recorded". The schema version stays 1.
8. **Rounds, U1:** prompt suggestions shown with no drawing open are written into the next round and not into the one after. A round whose origin names a suggestion in an earlier round loads, and an unknown ID gives a problem line from `load_session`, not an error.
9. **Controller, S03, R07:** `start_drawing("p1_car", StudyCondition.NO_AI)` gives a view with the condition and suggestions off, and `suggest_prompt` then raises without calling the suggester. With `AI`, `suggest_prompt` returns a view whose ID is in the drawing's record, and a second press on the same sketch has `ran_suggester` false. A suggester stub that raises records a failed entry and re-raises one line.
10. **Page handlers without a browser, R07:** in study mode the page builds no Generate, gallery, choice, seed, sliders, or new-session components, and no header text names the backend or suggesters. Start is disabled without a name or without a condition. Start with `no_ai` returns invisible updates for both suggest buttons and `ai` visible ones, clears the name and condition, and closes the panel. The participant status for an open drawing is the same string in both conditions and contains neither the name nor the condition. The prompt placeholder names no object. The kept ID is sent with Suggest next strokes, Generate, and Finish, and cleared by Start.
11. **Settings and CLI, R08:** `[study] enabled = true` serves the study page, `--no-study` overrides it, and a non-boolean `enabled` fails with one line naming the key. `export_dir` resolves relative to the settings file. `[prompt_suggester] name` builds `TemplateSuggester` by default and `FlorenceSuggester` for `florence` with the fake image backend, and an unknown name fails with one line naming the key before anything is created.
12. **Export, S04, S05 (with an injected code maker and `tmp_path`):** a runs folder with one `ai` and one `no_ai` finished drawing, one unfinished, one finished without a frame, one with a changed canvas, a T09 drawing without `study`, a time-named session, and a broken `session.json` gives two PNGs named by the injected codes, a `key.csv` with the `KEY_COLUMNS` header holding two exported and three missing rows with their reasons, a skipped line for the broken session, and an `export.json` with the counts. The exported PNGs decode to the recorded canvases' pixels, contain no text chunks, and their names and bytes contain neither experiment name. `--match "p1_*"` limits the rows. A non-empty output folder, an output inside the runs folder, and a runs folder with no study drawings are refused with nothing written. A write that fails midway leaves neither the output nor a temporary folder. A code collision draws again.
13. **Git check, S05:** with a temporary Git repository, an output folder under an ignored path passes and one under a tracked path is refused. Without a repository, the check is skipped with one line.
14. **Summary, S04:** two drawings print one row each with their condition, counts, origin counts, and final-drawing state, and an open window is marked as computed.

## Checks only a person can do

In the pilot on the M2 Pro, with the real suggesters, before T10 is called done:

- **Webcam:** the stream keeps running and the overlay keeps lining up while the experimenter panel is closed for a whole drawing.
- **Same page:** in both conditions, the page looks the same apart from the two buttons, and nothing shifts when they are removed. Check light and dark modes and readability from the drawing position.
- **Experimenter flow:** starting and finishing eight experiments in a row, with the condition order varied, never carries a name or condition over, and a mistyped name is caught before anything is saved.
- **Prompt suggestions:** Florence-2 prompts appear in the AI condition, and the records show typed, suggested, and edited correctly for prompts the pilot participant actually used.
- **Export:** the exported drawings are readable enough for the experts at the canvas size, the key joins back to every experiment, and `git status` shows nothing new.
- **Follow rule:** hand labels of the pilot's suggestions against the recorded decisions, carried over from T09.
- **Disk and timing:** disk use per drawing and the time per suggestion with both suggesters loaded.

## Out of scope

- A task field, task list, or anything that tells the system the task.
- A participant identity system or demographics.
- Continuous or automatic suggestions, which [ADR 0021](../decisions/0021-guided-sketching-and-user-study.md) rejected.
- Changing the follow rule, which waits for the pilot's hand labels.
- Everything listed as deferred above.

## Open questions for the owner

None remain. The owner answered every question this design raised, as listed under [Decisions this design applies](#decisions-this-design-applies).
