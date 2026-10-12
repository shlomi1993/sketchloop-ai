# Gradio implementation guidance for the T06 interface

## Question and decision informed

How should T06 (R07) build the local research interface with Gradio, which [ADR 0006](../decisions/0006-first-backend-store-and-ui.md) already chose? This note answers six concrete questions: the version to pin and its footprint, telemetry and local-only serving, browser webcam capture that keeps the original encoded bytes, multi-select of candidates, progress and errors for a slow CPU round with one loaded backend, and offline testing. It informs the `ui` optional extra, the module split under `src/sketchloop/`, and any ADR amendment. The toolkit comparison stays in [UI toolkit options](ui-toolkit-options.md).

## Sources (read 2026-10-12)

Source code was read on the `main` branch while its `gradio/package.json` reported 6.30.0, which matches the latest PyPI release. Line-level behavior can change in later releases.

- PyPI JSON: [gradio 6.30.0](https://pypi.org/pypi/gradio/6.30.0/json), uploaded 2026-10-08, Apache-2.0. [gradio-client 2.7.2](https://pypi.org/pypi/gradio-client/2.7.2/json), uploaded 2026-10-02, Apache-2.0. [hf-gradio 0.4.1](https://pypi.org/pypi/hf-gradio/json), MIT.
- Gradio docs: [Image](https://www.gradio.app/docs/gradio/image), [Gallery](https://www.gradio.app/docs/gradio/gallery), [CheckboxGroup](https://www.gradio.app/docs/gradio/checkboxgroup), [SelectData](https://www.gradio.app/docs/gradio/selectdata), [Progress](https://www.gradio.app/docs/gradio/progress), [Error](https://www.gradio.app/docs/gradio/error), [State](https://www.gradio.app/docs/gradio/state), [Blocks and launch](https://www.gradio.app/docs/gradio/blocks).
- Gradio guides: [environment variables](https://www.gradio.app/guides/environment-variables), [file access](https://www.gradio.app/guides/file-access), [state in Blocks](https://www.gradio.app/guides/state-in-blocks), [queuing](https://www.gradio.app/guides/queuing), [Gradio 6 migration](https://www.gradio.app/main/guides/gradio-6-migration-guide), [alerts](https://gradio.app/guides/alerts).
- Gradio source on GitHub: [components/image.py](https://github.com/gradio-app/gradio/blob/main/gradio/components/image.py), [image_utils.py](https://github.com/gradio-app/gradio/blob/main/gradio/image_utils.py), [components/image_editor.py](https://github.com/gradio-app/gradio/blob/main/gradio/components/image_editor.py) for `WebcamOptions`, [js/image/shared/Webcam.svelte](https://github.com/gradio-app/gradio/blob/main/js/image/shared/Webcam.svelte), [js/image/shared/stream_utils.ts](https://github.com/gradio-app/gradio/blob/main/js/image/shared/stream_utils.ts), [analytics.py](https://github.com/gradio-app/gradio/blob/main/gradio/analytics.py), [blocks.py](https://github.com/gradio-app/gradio/blob/main/gradio/blocks.py), [events.py](https://github.com/gradio-app/gradio/blob/main/gradio/events.py), [helpers.py](https://github.com/gradio-app/gradio/blob/main/gradio/helpers.py), [themes/default.py](https://github.com/gradio-app/gradio/blob/main/gradio/themes/default.py), [CHANGELOG.md](https://github.com/gradio-app/gradio/blob/main/gradio/CHANGELOG.md), and [test/conftest.py](https://github.com/gradio-app/gradio/blob/main/test/conftest.py).
- MDN: [Secure contexts](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts) and [getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

## Findings

### 1. Version, Python support, and footprint

- Verified: the latest stable release is `gradio==6.30.0`, uploaded 2026-10-08. The earlier note pinned 6.29.0. The 6.30.0 changelog touches `gr.Workflow`, `ImageEditor` alpha, Accordion, Chatbot, and LaTeX, and nothing this app uses.
- Verified: `requires_python >=3.10`, with classifiers for 3.10 to 3.13. The wheel is pure Python (`py3-none-any`, about 31 MB), so Windows, macOS, and Linux on 3.11 and 3.12 are covered.
- Verified: direct requirements are `anyio`, `brotli`, `fastapi<1.0,>=0.115.2`, `gradio-client==2.7.2`, `groovy`, `hf-gradio`, `httpx`, `huggingface-hub<3.0,>=1.16.0`, `jinja2`, `markupsafe`, `numpy<3.0`, `orjson`, `packaging`, `pandas<4.0`, `pillow<13.0`, `pydantic>=2.0`, `pydub`, `python-multipart`, `pytz`, `pyyaml`, `safehttpx`, `semantic-version`, `starlette<2.0,>=1.0.1`, `tomlkit`, `typer`, `typing-extensions`, and `uvicorn`. `audioop-lts` is added only on Python 3.13 and later. `gradio-client` adds `fsspec`, and `hf-gradio` adds a CLI on top of `gradio-client` and `typer`.
- Verified: the `generation` extra pins `huggingface_hub==1.33.0`, which satisfies Gradio's `>=1.16.0,<3.0` range. The project's `numpy` and `opencv-python` stay compatible with `numpy<3.0`.
- Inference: the full install is roughly 150 to 250 MB with pandas, pydantic-core, orjson, and brotli wheels. It belongs in an optional `ui` extra, as ADR 0006 already says.

### 2. Telemetry and local-only serving

- Verified: `analytics.analytics_enabled()` returns `os.getenv("GRADIO_ANALYTICS_ENABLED", "True") == "True"`, so analytics are on by default and any other value disables them.
- Verified: `gr.Blocks(analytics_enabled=...)` takes precedence over the variable. When analytics are enabled, `Blocks.__init__` starts a `version_check` thread that requests `https://api.gradio.app/pkg-version` and sends `initiated_analytics`, and `launch()` sends `launched_analytics` with the app's component structure. When disabled, `Blocks.__init__` sets `HF_HUB_DISABLE_TELEMETRY=True`. `error_analytics` would send raw error text, which could contain local paths.
- Verified: `analytics.py` makes no network request at import time. Its top level only defines constants and an optional pyodide import.
- Verified: the default theme uses `LocalFont` for both fonts, so the page does not fetch Google Fonts.
- Verified: `launch(share=None)` defaults to `True` on Colab and hosted notebooks, otherwise to `GRADIO_SHARE`. An explicit `share=False` overrides both. `server_name=None` binds to `GRADIO_SERVER_NAME` or `127.0.0.1`. `ssr_mode=None` reads `GRADIO_SSR_MODE`, which defaults to off outside Hugging Face Spaces, and SSR would need Node 20+.
- Verified: `launch()` makes local requests to its own server for startup checks. It calls `networking.setup_tunnel` only when sharing.
- Recommendation: pass `analytics_enabled=False` to `gr.Blocks` and also set `GRADIO_ANALYTICS_ENABLED=False` and `HF_HUB_DISABLE_TELEMETRY=1` before `gradio` is imported. Launch with `server_name="127.0.0.1"`, `share=False`, and `ssr_mode=False` passed explicitly.

### 3. Webcam capture and original bytes

- Verified: `gr.Image` has `sources: list["upload", "webcam", "clipboard"]`, `type: "numpy" | "pil" | "filepath" = "numpy"`, `image_mode = "RGB"`, `format = "webp"`, `streaming = False`, and `webcam_options: gr.WebcamOptions | None`. Gradio 6 removed `mirror_webcam` and `webcam_constraints` in favor of `webcam_options`.
- Verified: a non-streaming webcam snapshot draws the video frame to a canvas and encodes it with `canvas.toBlob(..., "image/png")` in the browser, then uploads it like a file. The default `getUserMedia` constraints ask for an ideal 1920 x 1440. A device picker comes from `enumerateDevices`.
- Verified: `gr.WebcamOptions` has `mirror: bool = True` and `constraints: dict | None = None`. With `mirror=True`, `take_picture` draws the canvas flipped, so the saved snapshot is mirrored, not only the preview. A mirrored sketch would reach preprocessing and the model, so the UI must pass `gr.WebcamOptions(mirror=False)`.
- Verified: in `image_utils.preprocess_image`, when `type="filepath"` and `image_mode` is `None` or equals the file's mode, the function returns the cached upload path unchanged, with no re-encode, EXIF transpose, or mode conversion. With the default `image_mode="RGB"`, a file in another mode, such as an RGBA PNG or a grayscale scan, is converted and re-saved. With `type="numpy"` or `"pil"`, the encoded bytes are lost.
- Verified: uploads are moved into Gradio's cache, under `GRADIO_TEMP_DIR` or the system temp directory, and `gr.Blocks(delete_cache=(frequency_seconds, age_seconds))` cleans it. The cache is also cleared when the server restarts.
- Inference: with `type="filepath"` and `image_mode=None`, the handler receives the browser-encoded PNG for webcam captures and the person's original file bytes for uploads. `capture.load_sketch_file(Path(path))` can then read, sniff, checksum, and describe it exactly like a CLI file, so no new capture code is needed. For a webcam, the browser PNG is the earliest encoding the server can see. The camera's own frames are never available, so the record should say the raw sketch is a browser snapshot.
- Verified: browser camera access needs a secure context. `http://localhost` and `http://127.0.0.1` qualify, but a LAN address over plain HTTP does not. Gradio's frontend reports "no webcam support" when `navigator.mediaDevices` is missing and asks for permission on `NotAllowedError`. On Colab, the share link is HTTPS, so the webcam works there.
- Inference: the browser webcam gives a live preview by itself, which settles the ADR 0006 risk about live preview without `gr.Timer` or NiceGUI. The CLI's OpenCV capture remains available for server-side cameras.

### 4. Selecting zero or more candidates

- Verified: `gr.Gallery(columns=2, allow_preview=True, interactive=None, type="filepath", format="webp", buttons=None, ...)` accepts a list of paths or `(path, caption)` tuples. Its `select` event gives `gr.SelectData` with `index`, `value`, and `selected`. The docs describe no multi-select, and a click opens the preview.
- Verified: `gr.CheckboxGroup(choices=[(label, value), ...], value=None, type="value")` passes the list of checked values, and `type="index"` passes indices. Returning `gr.CheckboxGroup(choices=..., value=[])` from a handler replaces the options.
- Verified: returned file paths are served only from the working directory, the system temp directory, or `launch(allowed_paths=[...])`, and are copied into the cache. `runs/` under the working directory works by default, but `--runs-dir` elsewhere needs `allowed_paths=[str(runs_dir)]`.
- Recommendation: show candidates in a non-interactive `gr.Gallery` with captions "1", "2", and so on, plus a "Fake backend" label when `result.backend.is_fake`. Put a `gr.CheckboxGroup` beside it whose values are candidate IDs and labels are the same numbers. One "Record selection" button calls `select_candidates` or `record_no_selection` when nothing is checked. Using `Gallery.select` toggles would need hidden state and click semantics that also open the preview, so it is more fragile than an explicit checkbox list.
- Recommendation: set `format="png"` on the gallery so any non-path output is not re-encoded to WebP for display. Return paths anyway, so the browser shows the stored files.

### 5. Progress, errors, and one loaded backend

- Verified: the queue is always on. Each event listener's `concurrency_limit` defaults to 1, set globally with `Blocks.queue(default_concurrency_limit=...)` or `GRADIO_DEFAULT_CONCURRENCY_LIMIT`. Listeners sharing a `concurrency_id` share the lowest limit. `trigger_mode="once"`, the default for clicks, ignores repeated clicks while an event is pending.
- Verified: `gr.Progress(track_tqdm=False)` is declared as a default argument after the inputs, and is called as `progress(fraction_or_tuple, desc=..., total=..., unit="steps")` or `progress.tqdm(iterable)`. Outside an event, the call is a no-op that returns its argument.
- Verified: with `track_tqdm=True`, Gradio patches `tqdm.tqdm` methods on the class and rebinds `tqdm.auto.tqdm`. Inference: the Diffusers pipeline step bar is a `tqdm.auto` subclass of `tqdm.tqdm`, so it should appear in the browser without changes to `diffusers_backend.py`. This must be tried on the laptop.
- Verified: `raise gr.Error(message, duration=10, title="Error", print_exception=True)` stops the handler and shows a modal. `gr.Warning(message)` and `gr.Info(message)` are called, not raised, and outside an event they fall back to `warnings.warn` and `print`. `launch(show_error=False)` is the default. Inference: other exceptions then show a generic error without the message, so handlers should convert `SketchLoopError` to `gr.Error` with the same one-line message.
- Verified: `gr.State` deep-copies its initial value, requires deep-copyable values, and is reset on page refresh. The state guide says module-level variables are shared by every user and suggests a global dictionary keyed by `request.session_hash` for objects that cannot be deep-copied.
- Recommendation: build the generator once before `build_app` and hold it, with the current `SketchSession`, in a plain controller object closed over by the handlers. Do not put a Torch pipeline in `gr.State`, and avoid a module-level global, so tests can inject a fake. For this single-user local app, every tab shares the one controller. Give generate, select, and new-session listeners the same `concurrency_id="session"` with `concurrency_limit=1`, so a selection cannot interleave with a running round.

### 6. Offline testing and import side effects

- Verified: handlers are plain Python functions, `gr.Progress` and `gr.Warning` degrade to no-ops or warnings outside an event, and Gradio's own fixtures build `gr.Blocks` without `launch()`.
- Verified: with `analytics_enabled=False`, `Blocks.__init__` starts no version-check thread. Inference: building the Blocks object in a test therefore makes no network call. This should be confirmed once with a socket-blocking fixture.
- Inference: importing `gradio` pulls FastAPI, pandas, pydantic, and huggingface_hub and takes a few seconds. No import-time network call was found in `__init__.py` or `analytics.py`, but not every transitive module was read.
- Recommendation: keep the controller in a module that never imports `gradio`, so default tests cover it with `FakeGenerator` and `tmp_path` and need no `ui` extra. A single optional test uses `pytest.importorskip("gradio")` to build the Blocks object without launching it.

## Options compared

| Question | Option | Verdict |
| --- | --- | --- |
| Image value type | `numpy` or `pil` | Loses encoded bytes and checksum provenance |
| Image value type | `filepath` with `image_mode=None` | Recommended. The original upload or browser PNG reaches `load_sketch_file` |
| Multi-select | `Gallery.select` toggles in `gr.State` | Fragile, since a click also opens the preview |
| Multi-select | `Gallery` plus `CheckboxGroup` of candidate IDs | Recommended. Explicit, testable, and empty means none |
| Backend lifetime | `gr.State` | Deep-copies and resets on refresh, unsuitable for a pipeline |
| Backend lifetime | Module global | Works but hides the dependency from tests |
| Backend lifetime | Controller injected into `build_app` | Recommended |

## Recommendation and confidence

Pin `gradio==6.30.0` in a new `ui` extra. Capture with `gr.Image(sources=["webcam", "upload"], type="filepath", image_mode=None, webcam_options=gr.WebcamOptions(mirror=False))`, and pick candidates with a gallery plus a checkbox group. Hold one generator and session in an injected controller, and serialize session events with a shared `concurrency_id`. Disable analytics in code and by environment, and launch on `127.0.0.1` with `share=False`. Confidence: high for the APIs, which were read in source and docs. It is medium for the `track_tqdm` progress with Diffusers and for the absence of import-time network calls, which must be checked on the laptop.

## Implementation guidance

- Target modules, for the architect to confirm: a gradio-free `sketchloop/ui_controller.py` (or a `ui/` package) holding the session and returning plain values, and `sketchloop/gradio_app.py` with `build_app(controller, runs_dir) -> gr.Blocks` and the launch call. A `sketchloop ui` subcommand imports the Gradio module lazily, so the CLI works without the extra. The ROADMAP's move of `GenerationBackend` and `make_generator` into `generation.py` comes first.
- `pyproject.toml`: `ui = ["gradio==6.30.0"]`. Add `.gradio/` to `.gitignore` in case Gradio writes local state there.
- Errors: convert `SketchLoopError` to `gr.Error(str(error))`. Unsupported controls raise `UnsupportedConfigurationError`, which is a `SketchLoopError`. The session still writes a failed round record, which the UI should mention.

Minimal outline (illustrative, not project code):

```python
# In the CLI's `ui` command, before importing the Gradio module.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from sketchloop.gradio_app import launch_app

# sketchloop/ui_controller.py: no gradio import.
class UiController:
    def __init__(self, generator: Generator, runs_dir: Path) -> None:
        self.generator = generator
        self.runs_dir = runs_dir
        self.session = SketchSession(generator, runs_dir)

    def generate(self, sketch_path: str | None, prompt: str, n_candidates: int) -> list[tuple[str, str, str]]:
        if not sketch_path:
            raise InvalidSketchError("Capture or upload a sketch first.")

        raw_sketch, payload = load_sketch_file(Path(sketch_path))
        outcome = self.session.run_round(raw_sketch, payload, Guidance(prompt=prompt), n_candidates=n_candidates)
        fake = " (fake backend)" if outcome.iteration.result.backend.is_fake else ""
        return [(str(outcome.folder / candidate.image.path), f"{candidate.index + 1}{fake}", candidate.id)
                for candidate in outcome.iteration.result.candidates]

    def record_choice(self, candidate_ids: list[str]) -> None:
        iteration = self.session.rounds[-1].iteration
        selection = select_candidates(iteration, candidate_ids) if candidate_ids else record_no_selection(iteration)
        self.session.record_selection(selection)

# sketchloop/gradio_app.py
def build_app(controller: UiController) -> gr.Blocks:
    with gr.Blocks(analytics_enabled=False, title="SketchLoop", delete_cache=(3600, 3600)) as app:
        sketch = gr.Image(sources=["webcam", "upload"], type="filepath", image_mode=None,
                          webcam_options=gr.WebcamOptions(mirror=False), label="Sketch")
        prompt = gr.Textbox(label="Prompt")
        count = gr.Slider(1, 4, value=2, step=1, label="Candidates")
        generate = gr.Button("Generate", variant="primary")
        gallery = gr.Gallery(columns=4, interactive=False, format="png", label="Candidates")
        picks = gr.CheckboxGroup(label="Selected candidates")
        record = gr.Button("Record selection")
        status = gr.Markdown()

        def run_generation(path: str | None, text: str, number: float,
                           progress: gr.Progress = gr.Progress(track_tqdm=True)) -> tuple[list, gr.CheckboxGroup]:
            progress(None, desc="Generating")
            try:
                items = controller.generate(path, text, int(number))
            except SketchLoopError as error:
                raise gr.Error(str(error)) from error

            return [(path, label) for path, label, _ in items], gr.CheckboxGroup(choices=[(label, cid) for _, label, cid in items], value=[])

        def save_selection(candidate_ids: list[str]) -> str:
            try:
                controller.record_choice(candidate_ids)
            except SketchLoopError as error:
                raise gr.Error(str(error)) from error

            return f"Recorded {len(candidate_ids) or 'no'} selection(s). Change the sketch or prompt and generate again."

        generate.click(run_generation, [sketch, prompt, count], [gallery, picks], concurrency_id="session")
        record.click(save_selection, [picks], [status], concurrency_id="session")
    return app

def launch_app(controller: UiController, runs_dir: Path) -> None:
    build_app(controller).launch(server_name="127.0.0.1", share=False, ssr_mode=False, inbrowser=True,
                                 allowed_paths=[str(runs_dir.resolve())], max_file_size="20mb", show_error=False)
```

Parameters to record: the existing round record already covers request, settings, seeds, checksums, and timings. Add the capture source, browser webcam or browser upload, if the architect wants it in `RoundRecord`, since both arrive as a file path. `collect_environment` should list `gradio` among package versions when the UI runs.

Pitfalls:

- The default `image_mode="RGB"` silently re-encodes non-RGB files, and the default `mirror=True` flips webcam snapshots.
- Phone JPEGs with EXIF orientation reach preprocessing unrotated, the same as in the CLI. The existing `rotation` option covers this.
- A `--runs-dir` outside the working directory needs `allowed_paths`, or showing the gallery raises an error.
- Sketches also stay in Gradio's temp cache until `delete_cache` or a restart removes them.
- Two open tabs share one controller and session, which is acceptable for a single-user local tool but should be stated in the UI help.
- Gradio 6 moved `theme`, `css`, and `head` from `gr.Blocks` to `launch()`, so older examples on the web fail.
- `share=True` sends sketches through Gradio's relay and is the owner's decision.

Test ideas:

- Default, no `ui` extra: `UiController` with `FakeGenerator` and `tmp_path`. Cover the round folder and gallery paths, fake labels, empty sketch path errors, selecting none or several IDs, a second round linking to the first, and a failed round still writing `round.json`. Write a PNG fixture with OpenCV to simulate the browser file path.
- Optional, `pytest.importorskip("gradio")`: build the app with a fake-backed controller, assert it is a `gr.Blocks`, and check that `analytics_enabled` is false. Use a fixture that blocks `socket.socket.connect` to confirm no network access, and do not launch.
- Manual, recorded in STATUS: on the laptop, check that the webcam snapshot is not mirrored, that the Diffusers step progress appears, and that a 100 s round completes without a timeout.

## Open questions for the owner

- Should the record distinguish a browser webcam snapshot from a browser upload, and from the CLI's OpenCV capture?
- Should the UI start a new session on page refresh, or keep one session for the life of the server, which is the simpler default above?
- Is it acceptable to keep the CLI as the only path for `show` and `rerun` in T06, with a read-only history view later?
