# ADR 0012: Local Gradio interface, app controller, and sketch input source

Status: accepted

## Context

T06 (R07) needs a local interface for capture, guidance, a manual trigger, progress and errors, a candidate gallery, explicit selection, the next round, and history. [ADR 0006](0006-first-backend-store-and-ui.md) chose Gradio as an optional extra with analytics off. [The Gradio implementation note](../research/gradio-ui-implementation.md) verified Gradio 6.30.0. Its findings are that `gr.Image(type="filepath", image_mode=None)` passes the uploaded or browser-encoded bytes unchanged, that webcam snapshots are mirrored unless `mirror=False` is set, that `gr.State` deep-copies and resets on reload, and that analytics are on unless disabled. A Torch pipeline takes minutes to load on the laptop CPU, so it must stay loaded across rounds and reloads ([ADR 0010](0010-sessions-and-evaluator.md)).

The owner decided that round records name where the raw sketch came from, that a browser reload keeps the same session, and that the UI history covers only the current session while `show` and `rerun` stay in the CLI. [ADR 0011](0011-session-records-and-subcommands.md) fixed `round.json` at schema version 1 and said a changed field needs a version bump. Sessions saved since T05, including webcam captures, exist without the new field.

## Decision

Adopt [the T06 design](../design/t06-ui.md):

- **Controller:** a new `sketchloop.app.AppController` that never imports gradio. It holds one injected generator and the current `SketchSession`, and returns plain view records. It refuses a selection for any round except the latest completed round shown to the person. One instance per process is closed over by the Gradio handlers, never in `gr.State` and never a module global. `SketchSession` exposes `records`, which includes failed rounds, for the history.
- **Gradio module:** `sketchloop.ui` with `build_app` and `launch_app` is the only module that imports gradio. It has separate webcam and upload images chosen by a radio, because one image with two sources cannot say which source was used. Candidates appear in a gallery with a checkbox group of candidate IDs, where nothing ticked means an explicit none. State-changing events share `concurrency_id="session"` with a limit of 1, and page load restores the latest round and history. `SketchLoopError` and `OSError` become `gr.Error` with the same one-line message.
- **Serving:** `gr.Blocks(analytics_enabled=False)`, and `GRADIO_ANALYTICS_ENABLED=False` and `HF_HUB_DISABLE_TELEMETRY=1` are set before import. It launches on `127.0.0.1` with `share=False`, `ssr_mode=False`, and `allowed_paths` limited to the runs folder.
- **CLI:** `sketchloop ui` with `--backend`, `--mode`, `--runs-dir`, `--candidates`, and `--seed`. It imports gradio only when it runs.
- **Dependency:** `gradio==6.30.0` (Apache-2.0) is a core dependency, amended from an optional `ui` extra after implementation because the UI is now the main way to use SketchLoop. This overrides the optional-extra choice in [ADR 0006](0006-first-backend-store-and-ui.md). It makes installs and CI heavier, so `sketchloop ui` still imports gradio lazily and the other commands start without it.
- **Input source:** `round.json` gains `input.source`, and `RoundRecord` gains `source`, with the value `file`, `camera` (OpenCV on the host), `webcam` (the newest streamed browser frame, saved as PNG), or `upload` (browser upload). `run_round` requires it as a keyword. A rerun copies the original round's value, because it reuses the same raw bytes and `session.json` `rerun_of` already marks the rerun.
- **Schema rule:** `SCHEMA_VERSION` stays 1. A record without `input.source` loads as `Unavailable("recorded before input sources were tracked")`. This refines ADR 0011: adding an optional field whose absence has a documented unavailable value keeps the version, while renaming, removing, or changing the meaning of a field bumps it.

Alternatives considered:

- Keeping the generator in `gr.State` deep-copies a pipeline and loses it on reload. A module global hides the dependency from tests.
- A single image with both sources cannot tell webcam from upload.
- Selecting through `Gallery.select` toggles is fragile, because a click also opens the preview.
- Defaulting a missing source to `file` would fabricate provenance for T05 webcam sessions.
- Bumping to schema version 2 would need a migration before `show` could open any T05 session.
- A distinct `rerun` source would duplicate `rerun_of` and hide the physical origin.

## Consequences

Default tests cover the controller, record change, and CLI wiring offline with the fake backend. One optional test builds the Blocks object when gradio is installed. Every existing `run_round` caller must pass `source`. T05 sessions still open in `show`, which reports their source as unavailable. Records written by T06 also open in the T05 reader, which ignores the new key. Sketches stay in Gradio's temporary cache for up to an hour after upload. All browser tabs share one session, which suits one local user only.

Unverified until tried on the laptop: Diffusers step progress through `track_tqdm`, an unmirrored frame, a 100 s round without a timeout, the history updating through `.then` after a failed round, and a blank seed reaching the handler as None. A share link for Colab, `show` and `rerun` in the UI, and history across sessions are deferred to the owner.
