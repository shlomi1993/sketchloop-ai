# ADR 0016: Keep the Gradio page and the folder store

Status: accepted

## Context

R06 requires durable records that can be reopened and rerun, and R07 requires a browser interface for capture, guidance, generation, selection, and history. [ADR 0006](0006-first-backend-store-and-ui.md) chose Gradio and a filesystem store for the first version. [ADR 0012](0012-gradio-ui-and-input-source.md) built the page with Gradio, and [ADR 0014](0014-replaceable-components-and-settings.md) put storage behind the `ExperimentStore` interface. After T07, the owner asked whether a separate JavaScript frontend with a Python backend, or an SQL database, would serve the project better.

Facts:
- The tool is local and single-user, run by one researcher on one machine.
- The page is one Python module, `ui.py`, with its CSS and JavaScript in `src/sketchloop/static/`.
- Starting the webcam on page load depends on clicking Gradio's own controls by their titles, which can break when Gradio is upgraded.
- A run consists mostly of images: the raw sketch, the processed sketch, and the candidates. `session.json` and `round.json` next to them hold the records.
- `show` and `rerun` read those folders directly, with no server.

Assumption: the next tasks are T08 (latency measurement) and T09 (evaluation and demos), and no task before them needs queries across many sessions.

## Decision

Keep the Gradio page and the folder store with JSON records.

Alternatives considered:
- **A JavaScript frontend with a Python API, such as a single-page app over FastAPI.** This gives full control of the page and the webcam, but adds a second codebase with its own build tools and an API contract to keep in sync. That is more to maintain than a local, single-user tool needs now.
- **An SQL database for records.** Images belong in files either way. JSON records beside them stay readable, diffable, and rerunnable without a running server, while a database would add a dependency and a migration path before any query needs it.

Revisit when one of these happens:
- **The UI:** the page grows substantially, for example for continuous suggestions or collaborative drawing, both optional extensions, or Gradio workarounds such as the webcam start keep piling up. Then reconsider a separate frontend.
- **Storage:** T08 or T09 needs queries across many sessions, such as comparing stage timings. Then add SQLite, either as a second `ExperimentStore` or as a searchable index beside the folders, keeping the images in files.

## Consequences

No code or record changes. The page keeps Gradio's limits, including the webcam start that must be rechecked on Gradio upgrades, as STATUS lists under open questions. Analysis across sessions means reading many JSON files until an index exists. Because storage already sits behind `ExperimentStore`, adding SQLite later needs no change to `SketchSession` or the UI. A frontend rewrite would replace only `ui.py` and `static/`, since the page already calls only `AppController`.
