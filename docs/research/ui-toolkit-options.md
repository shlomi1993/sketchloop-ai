# UI toolkit for the local research interface

## Question and decision informed

Which toolkit should provide the T06 interface (R07): camera preview, prompt and control entry, manual generate trigger, progress and errors, candidate gallery, explicit selection, and next round, running locally on Windows and possibly on Colab? Informs the UI ADR and the `ui` optional extra.

## Sources (read 2026-10-02)

- PyPI JSON metadata: `gradio` 6.29.0 (2026-09-29, Apache-2.0), `streamlit` 1.64.0 (2026-09-15, Apache-2.0), `nicegui` 3.17.1 (2026-09-18, MIT), `fastapi` 0.142.2 (MIT). All Python >=3.10.
- Gradio docs: [Image](https://www.gradio.app/docs/gradio/image), [Gallery](https://www.gradio.app/docs/gradio/gallery), [SelectData](https://www.gradio.app/docs/gradio/selectdata), [Timer](https://www.gradio.app/docs/gradio/timer), [sharing your app](https://www.gradio.app/guides/sharing-your-app).
- Streamlit docs: [st.camera_input](https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input), [main concepts](https://docs.streamlit.io/get-started/fundamentals/main-concepts).
- NiceGUI docs: [interactive_image](https://nicegui.io/documentation/interactive_image) with the linked OpenCV webcam example.
- MDN: [getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

## Findings

- Verified: Gradio `gr.Image(sources=["webcam"])` takes webcam snapshots in the browser, supports streaming and mirroring through `webcam_options`, and returns numpy, PIL, or a file path.
- Verified: `gr.Gallery` accepts paths or images with captions and has a `select` event whose `gr.SelectData` exposes `index` and `value`. `gr.Timer` ticks at a set interval, which can refresh a server-side preview.
- Verified: Gradio runs locally by default with `share=False` except on Colab. Share links expire after one week. Gradio collects developer analytics by default, disabled with `analytics_enabled=False` or `GRADIO_ANALYTICS_ENABLED=False`.
- Verified: browser camera access needs a secure context, and `localhost` and HTTPS both qualify.
- Verified: Streamlit reruns the whole script on each widget interaction. `st.camera_input` is snapshot-only, and there is no built-in selectable gallery.
- Verified: NiceGUI documents a live OpenCV webcam preview through `ui.interactive_image.set_source()`, but gallery and selection are assembled from lower-level elements.
- Verified: dependency weight. Gradio pulls FastAPI, uvicorn, pandas, numpy, pillow, huggingface-hub, and others. Streamlit pulls pandas, pyarrow, altair, and pydeck. NiceGUI pulls FastAPI, uvicorn, and socket.io.
- Inference: Gradio's browser webcam plus a Colab share link would let a laptop camera feed a model running on Colab with no extra networking code. This could settle the remote-execution question simply, but it must be tried.
- Inference: Streamlit's rerun model makes long generation calls and multi-round state (selected candidate, lineage) awkward without careful session state.
- Inference: a plain local page (FastAPI plus hand-written HTML and JavaScript) is the most controllable but means owning front-end code, which goes against the owner's preference for small solutions.

## Options compared

| Criterion | Gradio | Streamlit | NiceGUI | Plain local page |
| --- | --- | --- | --- | --- |
| Camera preview | Browser webcam, or server frames via Timer | Snapshot only | Live OpenCV preview example | Own JavaScript |
| Gallery and selection | Built in (`Gallery.select`) | Manual | Manual | Own code |
| Long generation calls | Event handlers with progress | Script rerun model | Async handlers | Own code |
| Colab | First-class, share links | Needs a tunnel | Needs a tunnel | Needs a tunnel |
| Install weight | Heavy | Heavy | Medium | Light (FastAPI) |
| Lines of UI code | Fewest | Few | Moderate | Most |

## Recommendation and confidence

Use Gradio (pin `gradio==6.29.0`) in an optional `ui` extra for T06, with analytics disabled in code. NiceGUI is the fallback if live OpenCV preview inside Gradio proves clumsy. Confidence: medium-high. Gradio fits the gallery and selection flow and Colab best, and camera preview is the main risk.

## Implementation guidance

- Target module: `sketchloop.ui.gradio_app`, which calls the application service only. No model, store, or Gradio types cross into `domain`, `orchestration`, or `experiments`.
- Camera choice for the architect: either (a) browser snapshot wrapped by a small `capture` adapter that records `source="browser-webcam"` with size and time, or (b) library OpenCV capture with a `gr.Timer` preview at about 2-5 fps. Option (a) also works on Colab. Option (b) keeps capture on the server. Both must produce the same capture record (R01).
- Gallery items carry candidate IDs as captions or in parallel state, and `select` maps `evt.index` to the iteration's candidate list before calling `select_candidates`. Never infer a selection from scores.
- Show unsupported-control feedback from `validate_request` before generation. Show a real-or-fake execution label on every candidate.
- Pitfalls: Gradio's major versions change component APIs, so keep the UI module thin and pinned. Launch on `127.0.0.1` with `share=False` locally, and use share links only by explicit choice.
- Tests: default tests cover the application service with fakes, not the UI. Optionally build the Blocks object in a test without launching it, to catch wiring errors offline.

## Open questions for the owner

- Should capture go through the browser (simplest, Colab-friendly) or through OpenCV on the machine running the app?
- Is a temporary public share link acceptable when running on Colab, given that sketches pass through Gradio's share relay?
