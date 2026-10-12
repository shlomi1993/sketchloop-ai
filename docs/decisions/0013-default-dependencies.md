# ADR 0013: Runtime libraries as default dependencies

Status: accepted

## Context

[ADR 0006](0006-first-backend-store-and-ui.md) kept the model and UI libraries in optional extras, so the base install and default checks stayed light. [ADR 0009](0009-diffusers-backend.md) added the `generation` extra and [ADR 0012](0012-gradio-ui-and-input-source.md) the `ui` extra. In practice SketchLoop is run as an application, and a missing extra was the most likely first-run failure. The owner asked for one install that runs everything (R07, R10).

## Decision

- `pip install -e .` installs every runtime library: OpenCV, Rich, Gradio, and the Diffusers stack, with the versions pinned in ADR 0009 and ADR 0012.
- The only extra is `dev`, holding pytest and ruff for developers and CI. A separate `ci` extra was rejected, since CI needs exactly the same tools, and an extra can only add packages, never remove default ones.
- Model weights stay out of the install and come from `scripts/download_models.py`, so default checks remain offline and need no weights.
- `torch`, `diffusers`, and `gradio` are still imported only by the code that needs them, so the fake backend, `show`, and `rerun` start quickly.
- On Linux, CI sets `PIP_EXTRA_INDEX_URL` to the PyTorch CPU index, so it installs the CPU build instead of the default build with about 3 GB of CUDA libraries.

## Consequences

Installs are much larger and slower, about 1 GB on Windows and macOS and more on Linux with a GPU build. CI takes longer, partly offset by the CPU torch index. A Linux user without a GPU can install CPU torch first in the same way. The default checks remain offline and fast, because the heavy imports stay lazy and the real-model test stays behind the `model` marker. This supersedes the optional-extra parts of ADR 0006, ADR 0009, and ADR 0012.
