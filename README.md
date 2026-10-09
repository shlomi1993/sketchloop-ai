# SketchLoop AI

Modular Python research infrastructure for iterative design using physical sketches and generative AI.

The intended loop is **draw → capture → preprocess → generate alternatives → inspect and select → revise → repeat**. Experiment history makes configurations, outputs, timings, and relationships between iterations available for research and replay.

## Progress

| Task | What you can run | Status |
| --- | --- | --- |
| T01 | `sketchloop examples/sketch.png --prompt "..."` with fake candidates | ✅ Done |
| T02a | Load PNG or JPEG and preprocess the sketch (`--raw` to skip) | ✅ Done |
| T02b | `sketchloop --camera` captures the sketch from a webcam | ✅ Done |
| T03 | `--backend diffusers` makes real AI candidates with SD 1.5 and ControlNet scribble | ✅ Done |
| T04 | Several rounds in one session | ⏳ Next |
| T05 | Reopen and rerun an earlier session | Planned |
| T06 | A simple window instead of the terminal | Planned |
| T07 | Switch model, camera, or storage through a setting | Planned |
| T08 | Timing per stage | Planned |
| T09 | Full demo and evaluation report | Planned |

Details are in the [roadmap](docs/ROADMAP.md) and [current status](docs/STATUS.md).

## Start here

- Agents: read [AGENTS.md](AGENTS.md), then [current status](docs/STATUS.md). Claude Code loads these through [CLAUDE.md](CLAUDE.md).
- Product scope and acceptance criteria: [project specification](docs/PROJECT.md).
- Design: [architecture](docs/ARCHITECTURE.md) and [experiment contract](docs/EXPERIMENTS.md).
- Work sequence: [roadmap](docs/ROADMAP.md).
- Setup and checks: [development guide](docs/DEVELOPMENT.md).
- Publication rules: [privacy](docs/PRIVACY.md).
- Research context: [references](docs/REFERENCES.md).

## Quick start

After cloning, use Python 3.11 or newer to run the installer from the repository root:

```sh
python3 install.py && . .venv/bin/activate
```

On Windows, run `py -3 install.py` and activate with `.venv\Scripts\Activate.ps1`. The installer creates or reuses `.venv`, installs the development extra, checks dependency compatibility, and runs two quick readiness tests. Installation requires package-index access or cached dependencies.

Run the full checks with `python scripts/check.py`. Run just the tests with `python -m pytest`. After installation, checks run offline. No model, camera, API account, or GPU is needed. All required project context is in the documents above; the private source proposal is deliberately excluded from Git.

The environment stays in `.venv` and uses the shell prompt name `sketchloop-ai`. Running `python3 install.py` alone prepares it but cannot activate it in the calling terminal. Run `. .venv/bin/activate` afterward, or use the combined command above. Check the selected interpreter with `python -c "import sys; print(sys.prefix)"`, which should point to this repository's `.venv`. Rerunning the installer refreshes the prompt name of an existing environment without removing its packages.

## Try it

Run `sketchloop examples/sketch-photo.jpg --prompt "modern chair"` in the activated environment. It keeps the raw photo, saves a cropped, contrast-normalized, 512 px grayscale `sketch.png` and four candidates under `runs/`, and asks you to pick some by number. Add `--raw` to generate from the raw image. Run `sketchloop --camera --prompt "modern chair"` to take the sketch from your webcam instead: a preview window opens, Space captures (saved as `sketch-raw.png`), and Esc cancels. Use `--camera-index 1` for a second camera.
By default the images come from a labeled fake backend (deterministic noise).

## Real model

Generate real candidates with Stable Diffusion 1.5 and the ControlNet scribble model (no GPU needed):

```sh
pip install -e ".[generation]"
python scripts/download_models.py
sketchloop examples/sketch-photo.jpg --prompt "modern chair" --backend diffusers
```

The download script fetches about 6 GB into the ignored `models/` folder and skips files already present. The backend uses CUDA, then Apple MPS, then the CPU. `--mode fast` (4 steps with LCM-LoRA, no negative prompt) is the default on CPU, and `--mode quality` (20 steps) the default on a GPU. Run the real-model test with `python -m pytest -m model`, since default checks skip it.

## Using the project environment with Conda

If your terminal shows `(base)`, leave Conda before activating the project environment:

```sh
conda deactivate
. .venv/bin/activate
```

If you already see both `(.venv)` (or `(sketchloop-ai)`) and `(base)`, deactivate them in reverse order,
then activate only the project environment:

```sh
deactivate
conda deactivate
. .venv/bin/activate
```

To stop Conda from automatically activating `base` in future terminals:

```sh
conda config --set auto_activate_base false
```

This changes Conda's startup preference. Use `conda deactivate` to leave an already active environment.
You can still activate Conda explicitly with `conda activate base` when needed.
