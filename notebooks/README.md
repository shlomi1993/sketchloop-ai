# Notebooks

Exploratory research: trying models, preprocessing ideas, and measurements before they become library code.

- Name notebooks `NN-short-topic.ipynb` (for example `01-controlnet-scribble.ipynb`).
- Import from `sketchloop` instead of copying library code. When code is reused, move it into `src/` with tests.
- Clear all outputs before committing. Outputs can embed images, prompts, and local paths that the privacy guard cannot inspect.
- Read inputs from and write results to the ignored `data/`, `runs/`, and `models/` directories, never into `notebooks/`.
- Notebooks are not evidence for a milestone on their own. Record conclusions in `docs/research/` and decisions in an ADR.

## Colab

For GPU work, clone the repository in Colab, run `pip install -e .` plus the notebook's extra packages, and set `HF_TOKEN` through Colab secrets rather than in the notebook. Record the GPU type and package versions in the notebook's first cell, because they affect results and replay.
