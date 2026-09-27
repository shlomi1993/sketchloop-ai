# Research and tooling references

The following reading list is preserved from the source proposal (p. 7, §9), using publication titles and links without personal author names. The proposal's description of these works is summarized in [PROJECT.md](PROJECT.md); the papers have not been independently reviewed during environment setup. Future research notes should distinguish what the proposal says from claims verified in the papers.

| Reference | Role in this project |
| --- | --- |
| [Rethinking sketching: Integrating hand drawings, digital tools, and AI in modern design](https://doi.org/10.3390/designs9050119), Designs 9(5), 119 (2025) | Conceptual foundation: iterative human sketching with AI. Its exact implementation and vehicle case study are not requirements. |
| [High-resolution image synthesis with latent diffusion models](https://doi.org/10.1109/CVPR52688.2022.01042), CVPR, 10684-10695 (2022) | Background for latent-space image synthesis. |
| [Adding conditional control to text-to-image diffusion models](https://doi.org/10.1109/ICCV51070.2023.00355), ICCV, 3836-3847 (2023) | Background for structural conditioning such as ControlNet. |
| [Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a.html), PMLR 139, 8748-8763 (2021) | Background for CLIP and text/image similarity; possible ranking signal. |
| [Diffusers documentation](https://huggingface.co/docs/diffusers/) | Candidate model-integration library; not mandated. |
| [MLflow tracking documentation](https://mlflow.org/docs/latest/ml/tracking/) | Candidate experiment tracking integration; not mandated. |
| [OpenCV documentation](https://docs.opencv.org/) | Candidate camera and image-processing toolkit; not mandated. |
| [Gradio documentation](https://www.gradio.app/docs/) | Candidate research UI toolkit; not mandated. |

Before implementing against a tool, read current official documentation and record the version actually selected. Preserve bibliographic provenance without copying private cover-page details or personal retrieval history.
