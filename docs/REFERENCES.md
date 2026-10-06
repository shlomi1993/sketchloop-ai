# Research and tooling references

The following reading list is preserved from the source proposal (p. 7, §9), with standard citations of the published works. The proposal's description of these works is summarized in [PROJECT.md](PROJECT.md); the papers have not been independently reviewed during environment setup. Future research notes should distinguish what the proposal says from claims verified in the papers.

| Reference | Role in this project |
| --- | --- |
| [Rethinking sketching: Integrating hand drawings, digital tools, and AI in modern design](https://doi.org/10.3390/designs9050119), Donnici, Galiè, and Frizziero, Designs 9(5), 119 (2025), CC BY 4.0, [local PDF](articles/Rethinking%20Sketching.pdf) | Conceptual foundation: iterative human sketching with AI. Its exact implementation and vehicle case study are not requirements. Summary: [research note](research/sde-sketching-paper.md). |
| [High-resolution image synthesis with latent diffusion models](https://doi.org/10.1109/CVPR52688.2022.01042), Rombach et al., CVPR, 10684-10695 (2022) | Background for latent-space image synthesis. |
| [Adding conditional control to text-to-image diffusion models](https://doi.org/10.1109/ICCV51070.2023.00355), Zhang, Rao, and Agrawala, ICCV, 3836-3847 (2023) | Background for structural conditioning such as ControlNet. |
| [Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a.html), Radford et al., PMLR 139, 8748-8763 (2021) | Background for CLIP and text/image similarity; possible ranking signal. |
| [Diffusers documentation](https://huggingface.co/docs/diffusers/) | Candidate model-integration library; not mandated. |
| [MLflow tracking documentation](https://mlflow.org/docs/latest/ml/tracking/) | Candidate experiment tracking integration; not mandated. |
| [OpenCV documentation](https://docs.opencv.org/) | Candidate camera and image-processing toolkit; not mandated. |
| [Gradio documentation](https://www.gradio.app/docs/) | Candidate research UI toolkit; not mandated. |

## Sources added by research notes

| Reference | Role in this project |
| --- | --- |
| [Latent consistency models](https://arxiv.org/abs/2310.04378), Luo et al. (2023), and [LCM-LoRA](https://arxiv.org/abs/2311.05556), Luo et al. (2023) | Few-step sampling for CPU generation. See [first-model options](research/first-model-options.md). |
| [T2I-Adapter](https://arxiv.org/abs/2302.08453), Mou et al. (2023) | Lightweight sketch conditioning, fallback backend. |
| [SDXS: Real-time one-step latent diffusion models with image conditions](https://arxiv.org/abs/2403.16627), Song et al. (2024) | One-step sketch ControlNet, CPU speed fallback. |
| [One-step image translation with text-to-image models](https://arxiv.org/abs/2403.12036), Parmar et al. (2024) | pix2pix-turbo sketch-to-image, GPU-only reference. |
| [ControlNet v1.1 scribble model card](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble) and [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | Recommended first backend weights and license (CreativeML OpenRAIL-M). |
| [Diffusers reproducibility guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/reusing_seeds) | Seed and determinism limits for replay (R06). |
| [MLflow backend stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) | File backend status and default SQLite store. See [store options](research/experiment-store-options.md). |
| [Streamlit](https://docs.streamlit.io/) and [NiceGUI](https://nicegui.io/documentation) documentation | UI alternatives. See [UI toolkit options](research/ui-toolkit-options.md). |

Before implementing against a tool, read current official documentation and record the version actually selected. Preserve bibliographic provenance without copying private cover-page details or personal retrieval history.
