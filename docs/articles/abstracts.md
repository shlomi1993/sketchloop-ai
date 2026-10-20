# Article abstracts and project relevance

Companion to [READING-LIST.md](READING-LIST.md), covering all 15 papers in the same roadmap task (Step) order.

The **abstract summaries are paraphrases of the linked source abstracts**, rather than verbatim reproductions or
full-paper reviews. The takeaways and project uses are interpretations for SketchLoop, mapped to the
[roadmap](../ROADMAP.md).

**Reading approach:** Follow the roadmap's Step order, from design context and sketch conditioning to generation,
evaluation, user interaction, component alternatives, and latency optimization.

## 1. Rethinking Sketching

**Paper:** [Rethinking Sketching: Integrating Hand Drawings, Digital Tools, and AI in Modern Design](https://www.mdpi.com/2411-9660/9/5/119)

**First relevant task:** T01.

**Abstract summary:** The paper proposes an automotive-design method combining traditional and digital drawing,
Stylistic Design Engineering, and generative AI. It describes a repeating process that uses generated alternatives
to support ideation, style exploration, and concept development. The authors report improvements in efficiency and
precision while maintaining the designer's creative role, and discuss applications beyond automotive styling.

**Takeaway:** Hand drawing and AI generation can support structured, designer-led iteration.

**Project use:** Structure manual sketch–generate–select iterations and measure their efficiency.

## 2. ControlNet

**Paper:** [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

**First relevant task:** T02a.

**Abstract summary:** ControlNet introduces additional spatial inputs into an existing image diffusion model while
keeping the pretrained model fixed. A trainable branch, connected through initially zero-valued convolutions, learns
to use controls such as edges, depth, segmentation, or pose. The authors demonstrate different controls individually
and in combination, both with and without text prompts, across different training-data scales.

**Takeaway:** Spatial conditioning steers image structure alongside textual guidance.

**Project use:** Tune sketch preprocessing and record conditioning strength for the current backend.

## 3. It's All About Your Sketch

**Paper:** [It's All About Your Sketch: Democratising Sketch Control in Diffusion Models](https://arxiv.org/abs/2403.07234)

**First relevant task:** T03.

**Abstract summary:** This work examines why existing diffusion systems can misinterpret rough sketches, particularly
drawings made by non-experts. It proposes a method that accounts for sketch abstraction through an adapter, adjusted
timestep sampling, and guidance from a sketch-to-photo retrieval model. The resulting system generates images from
rough drawings without requiring a text prompt at inference, and the authors evaluate its sketch-image correspondence.

**Takeaway:** Sketch conditioning must account for rough drawings and varying abstraction levels.

**Project use:** Test sketch faithfulness across drawing styles and levels of detail.

## 4. Latent Diffusion Models

**Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

**First relevant task:** T03.

**Abstract summary:** The paper addresses diffusion models' high computational cost by performing denoising in a
compressed representation produced by a pretrained autoencoder. Cross-attention supplies conditioning information,
including text and other inputs. The authors report strong results across image generation, inpainting, and
super-resolution, while reducing the computational demands associated with diffusion directly over image pixels.

**Takeaway:** Diffusion in compressed image space reduces computation while supporting flexible conditioning.

**Project use:** Explain the backend's components and assess resolution, memory, and runtime trade-offs.

## 5. LCM-LoRA

**Paper:** [LCM-LoRA: A Universal Stable-Diffusion Acceleration Module](https://arxiv.org/abs/2311.05556)

**First relevant task:** T03.

**Abstract summary:** This report extends latent consistency distillation using low-rank adaptation, reducing the memory
needed to train acceleration modules for several Stable Diffusion variants. The resulting LCM-LoRA parameters can be
combined with compatible fine-tuned models and other LoRAs without retraining each combination. The authors describe
the module as a learned solver that enables image generation with very few inference steps.

**Takeaway:** Learned adapter weights enable few-step generation in compatible Stable Diffusion models.

**Project use:** Explain fast mode and measure its speed, quality, and sketch-faithfulness trade-offs.

## 6. Sketch-to-Architecture

**Paper:** [Sketch-to-Architecture: Generative AI-aided Architectural Design](https://arxiv.org/abs/2403.20186)

**First relevant task:** T03.

**Abstract summary:** This work presents an architectural-design workflow that uses generative models to develop
conceptual floorplans and three-dimensional models from simple sketches. Text descriptions help control architectural
renderings, with the aim of supporting rapid exploration. The authors present the workflow as a demonstration of
generative AI's potential within computer-aided architectural design.

**Takeaway:** Sketches and text provide complementary guidance for architectural generation.

**Project use:** Develop architectural sketch-to-image demonstrations and define their input and guidance conventions.

## 7. CLIP

**Paper:** [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

**First relevant task:** T04.

**Abstract summary:** CLIP learns visual representations by matching images with their associated text across a large
collection of internet image-caption pairs. After training, natural-language descriptions can specify categories
for tasks without additional task-specific training. The authors evaluate transfer across more than 30 vision datasets
and report performance that is competitive with supervised baselines on a range of tasks.

**Takeaway:** Shared image-text representations provide a semantic similarity signal.

**Project use:** Add optional prompt-alignment scoring while keeping design quality and human preference as separate judgments.

## 8. Generative AI and Design Fixation

**Paper:** [The Effects of Generative AI on Design Fixation and Divergent Thinking](https://arxiv.org/abs/2403.11164)

**First relevant task:** T04.

**Abstract summary:** A between-participants experiment with 60 people investigates how AI-generated images affect
visual ideation. In this study, participants using an image generator became more attached to an initial example
and produced fewer, less varied, and less original ideas than the baseline group. Qualitative analysis links the
experience to prompting approaches and how participants responded to the suggested images.

**Takeaway:** AI-generated suggestions can increase fixation and reduce idea diversity.

**Project use:** Evaluate fixation, diversity, and originality alongside speed and similarity.

## 9. Text-to-Image for Architectural Ideation

**Paper:** [Using Text-to-Image Generation for Architectural Design Ideation](https://arxiv.org/abs/2304.10182)

**First relevant task:** T04.

**Reference caveat:** This is the reading list's provisional substitute for
*The Benefits and Challenges of AI Image Generators for Architectural Ideation*.

**Abstract summary:** In a laboratory study, 17 architecture students develop a cultural-center concept with Midjourney,
Stable Diffusion, and DALL-E. Questionnaires and group interviews suggest that image generation can support unexpected
discoveries and imaginative exploration, provided design constraints receive attention. The authors identify practical
challenges and discuss implications for tools and design education.

**Takeaway:** Image generation can support exploration when designers account for task constraints.

**Project use:** Evaluate whether generated alternatives address stated constraints and support useful exploration.

## 10. Design Ideation with AI

**Paper:** [Design Ideation with AI - Sketching, Thinking and Talking with Generative Machine Learning Models](https://doi.org/10.1145/3563657.3596014)
([author copy and abstract](https://www.diva-portal.org/smash/get/diva2%3A1814153/FULLTEXT01.pdf)).

**First relevant task:** T06.

**Abstract summary:** The authors study a workshop in which design practitioners and researchers use GPT-3 while
developing concepts for a design brief. Their analysis identifies practical uses and limitations, the influence of
interaction design on expectations, and how wider assumptions about AI affect creative collaboration. The discussion
considers ways to understand and design such collaboration beyond simply replacing human activities.

**Takeaway:** Interaction design shapes expectations and creative collaboration with AI.

**Project use:** Design prompt entry and feedback that communicate user control and model limitations.

## 11. Baudoux Wizard-of-Oz Study

**Paper:** [Automating Analogical Reasoning: A Wizard of Oz Study on the Benefits and Pitfalls of a Sketch-Based AI Image Generator for Design](https://codesign.berkeley.edu/papers/baudoux-WOZ-DCC/)

**First relevant task:** T06.

**Reference caveat:** This is the reading list's provisional substitute for
*Generative Image AI Using Design Sketches as Input: Opportunities and Challenges*.

**Abstract summary:** Nine architects take part in a four-hour residential-design task using a simulated image-generation
interaction. The researchers examine which visual suggestions participants use, what they extract from those images,
and how they bring the material back into their designs. Participants use images for inspiration and assessment,
reusing both broad formal features and functional details, with a preference for atmospheric images.

**Takeaway:** Designers use suggested images for both inspiration and evaluation.

**Project use:** Study how gallery images influence later sketches and selections using a clearly identified simulation.

## 12. Human-MLLM Sketch Ideation Taxonomy

**Paper:** [A Taxonomy of Human–MLLM Interaction in Early-Stage Sketch-Based Design Ideation](https://arxiv.org/abs/2602.22171)

**First relevant task:** T06.

**Reference caveat:** This CHI 2026 poster is the reading list's provisional substitute for
*Collaborative Sketching: Human-AI Collaboration Modes for Enhancing Designers' Performance and Experience in Sketching*.

**Abstract summary:** A study of 12 participants examines sketch-based ideation with a system using a multimodal large
language model. The authors analyze interaction logs and interviews using four predefined modes that allocate creative
responsibility differently between the person and AI. They observe movement between human-led and AI-led activity,
rather than participants consistently following one mode throughout the task.

**Takeaway:** Human and AI creative roles can shift throughout a design session.

**Project use:** Analyze changes in creative responsibility in future interaction studies.

## 13. SDEdit

**Paper:** [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)

**First relevant task:** T07.

**Abstract summary:** SDEdit uses a pretrained diffusion model to turn user-provided image guidance into a more realistic
output. It first perturbs the input with noise and then denoises it using the model's learned image distribution.
The method avoids application-specific training and inversion, and the authors report favorable human judgments
against GAN-based alternatives for tasks including stroke-guided editing and image compositing.

**Takeaway:** Input noise controls the balance between preserving a sketch and reinterpreting it.

**Project use:** Evaluate sketch preservation against realism when exploring image-to-image backends.

## 14. T2I-Adapter

**Paper:** [T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453)

**First relevant task:** T07.

**Abstract summary:** The paper introduces lightweight adapters that connect external control signals to a pretrained
text-to-image model while leaving that model fixed. Different adapters provide control over visual properties such as
structure and color. The authors demonstrate generation and editing applications, and describe the adapters' ability
to work in combination and generalize across settings.

**Takeaway:** Lightweight adapters add composable visual controls to a frozen generative model.

**Project use:** Compare a conditioning alternative through the existing generator interface.

## 15. Latent Consistency Models

**Paper:** [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)

**First relevant task:** T08.

**Abstract summary:** Latent Consistency Models address slow iterative sampling by learning to predict the solution of
a guided diffusion trajectory directly in compressed image space. Distillation from an existing model enables generation
in a small number of steps. The paper reports few-step image-synthesis results and introduces a method for adapting
these models to custom image datasets.

**Takeaway:** Learning the denoising trajectory enables image generation with fewer sampling steps.

**Project use:** Measure how few-step sampling changes generation time and total round latency.

