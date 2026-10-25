---
name: s
description: Answer a question in the shortest complete form, with no preamble, extras, or follow-up offers.
disable-model-invocation: true
argument-hint: "<question>"
---

# Short answer

Answer this: $ARGUMENTS

- Lead with the answer itself. One to three sentences, or a short list only when the answer is a set of items.
- Be direct and coherent: no preamble, no restating the question, no hedging, no caveats unless one changes the answer.
- Include only what the question needs. Skip background, alternatives, and offers of more detail.
- Write any file you mention as a clickable Markdown link with a path relative to the repository root.
- If the question cannot be answered without a fact you lack, check it in the code first, and ask only when it is the owner's decision.
