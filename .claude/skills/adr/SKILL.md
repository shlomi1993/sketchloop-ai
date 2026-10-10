---
name: adr
description: Create a new SketchLoop architecture decision record from the repository template, numbered and linked. Use when a durable interface, storage, dependency, tooling, or scope decision is made or proposed.
argument-hint: "[short decision title]"
---

# Record a decision

Decision to record: $ARGUMENTS

1. List `docs/decisions/` and take the highest number plus one, as four digits. Name the file `NNNN-short-kebab-title.md`.
2. Copy the structure of `docs/templates/DECISION.md` exactly: `# ADR NNNN: Title`, a `Status:` line, then Context, Decision, and Consequences.
3. Set the status to `proposed` unless the user already approved the decision, in which case use `accepted`.
4. Context: requirement IDs (R01-R10), the problem, constraints, and evidence. Separate facts from assumptions. Link research notes in `docs/research/` if they exist.
5. Decision: the chosen approach, the alternatives considered and why they lost, and pinned versions for any dependency or runtime.
6. Consequences: tradeoffs, compatibility or migration impact, how it was or will be validated, and what remains uncertain.
7. If it supersedes an earlier ADR, say so in the new status line and change the old ADR's status to `superseded by ADR NNNN`.
8. Update the documents the decision affects, such as `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, the open decisions in `docs/ROADMAP.md`, or `docs/DEVELOPMENT.md`, and mention the ADR in `docs/STATUS.md`.
9. Keep it short and factual. No private source text, personal data, or absolute paths.
10. Run the full check with the project interpreter so the Markdown links are verified, and tell the user the ADR number and file.
