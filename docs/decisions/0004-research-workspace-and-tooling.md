# ADR 0004: Research workspace, lint, and citation policy

Status: accepted

## Context

The project is an M.Sc. final project that ends in a written report and should support later lab research (R09, R10). The repository had strong implementation scaffolding but no place for literature notes, exploratory notebooks, or the report outline. Code style rules in `AGENTS.md` were not machine-checked. The privacy policy also removed author names from citations, which is unusual for academic work since published authors are public. The development machine has no GPU.

## Decision

- Add `docs/research/` for research notes, `docs/THESIS.md` for a report outline mapped to repository evidence, and `notebooks/` for exploratory work with outputs cleared before commit.
- Pin `ruff==0.16.9` in the `dev` extra and run `ruff check` in `scripts/check.py` with rules `E`, `F`, `W`, and `ANN` (except `ANN401`) and a 130-character hard limit. The 120-character target and the blank-line, import-block, and naming rules stay review-enforced, because Ruff's isort cannot express the project's import blocks. The formatter is not enforced.
- Allow standard citations with published authors in `docs/REFERENCES.md`. The proposal's own personal and administrative details stay excluded.
- Allow openly licensed articles as PDFs directly under `docs/articles/`, in both `.gitignore` and the publication guard, which skips content scanning for them. Each article gets a concise note in `docs/research/` so agents read the note instead of the full PDF.
- Remove the pull request template, because the project has a single developer.
- Record the known hardware in the roadmap and architecture, and keep generation usable against a remote runtime such as Colab.

## Consequences

Checks now require the `dev` extra to include Ruff, which the installer already installs. Research and report work have a defined home, and every reported number should trace to a saved run or script. The citation change makes references thesis-ready without weakening protection of personal data. Model choice (T03) must account for CPU-only local execution or a remote GPU.
