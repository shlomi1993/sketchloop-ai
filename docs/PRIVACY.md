# Privacy and public repository boundary

The private proposal is a source for requirements, not a public artifact. Its personal and administrative details must not appear in code, Markdown, comments, examples, tests, package metadata, GitHub content, or generated documents. The sanitized specification deliberately omits personal names, identifiers, affiliation details, and source metadata. Standard citations of published works, including their authors, are public and belong in the references.

## What belongs in Git

Technical requirements, paraphrased research context, architecture, anonymous/synthetic examples, source code, tests, and public research/tool links. Use portable relative paths. Keep study information separate from reusable software.

## What stays local

The original PDF and copies, extracted text, page renders, participant data, camera captures, prompts containing private content, experimental outputs, model weights, credentials, and machine-specific configuration. Store these under ignored `private/`, `data/`, `runs/`, `models/`, or `tmp/` as appropriate. `.env.example` may contain placeholders only; real `.env` variants are ignored. PDF files are excluded by default. The exception is openly licensed published articles placed directly in `docs/articles/`, after confirming the license allows redistribution, for example under CC BY, and citing them in `REFERENCES.md`.

Do not place the private values themselves in a committed blacklist or test. Synthetic regression inputs should be obviously fictional. The publication guard checks paths, common identifier/contact/absolute-home-path patterns, and credentials without printing matched content. It inspects both current publishable files and staged blob contents. Pattern scanning cannot recognize every personal name or sensitive fact, so manual diff review remains required.

## Before a public push

1. Run `python scripts/check.py` in the activated environment and inspect `git status --short` and the full proposed diff.
2. Review all text and binary additions for source-derived personal information. A passing regex check is not a privacy certification.
3. Confirm private artifacts are untracked. `.gitignore` does not untrack existing files or erase Git history. If private content was committed, stop publication and resolve the history exposure before pushing.
4. Ensure external model/tracking integrations do not upload private sketches or prompts by default. Make the destination and stored data explicit when such integrations are implemented.

The current guard checks the worktree and index, not all historical commits or remote content. Source removal must not break setup or tests. Keep all necessary technical knowledge in the public specification; never ask future agents to recover the private proposal.
