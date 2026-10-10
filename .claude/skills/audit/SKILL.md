---
name: audit
description: Audit the whole SketchLoop repository, not just a diff, for private or sensitive information that is published or would be published by a push, and for messy code that could be simpler. Use before pushing, before a milestone, or periodically.
argument-hint: "[optional: privacy | simplify | paths to focus on]"
---

# Full repository audit

Focus: $ARGUMENTS, or by default both privacy and simplification across the whole repository

Run the two reviews below in parallel as subagents. Give each the full scope explicitly, because both agents default to reviewing a diff. Then merge their results into one report. Do not modify files.

## 1. Privacy and exposure (`privacy-reviewer`)

Scope is everything that is or would become public:
- **Publishable files:** every file from `git ls-files --cached --others --exclude-standard`, including untracked files that are not ignored. Run the publication guard with the project interpreter.
- **About to be pushed:** commits in `git log @{u}..HEAD`, or all local commits if there is no upstream. Review their diffs and commit messages.
- **History:** `git log --all --name-only --format=` for any sensitive path ever committed, such as anything under `private/`, `.env` files, model weights, PDFs outside `docs/articles/`, captures, or run outputs. Content removed from the tree still exists in history.
- **Ignore coverage:** `private/`, `data/`, `runs/`, `models/`, `tmp/`, and `.env*` stay ignored, and nothing under them is tracked (`git ls-files private data runs models tmp`).
- **Manual reading** for what the regex guard misses: names of the proposal's people, identifiers, affiliations, contact details, absolute local paths, usernames or machine names, credentials, source-text passages, and notebook outputs or embedded images. Published authors in citations are fine, and so are openly licensed PDFs in `docs/articles/`.

Never open `private/`, and never print suspected values. Refer to them by file and line.

## 2. Simplification (`code-reviewer` with the `readability` skill)

Scope is every Python file (`src/`, `scripts/`, `tests/`, `install.py`), and `docs/`, `AGENTS.md`, `CLAUDE.md`, and `.claude/` for duplication and staleness. Look for:
- Dead or unused code, unused parameters, and leftover scaffolding.
- Duplicated logic that one helper could replace, and helpers or abstractions used only once that add indirection.
- Long functions, deep nesting, complex conditions, and clever code that a plain version would express better.
- Code that fights the language or the standard library, such as hand-rolled parsing or manual loops where a builtin fits.
- Names that are not self-explanatory, and style rules from `AGENTS.md` that are broken.
- Docs that repeat a rule stated elsewhere, or that describe files, commands, or behavior that no longer exist.

For each finding, propose the simpler version and estimate the lines saved. Prefer fewer, high-value findings over a long list.

## Report

1. **Blocking before push:** privacy or exposure findings, each with file:line or commit, category, and fix. Say explicitly if history needs rewriting, and never rewrite it without the user's approval.
2. **Simplifications:** ranked by value, each with location, current problem, simpler alternative, and approximate line change.
3. **Clean areas:** one line naming what was checked and found fine.

Offer to apply the simplifications as small separate commits.
