# Claude Code entry point

`AGENTS.md` is the single, tool-neutral working agreement for every coding agent. This file only imports it and adds Claude Code specifics. Put shared rules in `AGENTS.md`, not here.

@AGENTS.md

## Always-loaded context

The imports below cover the first two "Start every session" steps. Read `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, `docs/ROADMAP.md`, and relevant ADRs on demand.

@docs/STATUS.md
@docs/PROJECT.md

## Claude Code specifics

- Run checks with the project interpreter so they work on every OS: `.venv/Scripts/python.exe scripts/check.py` on Windows, `.venv/bin/python scripts/check.py` elsewhere. On Windows, `python3` may be a Microsoft Store stub, and `make` is often missing.
- Project skills: `/start-task` begins a bounded roadmap task, `/handoff` closes a session by updating `docs/STATUS.md`, `/adr` records a decision, `/commit` checks, privacy-reviews, and commits (user-invoked only), `/bisect` finds the commit that introduced an issue, `/readability` reviews names and clarity (preloaded by `code-reviewer`), and `/audit` reviews the whole repository for exposure risks and simplifications.
- Before committing, delegate a review of the staged diff to the `privacy-reviewer` subagent in addition to running the checks. It supplements the manual diff review in `docs/PRIVACY.md` and does not replace it.

## Subagent team

Project subagents live in `.claude/agents/`. The main session coordinates them and stays responsible for the result. Relay their findings to the user, since subagent reports are not shown directly.

| Agent | Use it to | Edits |
| --- | --- | --- |
| `architect` | Design contracts, boundaries, failure types, and records, and draft ADRs | `docs/` only |
| `researcher` | Read papers and official docs, compare models and dependencies, and write implementation guidance | `docs/research/`, `docs/REFERENCES.md` |
| `python-developer` | Implement a planned slice test-first under `src/` | Code and tests |
| `qa-engineer` | Plan tests against acceptance criteria, write tests, and diagnose failures | `tests/`, pytest config |
| `code-reviewer` | Review a diff for correctness, invariants, tests, and style | None |
| `privacy-reviewer` | Check a diff for private or source-derived content | None |

Typical flow for a roadmap task: `researcher` when a paper, model, or dependency is involved, then `architect` when an interface or record changes, then `python-developer` and `qa-engineer` (they can work in parallel from the same plan), then `code-reviewer` and `privacy-reviewer`, then `/handoff`. Skip steps a small change does not need. Give each subagent the task ID, requirement IDs, and paths to any plan or research note, because it starts without this conversation's context.
- `.claude/settings.json` is shared. It allows all shell commands and edits anywhere in the repository, including `.claude/` settings. It asks before `git push`, `merge`, `rebase`, and `pull` (which merges or rebases). It denies force pushes, deletes that target the root, home, parent, or absolute paths, privilege escalation, disk tools, and reading private local material. Being allowed is not being asked: still commit, install packages, or discard work only when the task or user calls for it. Put personal overrides in `.claude/settings.local.json`, which is ignored.
- For papers in `docs/articles/`, read the matching note in `docs/research/` first, for example `sde-sketching-paper.md` for the foundational paper. Open a PDF only for a detail the note lacks, and add what you learn to the note.
- Never store project knowledge only in Claude memory. Durable facts belong in `docs/` so every agent and human sees them.
