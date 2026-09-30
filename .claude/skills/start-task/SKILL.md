---
name: start-task
description: Begin a bounded SketchLoop roadmap task. Use at the start of a work session or when the user asks to pick up the next task.
---

# Start a roadmap task

1. Run `git status --short`. Note unrelated changes and preserve them.
2. Read the "Next action" in `docs/STATUS.md` and the matching row and "Next bounded task" in `docs/ROADMAP.md`. If the user named a different task, use theirs.
3. Read `docs/ARCHITECTURE.md` and `docs/EXPERIMENTS.md` if the task touches interfaces, records, or persistence. Read the ADRs in `docs/decisions/` that apply.
4. State to the user, briefly: the task ID, requirement IDs, the observable outcome, the files you expect to touch, and how you will verify it. Name any open decision from the roadmap that blocks the task, and ask instead of guessing hardware, models, providers, or licenses.
5. Delegate by need, following the flow in `CLAUDE.md`. Use `researcher` for unverified papers, models, or dependencies and `architect` for new or changed interfaces and records. Then use `python-developer` and `qa-engineer` for the smallest complete slice. Implement directly when the change is small.
6. Verify with focused tests, then the full check using the project interpreter (see `CLAUDE.md`). Run `code-reviewer`, fix what it finds, and run `privacy-reviewer` before any commit.
7. Finish with `/handoff`.
