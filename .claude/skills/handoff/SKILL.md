---
name: handoff
description: Close a SketchLoop work session by updating docs/STATUS.md, traceability, and ADRs. Use when a task is finished or the session is ending.
---

# Session handoff

1. Run the full repository check with the project interpreter and keep the actual result.
2. Replace the state summary in `docs/STATUS.md` rather than appending to it. Keep it short and factual. Cover the fields in `docs/templates/SESSION.md`: task and requirement IDs, outcome, decisions (link ADRs), checks and actual results, hardware or model checks omitted and why, open questions, and the exact next action.
3. Distinguish implemented, tested, proposed, and blocked work. Never report a fake-backend or offline test as a camera or model demonstration.
4. If a durable interface, storage, dependency, or scope decision was made, add an ADR from `docs/templates/DECISION.md` with the next free number.
5. Update the roadmap if task completion or the next bounded task changed.
6. Do not include transcripts, private source text, personal data, or absolute local paths.
7. Show the user the `git status --short` summary. Do not commit or push unless asked.
