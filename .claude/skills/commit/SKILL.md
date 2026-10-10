---
name: commit
description: Safely commit SketchLoop changes after running checks and a privacy review, with a message in the repository style.
disable-model-invocation: true
argument-hint: "[optional message hint or files]"
---

# Commit changes

Hint from the user: $ARGUMENTS

1. Run `git status --short` and `git diff`, plus `git diff --cached` if anything is staged. Decide which files belong in this commit. Leave unrelated changes out and tell the user about them.
2. Add or update docstrings for the functions, methods, and classes added or changed in `src/` and `scripts/`, following the Docstrings section of `AGENTS.md`. Skip tests. Never add module docstrings.
3. Run the full check with the project interpreter, `.venv/Scripts/python.exe scripts/check.py` on Windows or `.venv/bin/python scripts/check.py` elsewhere. If it fails, stop and report. Do not commit around a failure.
4. Stage the chosen files by name. Avoid `git add -A` unless every change belongs in the commit.
5. Delegate a review of the staged diff to the `privacy-reviewer` subagent. If it reports findings, stop and show them to the user.
6. Read `git log --oneline -10` and match its style. Write the message as one imperative sentence (no trailing period, no body), naming the task ID if there is one.
7. Do not add co-author or attribution trailers. Show the proposed message to the user and wait for approval before committing.
8. Commit with a heredoc or message file so formatting survives. Never use `--no-verify`, never amend an existing commit, and never push unless the user explicitly asks.
9. Show the resulting `git log --oneline -1` and a one-line summary.
