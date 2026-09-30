---
name: bisect
description: Understand a reported SketchLoop issue and find the commit that introduced it, using reproduction, git history search, blame, and bisect. Use when something that used to work is broken, or to learn when and why a line or behavior changed.
argument-hint: "[issue description, failing test, or file:line]"
---

# Find the commit that caused an issue

Issue: $ARGUMENTS

## 1. Understand the issue

Restate the symptom, the expected behavior, and where it shows up (test, command, output, or file:line). Read the code involved. If the description is too vague to reproduce, ask one focused question before continuing.

## 2. Reproduce it

Write the smallest check that passes on good code and fails on bad code. Prefer an existing failing test (`python -m pytest tests/test_x.py::test_name`). Otherwise write a short script under the ignored `tmp/` directory that exits 0 when good and 1 when bad. Confirm it fails on the current commit. If it does not fail, report that the issue does not reproduce and stop.

## 3. Narrow the candidates

- `git log --oneline -- <paths>` for commits that touched the relevant files.
- `git log -S "<text>"` or `git log -G "<regex>"` for when a symbol or line appeared or disappeared.
- `git log -L <start>,<end>:<file>` or `git log -L :<function>:<file>` for the history of a specific range or function.
- `git blame -w -C <file>` on the suspect lines, then `git show <commit>` to read the change in context.

Find a known-good commit, where the check passes, and a known-bad one. If blame alone clearly identifies the culprit, skip to step 5.

## 4. Bisect when the range has more than a few commits

Run bisect in a separate worktree so the user's working tree is untouched:

```sh
git worktree add ../sketchloop-bisect <bad-commit>
cd ../sketchloop-bisect
git bisect start <bad-commit> <good-commit>
git bisect run <reproduction command>
git bisect reset
cd - && git worktree remove ../sketchloop-bisect
```

Keep the reproduction script outside the worktree or pass it by absolute path, so every bisect step uses the same check. Exit code 125 skips commits that cannot be tested. Always reset and remove the worktree, even after a failure.

## 5. Confirm and report

Check the result. The check should fail on the culprit and pass on its parent. Read the culprit diff and explain in a few sentences which change causes the issue and why. Report the commit hash and subject, the evidence (reproduction command and results on the culprit and its parent), your confidence, and a suggested fix. Do not change code unless the user asks. If they do, hand the reproduction to the `python-developer` and turn it into a regression test with the `qa-engineer`.
