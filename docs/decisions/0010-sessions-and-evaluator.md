# ADR 0010: Sessions, round folders, and the evaluator protocol

Status: accepted

## Context

T04 (R04, R05) runs several manually triggered rounds in one command. A real model takes minutes to load on CPU, so it must stay loaded between rounds. Each round needs its own files and a link to the previous round, and candidate scoring must stay separate from generation and selection so CLIP can be added later.

## Decision

`sketchloop.session.SketchSession` holds one injected generator and an injected evaluator for the whole session. `run_round` preprocesses, generates, scores, builds an `Iteration` whose `parent_id` is the previous round's iteration, and writes files under `<runs-dir>/<YYYYMMDD-HHMMSS-suffix>/round-<n>/` (raw sketch, `sketch.png`, `candidates/`). `record_selection` accepts one selection for the latest round. Printing and prompting stay in `cli.py`.

`sketchloop.evaluation.Evaluator.evaluate(candidates, payloads, request)` returns a score by candidate ID and never changes candidates. `NoOpEvaluator` returns no scores, and the CLI shows a score column only when scores exist. Every round links to the previous one, whether or not the person selected anything. Feeding a selected image back as the next sketch is not part of this decision.

## Consequences

The session keeps its history in memory only. T05 adds JSON records to the same folder layout. Rounds keep their own copy of the raw sketch even when it did not change, which costs disk space but keeps each round self-contained.
