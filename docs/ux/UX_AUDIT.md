# Research cockpit UX audit

Baseline: main at `c6dab0d` (merged research laboratory). Inspection covered the Next.js cockpit, registered experiment API, replay artifacts and the existing browser test.

| Finding | Evidence | Impact | Resolution |
|---|---|---|---|
| UX-01 Duplicate navigation | AppShell and laboratory local tabs | Two competing paths and lost context | One six-stage research navigation; legacy tools remain available |
| UX-02 Raw identifiers | Run/feature/preset selectors and evidence | Difficult to explain to a novice | Human labels with exact identities retained in details |
| UX-03 Selection mismatch | Evaluation indexed independent of run | Could display evidence for the wrong checkpoint family | Stable keys, seed-aware matching, explicit missing evidence |
| UX-04 Dense replay | Current decision below four equal plots | Critical event difficult to see | Decision-first hierarchy, candles, equity, secondary diagnostics |
| UX-05 Form overload | Single training configuration form | Protocol and budget difficult to understand | Five-step wizard with estimate invalidation on edits |
| UX-06 Misleading positive control preview | Preview used synthetic noise for signal source | Incorrect data evidence | Use the same source loader as training; test both controls |
| UX-07 Ambiguous replay artifact | First file chosen without asset selection | Could replay unintended asset | Backward-compatible optional asset selection |
| UX-08 No guided narrative | Standalone technical screens | Difficult academic demonstration | Eight-stage presentation, selected evidence only |

Preserve the benchmark, independent baselines, chronological validation, original manual/competition replay, imported studies, lock and holdout process, exports and actual worker. Do not infer scientific conclusions from UI polish.
