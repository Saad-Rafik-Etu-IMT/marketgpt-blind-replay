# Engineering modernization action plan

Baseline and findings are in [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md).
Actions are sized for a local benchmark; P0 is reserved for verified critical
correctness/security/reliability, not hypothetical external threats.

## Prioritized execution backlog

| ID | Priority / findings | Implementation and exact files | Acceptance / validation | Risk / effort | Status |
|---|---|---|---|---|---|
| A01 | P0 F01 | `backend/src/services/replay_engine.py`: flush new session, initialize snapshot before shared commit, rollback on error. | Failure-injection regression + manual replay and existing initial cash/snapshot tests. | Low/S | COMPLETED |
| A02 | P1 F02 | `routes_studies.py`, `learning/mlp/run_study.py`, `services/study_artifacts.py`: atomic JSON writes, lock status observation, known queued total, record failed launch, preserve completed counters. | Failed launch gives 503 and recorded state, interrupted/failed counts preserved; saved-study E2E. | Medium/M | COMPLETED |
| A03 | P1 F03 | `run_study.py`, `export_comparison.py`: load split/profile once and pass explicitly to each run. | Two-run test verifies identical configuration objects and progress even when second run fails; timeline and competition E2E. | Low/S | COMPLETED |
| A04 | P2 F04/F09 | `study_artifacts.py`, study route and runner: shared run-key construction and overview derivation, small sidecar, retain legacy report fallback. | Old/new overview equality, avoid full report read on sidecar path, preserve full report and catalog shape. | Medium/M | COMPLETED |
| A05 | P2 F05 | `replay_engine.py`, `bar_queries.py`, `broker.py`, `portfolio_service.py`: SQL prefix limit and single-bar ordered lookup. | Window boundaries and missing index test; anti-lookahead, fills, portfolio and full replay invariants. No schema change. | Medium/M | COMPLETED |
| A06 | P1 F06/F09 | `frontend/components/research/CompetitionReplay.tsx`, `lib/research/replay-clock.ts`: shared decision/outcome bounds, terminal polling state, cache identity follows published selected runs. | Clock tests at zero, previous, current, final; signature/terminal tests; existing recorded replay E2E. | Medium/M | COMPLETED |
| A07 | P2 F07 | `frontend/app/studies/page.tsx`: key validation results by study/run base so period never leaks across selections. | Typecheck, build and saved study browser test; no dedicated cross-dataset browser fixture. | Low/S | COMPLETED |
| A08 | P1 F08 | `backend/pyproject.toml`, `tests/test_reference_transformer.py`, `.github/workflows/ci.yml`: declare NumPy, exercise all CPU transformer variants in separate job, add Alembic drift gate. | CPU training/checkpoint and probability checks on all variants; local migration check, full tests; remote job pending. | Medium/M | COMPLETED |
| A09 | P2 F09 | Extract local provider orchestration from study HTTP route into `backend/src/services/local_explanation.py`; share exact-fact guard across both providers. | Accepted reordered evidence; missing, duplicate, changed evidence and provider errors fall back; preserve HTTP fact sheet and labels. | Low/M | COMPLETED |
| A10 | P1 F10 | `replay_engine.py`, orders and API persistence: design a supported single/multiple writer policy, characterize real concurrent use with file-backed SQLite, choose minimal transaction controls. | Deterministic concurrency integration suite, no stale pointer/duplicate fills, explicit rollback. | High/L | DEFERRED |
| A11 | P1 F11 | Public deployment/auth and multi-worker study control only if explicitly required. | Threat model, authority, data isolation, operational topology and deployment-specific tests. | High/L | DEFERRED |
| A12 | P2 F12 | Run online advisory checks; plan lockfile upgrades with compatible versions. | Verified advisories and clean install on supported CI versions. | Medium/L | BLOCKED |
| A13 | P2 F12 | Progressively reduce historical strict-mypy/Black failures. | Baseline comparison, targeted typing and formatting per module. | Medium/L | DEFERRED |
| A14 | P2 F09 | Remaining decomposition of the fact builder, study report schemas and oversized JSX views. | Characterization for old artifacts and all routes; no broad markup movement. | Medium/L | DEFERRED |

A01–A09 are implemented in the review branch; A10–A14 are not represented as
completed features. A10 and A11 are high-risk changes under the mission's
constraints; no unsafe multi-user or production deployment is authorized.

## Dependencies and PR-sized units

| Unit / proposed PR title | Actions | Checklist, tests and acceptance | Dependency / rollback |
|---|---|---|---|
| Atomic replay state and bounded market queries | A01, A05 | Failure injection, date-boundary test, replay/fill/portfolio tests, unchanged migration and public schemas. | Baseline first; revert this commit to restore prior reads/transaction. |
| Durable study lifecycle and lightweight summaries | A02, A03, A04 | Shared atomic publication, failed/interrupted progress, frozen config/profile, legacy compatibility, artifact parity and saved-study E2E. | Baseline first; rollback ignores new optional sidecars, old reports still readable. |
| Recorded replay state boundaries | A06, A07 | Clock and query identity tests; typecheck/lint/build/browser tests; unchanged visual design. | Uses same API response; independent rollback. |
| Optional reference-model quality gate | A08 | Real CPU smoke test of every variant, migration drift check; dependency already in constraints, no lock replacement. | No API or data change; rollback job/test/manifest together. |
| Isolate local explanation providers | A09 | Move orchestration to existing service; exact-fact and fallback tests; existing explanation UI/API contract. | Delivered with A02–A04 in the study boundary commit because both touch the same route. |
| Future dedicated refactors | A10–A14 | Characterization, deployment decisions and dependency research first, separate PRs. | No implementation in this branch. |

These logical units, plus a post-review typing and JSON shape hardening commit,
are delivered as reviewable commits in one
modernization PR rather than multiple simultaneously competing changes to main.
No schema migration, destructive data operation, public deployment or merge.

## Review guardrails

1. Preserve supported behavior and domain rules. New sidecars are derived
   metadata, not inputs for trading decisions.
2. Keep a local single-user operating assumption; do not claim concurrency
   correctness beyond the regression tests.
3. Require full CI on the PR, including the optional CPU job, before merge.
4. Re-test retained legacy studies when changing an artifact contract.
5. Revisit remaining debt only if supported by an actual failing workflow,
   security advisory, test gap, or measured cost.
