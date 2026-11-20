# Engineering modernization implementation report

**Branch:** `feat/engineering-modernization` against main
`f2533792cd501c5b5b8286d1bbe07624d0222755`.
**Delivery:** one reviewable pull request, with logically grouped commits; no
merge, deployment or production mutations. The original dirty worktree was
preserved. Local dependency cache, temporary test databases and generated
research artifacts are not included in Git.

## 1. Implemented changes

| ID | Priority | Description and affected code | Status | Actual validation |
|---|---|---|---|---|
| A01 | P0 | `replay_engine.create_session` now owns the initial transaction; `portfolio_service.initialize_session(commit=False)` flushes its snapshot without an intermediate commit. | COMPLETED | Injected failure after snapshot write/flush; no surviving session; manual replay E2E. |
| A02 | P1 | Study route and worker use shared atomic JSON publication; queued total is computed; launch failure records failed status and returns 503; status checks synchronize with worker registration and preserve done/total. | COMPLETED | Failed launch, interrupted and partial failure tests; saved-study E2E. |
| A03 | P1 | Worker loads one immutable split and benchmark profile and passes both to every `export_comparison.export` invocation. | COMPLETED | Two-run configuration identity test; timeline and existing competition tests. |
| A04 | P2 | `services/study_artifacts.py` centralizes run keys and overview derivation. Per-run `.overview` contains the existing summary projection, with fallbacks for old/absent/stale/invalid-JSON sidecars. Full trace and `.meta` catalog unchanged. | COMPLETED | Same overview with/without sidecar; assert no full report read when sidecar available; damaged sidecar recovery; full saved-study browser flow. |
| A05 | P2 | Shared `services/bar_queries.py:bar_at`; broker and marks use ordered LIMIT 1 OFFSET instead of materializing full timeline. Visible-bars uses SQL LIMIT. | COMPLETED | Window and missing index regression; fill, portfolio, anti-lookahead and replay suites; browser manual and recorded replay. |
| A06 | P1 | `lib/research/replay-clock.ts` shares decision/outcome/market bounds; competition query identity includes selected available runs; stop recurring terminal-state polling. | COMPLETED | Zero/current/previous/final clock conditions, run signature and terminal states (10 frontend unit cases); browser suite. |
| A07 | P2 | Scope study results filtering state to selected run with React key. | COMPLETED | Frontend types/build and saved-study E2E; no dedicated cross-dataset fixture. |
| A08 | P1 | NumPy explicitly declared; optional CPU Transformer CI job runs five variants and verifies checkpoint; Alembic schema drift check included. | COMPLETED | All five actual local CPU variant tests passed; Alembic upgrade/check and `pip check` passed; remote CI not a locally verified outcome. |
| A09 | P2 | Study route delegates language provider orchestration to existing `services/local_explanation.py`; both provider paths require an exact permutation of measured facts. | COMPLETED | Seven provider-selection/fact-integrity tests with fake providers, existing decision/outcome API tests. No real local LLM service was configured. |
| A10 | P1 | Concurrency policy across legitimate parallel replay/order writes / process-local worker control. | DEFERRED | Requires file-backed multi-writer characterization and transaction design. |
| A11 | P1 | Public/remote deployment hardening and auth. | DEFERRED | Not within documented local-only scope; requires deployment threat model. |
| A12 | P2 | Online dependency advisory review and compatible upgrades. | BLOCKED | `npm audit --omit=dev --json` timed out after 20 seconds (exit 124); no advisory result. No lockfile upgrade attempted. |
| A13 | P2 | Historical strict-mypy and formatting backlog. | DEFERRED | Strict mypy improved, but 216 existing/remaining errors; Black remains at 39 files. |
| A14 | P2 | Remaining route fact construction/typed report contracts and large JSX rendering components. | DEFERRED | Substantial characterization still required, especially old trace formats; arbitrary splitting would add churn. |

### Compatibility and engineering benefit

- **Data integrity:** atomic session initialization now matches the existing
  atomic per-bar transaction boundary. Failure no longer persists a session
  without its initial snapshot.
- **Fairness of experiments:** each worker uses one configuration and cost/risk
  profile across the selected models/seeds; no change to values or training
  algorithms. Results remain validation-only.
- **Resource usage:** the observed synthetic E2E MLP run JSON was 1,350,040
  bytes; its derived overview was 280 bytes (about 4,821× fewer artifact bytes
  read for that report in overview polling, excluding filesystem metadata).
  Not an HTTP bandwidth or latency benchmark. Older studies still compute
  the same projection on demand. The ordered lookup returns one ORM row rather
  than the whole session. SQLite OFFSET can still scan preceding indexed rows;
  no constant-time database execution claim.
- **Maintainability:** artifact derivation and run identity shared by writer and
  reader; redundant full-history queries removed; one clock governs reveal
  limits; transport and facts no longer intertwined in HTTP handlers.
- **Safety:** measured explanations can only reorder existing verified facts.
  Same HTTP response keys/labels, not new model predictions. No changes to
  jury vote schema, API routing, financial logic, CSS or database migrations.

## 2. Validation evidence

Commands executed locally against the clean isolated baseline and updated branch.
Runtime Python 3.12.14, Node 24.22.3, optional Torch 2.15.0+cpu. CI uses
Python 3.11 / Node 20. Local cached dependencies were reused; a fresh
network install and remote Actions results have not been established by this
report.

| Check | Baseline | Final |
|---|---:|---:|
| `python -m pytest` (backend, coverage gate ≥85%) | 432 pass / 92.72% | **451 pass / 94.17%** |
| `python -m ruff check .` | Pass | Pass |
| `python -m alembic upgrade head`, `python -m alembic check` with disposable DB | Pass | Pass, no new operations |
| `python -m pip check` in installed environment | — | No broken requirements |
| `npm run typecheck` | Pass | Pass |
| `npm run lint` | Pass | Pass |
| `npm test` | 25 pass | **35 pass** |
| `NEXT_TELEMETRY_DISABLED=1 npm run build` | Pass | Pass |
| `npm run e2e` (Chromium; disposable DB/data) | 9 pass | **9 pass** |
| `python -m mypy` (optional strict) | 251 errors / 26 files | 216 errors / 25 files; still fails |
| `python -m black --check .` (optional) | 39 files need formatting | 39 files need formatting; still fails |
| `git diff --check` | — | Pass |
| `npm audit --omit=dev --json` | — | Timed out; not verified |

Final backend: 451 passed, 12 warnings (same count as baseline), 94.17%
coverage; no skipped reference-model variants in this installed CPU runtime.
Browser suite: nine passes (desktop/phone, light/dark, synthetic manual replay,
browser imports, comparison and persisted study). Final frontend 35 unit
tests, four files. Focused test batches were also executed after transaction,
artifact, clock and provider changes; the post-review focused backend run
passed 44 tests. Final full backend and E2E ran again after type and shape
hardening. The optional strict check did not become a mandatory gate: doing
so now would break established CI for legacy modules. New test/helper files
were annotated and formatted; historical formatting was not mass-rewritten.

## 3. Independent review pass

The implementation was then reviewed against the baseline, not approved just
because tests passed. Findings from the skeptical pass:

1. An initial approach left the commit in the portfolio helper. The final
   form explicitly commits in the replay engine, retaining the helper's
   default behavior for other users, and the regression injects a failure
   after the snapshot has been written instead of only before it.
2. Derived overview metadata could make an otherwise valid study fail to
   load. Added a fallback on invalid content/read error and a JSON object
   boundary check, with regression coverage.
3. Stopping competition polling at terminal state alone could leave a
   cached partial run. The cache identity now incorporates selected published
   run keys; new runs trigger a fresh query.
4. The first draft added optional strict-mypy errors. Added explicit types
   to affected helpers and new tests, including narrow optional import
   boundaries. No broad type-ignore rules added; final mypy error count
   decreased from baseline and Black count did not increase.

Review retained the small SQL and file helpers rather than adding abstract
repositories or a service container. The sidecar imposes one additional small
immutable file per run but avoids parsing much larger trace arrays on polling.
It is optional, not a second source of experimental truth. The replay clock
does not change the underlying recorded artifact or promise server-side blind
security; browser display gating and jury listener isolation remain distinct.

## 4. Remaining risks and next actions

1. **P1:** establish supported writer concurrency and characterize simultaneous
   legitimate state changes; process-local lock is not a distributed scheduler.
   Do not deploy as a multi-tenant service without separate design.
2. **P2:** obtain online supply-chain advisory results and execute compatible
   upgrade/clean-install CI in a dedicated PR. No claim of being vulnerability-free.
3. **P2:** further decouple study route fact assembly and type artifact contracts
   only with backward-compatibility fixtures for earlier report versions.
4. **P2:** progressively resolve strict-mypy and formatting backlog. Convert
   to gated checks after characterization and cleanup rather than silencing
   failures.
5. Validate actual local GGUF/Ollama and data-provider workflows when their
   runtime/model/data are available; the current checks use synthetic data and
   test doubles at the provider boundary.

**Verdict:** the local benchmark is materially more reliable, reproducible,
and testable. The core deterministic simulation, model methodology and
scientific claims were preserved. This is an incremental modernization, not
a declaration of production readiness, perfect static typing, or a guarantee
of investment performance.

---

## Research laboratory implementation (base ef5ba78)

The latest main already contained T1–T7 causal features, dataset manifests,
target registry, NumPy models, experiment orchestration and scientific
evaluation. The research-laboratory branch adds the integrated six-view
cockpit, persistent queue/supervisor, bounded search, corrected nested
walk-forward folds, development identity locks, epoch progress, registered
reference Transformer and MLP replay through the existing broker,
statistical evidence plots, prospective holdout lock/CLI, and report exports.

All 1,002 backend tests and 10 browser tests passed locally, as did the
frontend production build and migration checks. Five-seed positive and
noise controls were actually executed, with results in
[RESULTS_AND_CONCLUSIONS.md](../research/RESULTS_AND_CONCLUSIONS.md).
The synthetic prospective test is fixture-only; no protected real test was
evaluated. The [implementation backlog](IMPLEMENTATION_BACKLOG.md) precisely
identifies completed core scope, deferred extensions and the authorized
final evaluation blocker. See [validation evidence](VALIDATION_REPORT.md).

The original audit's figures and action IDs above are historical; they do
not describe the aggregate latest test count.
