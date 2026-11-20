# Laboratory validation evidence

Baseline: clean worktree from `ef5ba78`. Python 3.12, Node 24; existing CI
targets Python 3.11/Node 20. Baseline backend Ruff passed; pytest 982 passed,
7 skipped (optional Torch), 95.45% coverage. Frontend 35 tests passed, types,
lint and production build passed. CPU Torch 2.15.0 was installed in a disposable
local environment for the new registered-Transformer checks.

Commands:

```bash
cd backend
python -m ruff check .
python -m pytest
python -m alembic upgrade head
python -m alembic check
cd ../frontend
npm run typecheck
npm run lint
npm test
npm run build
npm run e2e
```

Focused tests verify bounded paired search, signal/noise controls, nested
fold date safety, immutable development identity, durable queue dispatch,
actual registered broker replay and preserved blind phase, exports, optional
Transformer train/reload/evaluation, and prospective holdout refusal and
one-time evaluation on generated fixture data. The browser workflow trains
a real synthetic candidate and navigates its evaluation, evidence and replay.

## Final aggregate results

| Check | Result |
|---|---|
| Backend Ruff | Passed |
| Full backend pytest (CPU Torch installed) | **1,002 passed**, 94.55% coverage, 12 warnings |
| Optional registered reference encoder and all five legacy variants | Passed |
| Alembic fresh upgrade and drift check | Passed, no new migration |
| Installed dependency consistency | `pip check` passed |
| Frontend typecheck and ESLint | Passed |
| Vitest | 35 passed |
| Next.js production build | Passed |
| Chromium E2E | **10 passed**, including the new full laboratory workflow |

An intermediate E2E run exposed an early-hydration tab click and a stale
legacy test assumption that default training used one seed. The tab now
waits for hydration; the legacy one-run scenario explicitly selects seed
0. The complete browser suite was rerun successfully. The same defaults
and existing legacy functionality remain intact.

No real held-out data, live financial system or external model service was
used; LLM explanations retain the existing local-provider contract.
