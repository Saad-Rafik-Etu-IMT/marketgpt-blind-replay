# Extraction provenance

## Origin and scope

- Source repository: `Saad-Rafik-Etu-IMT/market-replay-trading-lab`
- Source commit: `baa6101e6d72a38660efdfe2121c725fa3da0ac1`
- Copy date: **2026-11-16**
- Target: `marketgpt-blind-replay`
- Source was treated read-only. No source history, datasets, local databases,
  installed dependencies, logs or build outputs were copied.
- 116 explicitly selected files were copied before the edits listed below.
  Target paths equal source-relative paths unless a snippet destination is stated.
- Scope decisions and dependency cuts were approved before copying. A later
  approval permits an optional frontend-origin setting for isolated local ports;
  default CORS origins and trading behavior are unchanged.

This is pre-existing replay infrastructure, not the team's AI contribution.
The data pipeline, anonymizer, baselines, Transformer players, evaluation runner
and research page remain future work.

## Database baseline (declared before generation)

Exactly seven application tables:

1. `ohlcv_bars`
2. `replay_sessions`
3. `orders`
4. `fills`
5. `positions`
6. `portfolio_snapshots`
7. `risk_limits`

Alembic also owns `alembic_version`. The target-only revision `0001` was generated
from retained models. Old source revision IDs and unstamped source schemas are
not adopted; never connect this target to a source database.

## Copied files: modified

Each row names a copied file and the reason for its modification.

| File | Modification reason |
| --- | --- |
| `.github/workflows/ci.yml` | Install constrained runtime dependencies; supply CI Python to isolated E2E. |
| `.gitignore` | Exclude all datasets, SQLite files, model weights, checkpoints and logs. |
| `backend/alembic/env.py` | Register only retained ORM modules. |
| `backend/pyproject.toml` | Target identity; remove excluded library packaging, unused runtime/data extras and associated lint exceptions; retain 85% gate. |
| `backend/requirements.lock` | Regenerate runtime closure with universal Python/platform markers from the existing Python 3.10 minimum. |
| `backend/src/api/dependencies.py` | Remove excluded ORM registrations and old-source revision adoption; retain target upgrade locking. |
| `backend/src/api/routes_market.py` | Remove sample seeding and unrestricted full-history retrieval. |
| `backend/src/api/routes_portfolio.py` | Remove Greeks route/imports. |
| `backend/src/api/routes_risk.py` | Remove VaR/scenario routes/imports. |
| `backend/src/config.py` | Add approved optional frontend origin with no change to defaults. |
| `backend/src/main.py` | Register retained routes, set target identity and apply approved optional CORS origin. |
| `backend/src/models/order.py` | Remove option-specific fields; retain underlying discriminator and OCO/reduction fields. |
| `backend/src/models/portfolio.py` | Remove option-specific fields; retain contract key, multiplier and accounting state. |
| `backend/src/schemas/market.py` | Remove seed response and dangling status-document reference. |
| `backend/src/schemas/order.py` | Narrow instrument kind to underlying, explicitly rejecting options; remove option fields/validation. |
| `backend/src/schemas/portfolio.py` | Remove option/Greeks contracts. |
| `backend/src/schemas/risk.py` | Remove VaR response contract. |
| `backend/src/services/broker.py` | Remove option execution dependency; underlying fills, spread, fees and OCO logic preserved. |
| `backend/src/services/data_loader.py` | Receive unchanged DB-only quarantine detector from the excluded seeding service. |
| `backend/src/services/order_service.py` | Replace option contract-key construction with underlying symbol; remove option field transfer. |
| `backend/src/services/portfolio_service.py` | Remove option marks/Greeks; retain underlying accounting and copied bar-query helper; sort imports. |
| `backend/src/services/replay_engine.py` | Remove Desk trigger overlay, redirect quarantine import, sort imports; time transitions unchanged. |
| `backend/src/services/risk_check.py` | Remove option branches and helpers; preserve underlying pre-trade checks. |
| `backend/src/services/risk_service.py` | Remove VaR dependency; retain underlying metrics, limits and auto-pause. |
| `backend/tests/conftest.py` | Register retained ORM modules only. |
| `backend/tests/test_contract_routes.py` | Retain selected route contracts; add unsupported-option rejection and approved CORS regression. |
| `backend/tests/test_financial_integrity.py` | Omit excluded option/attribution cases; redirect retained quarantine regression import. |
| `backend/tests/test_migration_bootstrap.py` | Retain fresh-baseline coverage and verify seven tables plus repeated-startup data preservation. |
| `backend/tests/test_no_lookahead.py` | Omit three excluded overlay cases and their helper; retain all six core tests unchanged. |
| `backend/tests/test_routes_market.py` | Omit seed/old-history cases, add loaded-instrument history 404 test and verify duplicate refusal through visible bars. |
| `docs/anti_lookahead.md` | Document only retained guarantees and explicitly describe unchanged jump behavior and future blind-session limits. |
| `docs/architecture.md` | Narrow module map and document actual retained layering, fresh schema and standalone configuration. |
| `docs/fill_logic.md` | Remove option/audit dependencies; explicitly preserve reference-versus-executed-price spread distinction. |
| `frontend/__tests__/smoke.test.tsx` | Retain four relevant existing component smoke cases. |
| `frontend/app/layout.tsx` | Target title and description. |
| `frontend/app/replay/page.tsx` | Remove excluded navigation and expose retained simulation limitations. |
| `frontend/components/cockpit/CandlestickReplayChart.tsx` | Remove macro and report-annotation overlays/imports; retain existing chart and indicators. |
| `frontend/components/cockpit/MarketReplayCockpit.tsx` | Narrow composition to start/step/complete, chart, orders, portfolio and risk; remove sample loading. |
| `frontend/components/cockpit/OrderTicket.tsx` | Remove option inputs and academy link; preserve underlying order submission. |
| `frontend/components/cockpit/RiskPanel.tsx` | Remove academy link only. |
| `frontend/components/ui/Toast.tsx` | Remove excluded-feature example from comment only. |
| `frontend/components/ui/index.ts` | Remove unused table export and its uncopied dependency. |
| `frontend/e2e/golden-path.spec.ts` | Replace dataset seeding and excluded workflow with inline synthetic replay assertions through completion. |
| `frontend/hooks/useReplaySession.ts` | Remove excluded query invalidations and their documentation. |
| `frontend/lib/api.ts` | Keep only retained API wrappers and shared error handling; remove unrestricted-history wrapper. |
| `frontend/lib/queryKeys.ts` | Keep retained query keys only. |
| `frontend/lib/types.ts` | Keep retained transport contracts and underlying-only discriminator. |
| `frontend/package-lock.json` | Regenerate after package identity and unused chart-package removal. |
| `frontend/package.json` | Target identity; remove unused Recharts dependency. |
| `frontend/playwright.config.ts` | Use isolated ports/DB, target Python and approved origin; never reuse existing servers. |
| `start.ps1` | Use target-local environment, explicit installation errors and install-only mode; wire ports consistently and stop only owned process IDs. |

## Copied files: unchanged

The following 65 files retain source bytes.

| File |
| --- |
| `backend/alembic.ini` |
| `backend/alembic/script.py.mako` |
| `backend/src/__init__.py` |
| `backend/src/api/__init__.py` |
| `backend/src/api/errors.py` |
| `backend/src/api/routes_health.py` |
| `backend/src/api/routes_orders.py` |
| `backend/src/api/routes_replay.py` |
| `backend/src/core/__init__.py` |
| `backend/src/core/fill_logic.py` |
| `backend/src/core/pnl.py` |
| `backend/src/core/risk_limits.py` |
| `backend/src/core/risk_metrics.py` |
| `backend/src/database.py` |
| `backend/src/logging_config.py` |
| `backend/src/models/__init__.py` |
| `backend/src/models/fill.py` |
| `backend/src/models/market.py` |
| `backend/src/models/risk.py` |
| `backend/src/models/session.py` |
| `backend/src/schemas/__init__.py` |
| `backend/src/schemas/replay.py` |
| `backend/src/services/__init__.py` |
| `backend/src/services/data_validator.py` |
| `backend/src/services/instrument_catalog.py` |
| `backend/tests/test_api_smoke.py` |
| `backend/tests/test_broker.py` |
| `backend/tests/test_data_loader.py` |
| `backend/tests/test_data_validator.py` |
| `backend/tests/test_database.py` |
| `backend/tests/test_fill_logic.py` |
| `backend/tests/test_ledger.py` |
| `backend/tests/test_oco.py` |
| `backend/tests/test_pnl.py` |
| `backend/tests/test_portfolio_service.py` |
| `backend/tests/test_replay_engine.py` |
| `backend/tests/test_risk_limits.py` |
| `backend/tests/test_risk_metrics.py` |
| `frontend/.eslintrc.json` |
| `frontend/app/globals.css` |
| `frontend/app/providers.tsx` |
| `frontend/components/cockpit/ChartDrawingLayer.tsx` |
| `frontend/components/cockpit/PortfolioPanel.tsx` |
| `frontend/components/cockpit/ReplayControls.tsx` |
| `frontend/components/cockpit/RiskLimitsPanel.tsx` |
| `frontend/components/cockpit/TradeBlotter.tsx` |
| `frontend/components/ui/Alert.tsx` |
| `frontend/components/ui/Badge.tsx` |
| `frontend/components/ui/Button.tsx` |
| `frontend/components/ui/Card.tsx` |
| `frontend/components/ui/ConfirmDialog.tsx` |
| `frontend/components/ui/Input.tsx` |
| `frontend/components/ui/Select.tsx` |
| `frontend/components/ui/Skeleton.tsx` |
| `frontend/hooks/useOrders.ts` |
| `frontend/hooks/usePortfolio.ts` |
| `frontend/lib/cn.ts` |
| `frontend/lib/constants.ts` |
| `frontend/lib/indicators.ts` |
| `frontend/next-env.d.ts` |
| `frontend/next.config.js` |
| `frontend/postcss.config.js` |
| `frontend/tailwind.config.ts` |
| `frontend/tsconfig.json` |
| `frontend/vitest.config.ts` |

## Partial source extraction and new files

- `backend/src/services/seed_service.py::has_quarantined_wti` copied unchanged
  into `backend/src/services/data_loader.py`; no seeding or data files copied.
- `backend/src/services/derivatives_marks.py::session_bars` copied unchanged
  into `backend/src/services/portfolio_service.py`; no option engine copied.
- New `backend/alembic/versions/0001_initial_replay_schema.py`: generated baseline.
- New `frontend/app/page.tsx`: one-line entry to the retained replay page, not
  a copy of the source preparation homepage.
- New `backend/src/ai/README.md`, `backend/src/data/README.md`,
  `backend/src/eval/README.md`: README-only future-work placeholders, no code.
- New `README.md` and `PROVENANCE.md`: target operation and extraction records.

## Exclusions and test-case omissions

Excluded: `derivlab`, learning/preparation/practice, Desk/alarms/playbooks/standing
orders, macro, journal, pricing/hedging, scenarios, reports/attribution/bias,
advanced analysis views, coaching, command palette and demo page. Excluded their
routes, services, models, schemas, UI, tests and source migration history.
No sample CSVs (including the mislabeled WTI/SPY duplicate), price seeds or source
databases were copied. The existing quarantine detector remains a guard, not
certification of imported data.

Approved mixed-test omissions:

- Anti-lookahead: `test_desk_note_ignores_future_bars`,
  `test_alarm_never_trips_from_future_bars`, `test_playbook_match_ignores_future_bars`.
- Financial integrity: attribution future-benchmark case, American-option case
  and two option parameter cases. Underlying financial assertions remain.
- Contract routes: pricing, hedge, journal, reports, macro and the old full-history
  error contract.
- Market routes: seed endpoint and two old full-history cases. The replacement
  verifies 404 even when the instrument is loaded; the duplicate-upload test now
  observes all reachable bars through a completed replay rather than removed API.
- Migration bootstrap: legacy adoption, empty source-version adoption and old
  revision upgrade cases. Fresh migration coverage is strengthened, not skipped.
- Frontend smoke: Greeks, scenario, hedge, volatility surface and command palette.
- E2E: source seeding, journal and analysis-view steps replaced with synthetic
  upload, temporal/accounting checks and completion.

No retained failing test was skipped or weakened. The six retained anti-lookahead
test functions are AST-identical to the source. No coverage exclusions or trivial
coverage-filler tests were added; the 85% threshold remains unchanged.

## Required TODOs and limitations

- [ ] Instrument listing must be hidden/anonymized in blind sessions by the
  future anonymizer. Current symbols, display names and dates are not blind.
- [ ] Futures accounting relies on the `=F` suffix; the anonymizer must preserve
  accounting metadata, including contract multiplier and futures/cash treatment.
- [ ] AI players must be forward-only. Current backward-jump behavior is unchanged:
  forward jump requests are rejected, backward jumps are also rejected with
  `JumpDirectionError`, same-index jumps are allowed, and completed sessions
  are not reopened. Future AI observations must not bypass sequential stepping.
- [ ] Implement the team's data pipeline, anonymizer, baselines, Transformer
  players, evaluation runner and research page separately.
- [ ] Review inherited dependency deprecation/security notices before any broader
  use. Next.js 14.2.5 emits a known-vulnerability notice; no unrelated upgrade was
  performed. This application is local-only and not a hardened deployment.

Execution behavior is preserved: the pure reference price lies inside OHLC;
synthetic spread may put executed prices outside OHLC. Fees are separate.
Completion does not force liquidation; futures margin, variation settlement and
rolls are not simulated.

## Validation record

- Windows; Python 3.12.13 in the target-only virtual environment.
- Fresh Alembic upgrade and `alembic check`: passed; no model/schema differences.
- Backend: **238 passed, 94.96% coverage**, unchanged **85%** threshold; Ruff passed.
- Frontend: strict typecheck, ESLint, four unit tests and production build passed.
- Browser: one Chromium synthetic-data E2E passed, covering docs, upload,
  unrestricted-history 404, start, pending order, next-bar fill, portfolio/risk
  and completion with pending cancellation/no forced liquidation.
- Launcher: API and replay page both returned 200 on isolated :8100/:3100;
  validation services were stopped without touching the existing :3000 process.
- System Node 20.7.0 failed in Playwright's loader before tests ran. Using
  `npm exec --yes --package=node@20 -- node node_modules\playwright\cli.js test`
  with Node 20.20.2 resolved it without application changes or a global upgrade.
- The missing Chromium executable was installed using the existing Playwright
  tool. Its first CDN reported a certificate-chain error; its normal verified
  fallback succeeded. TLS checks were not disabled.
- Runtime locks include Python-version markers for the unchanged >=3.10 minimum.
  Only Python 3.12 was executed locally; CI continues to specify Python 3.11.

The final excluded-reference scan distinguishes intentional provenance mentions
and SQLite's `journal_mode`/WAL terminology from removed application modules.
Raw substring matches for SQLite storage terminology are not trading-journal
dependencies; those source bytes and tests remain unchanged.

## Final extraction report

123 deliverable files: 116 copied files (51 modified, 65 byte-identical) and seven
new files. All four retained deterministic core modules are byte-identical to
source. No commit or push was made.

```text
marketgpt-blind-replay/
  .github/workflows/ci.yml
  .gitignore
  README.md
  PROVENANCE.md
  start.ps1
  backend/
    pyproject.toml, requirements.lock, alembic.ini
    alembic/
      env.py, script.py.mako
      versions/0001_initial_replay_schema.py
    src/
      __init__.py, main.py, config.py, database.py, logging_config.py
      api/       retained six routers, dependencies, errors
      core/      fill_logic, pnl, risk_limits, risk_metrics
      models/    market, session, order, fill, portfolio, risk
      schemas/   market, replay, order, portfolio, risk
      services/  replay_engine, broker, order_service, portfolio_service,
                 risk_check, risk_service, data_loader, data_validator,
                 instrument_catalog
      ai/README.md
      data/README.md
      eval/README.md
    tests/       conftest.py and 18 test modules
  frontend/
    app/         root/replay pages, layout, providers, styles
    components/  retained cockpit and shared UI
    hooks/       useReplaySession, useOrders, usePortfolio
    lib/         api, types, constants, queryKeys, cn, indicators
    __tests__/smoke.test.tsx
    e2e/golden-path.spec.ts
    package/lock and Next/TS/lint/style/test configuration
  docs/
    architecture.md
    anti_lookahead.md
    fill_logic.md
```

Ignored installed environments/build outputs are not part of this tree.
Validation databases were temporary synthetic stores, not copied market data.

Source-only grep, from the target root:

```powershell
git grep --no-index --exclude-standard -I -n -i -E 'derivlab|learning|desk|macro|journal|pricing' -- backend/src backend/tests backend/alembic frontend/app frontend/components frontend/hooks frontend/lib frontend/e2e frontend/playwright.config.ts
```

Result: **nine infrastructure-only matches**: eight SQLite WAL/`journal_mode`
mentions in the unchanged database module/test and one Playwright `Desktop Chrome`
device name. There are **no excluded application-feature references**. Provenance
intentionally records removed module names and is not executable code.

```powershell
git grep --no-index --exclude-standard -I -n -i -E '(from|import|require|/api/).*(derivlab|learning|desk|macro|journal|pricing)' -- backend/src backend/tests backend/alembic frontend/app frontend/components frontend/hooks frontend/lib frontend/e2e frontend/playwright.config.ts
```

Result: **0 matches**, normal `git grep` exit code **1** (not a command failure).

Source verification after implementation:

- `git status --porcelain=v1`: **empty / clean**.
- `git rev-parse HEAD`: **baa6101e6d72a38660efdfe2121c725fa3da0ac1**, unchanged.
- Existing process on port 3000 was untouched; validation ports 3100/8100 released.

Suggested commit message: `Extract minimal replay benchmark foundation`.

## AI-assisted course work (after extraction)

The teacher authorized AI assistance for the course work; the team recorded this
on 2026-11-20. Files below were written by an AI assistant (Claude) and must be
reviewed by a team member before merge.

| Tranche | Files | Status |
| --- | --- | --- |
| T1 feature catalogue (2026-11-20) | `backend/src/features/` (all files), `backend/tests/feature_fixtures.py`, `backend/tests/test_feature_registry.py`, `backend/tests/test_feature_returns.py`, `backend/tests/test_feature_point_in_time.py`, `backend/tests/test_feature_baseline_parity.py`, `docs/features.md`; one README status row and link | Added; no existing code modified |
