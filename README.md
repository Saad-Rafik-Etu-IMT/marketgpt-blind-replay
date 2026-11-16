# MarketGPT Blind Replay

A local, educational replay foundation for a future comparison of a small
Transformer with rule-based players on anonymized historical market replays.
**The anonymizer, AI players, baselines and evaluation runner are not implemented.**
This is simulated execution, not a live broker or evidence of real trading skill.

Extracted from Market Replay Trading Lab. See [provenance](PROVENANCE.md) for the
source commit, complete copy manifest, dependency cuts and remaining TODOs.
No market datasets, source databases or trained weights are included.

## What is retained

- CSV validation/upload and instrument listing.
- Session creation, visible bars, stepping and completion.
- Market, limit and stop orders, protective OCO orders and reduction-only exits.
- Cash/futures accounting, fees, synthetic spread, positions and equity snapshots.
- Drawdown, Sharpe, win rate, expectancy and pre-trade risk limits.
- A minimal Next.js replay screen with orders, portfolio and risk.

## Execution and temporal integrity

The current bar is fully visible. Manual orders submitted at N cannot fill until
N+1 or later; market orders use the next open. Only the replay engine advances time.
The unrestricted `/api/market/bars/{instrument}` endpoint is absent and returns 404.

**Reference price is inside the fill bar's OHLC range. Executed price includes
synthetic spread and may lie outside that range.** Spread is not an observed quote;
commissions are separate cash flows. Limit orders still respect their limit.
No execution-price clamping was introduced during extraction.

Precommitted protection on a market entry can execute on the entry bar,
conservatively stop-first. Protection on limit/stop entries starts on a later bar.
Completion cancels pending orders but does not liquidate open positions.
Backward jumps are rejected; a same-index jump remains a no-op and does not reopen
a completed session. AI players must remain forward-only when implemented.

See [fill rules](docs/fill_logic.md) and [temporal integrity](docs/anti_lookahead.md).
Instrument names/dates are still visible: this is **not yet a blind-session API**.
Futures accounting relies on the `=F` suffix; anonymization must preserve the
accounting metadata, not merely rename symbols.

## Local setup (Windows / PowerShell)

Python >= 3.10; use a current Node.js 20 patch (CI uses Node 20).
Verified here with Python 3.12.13 and Node 20.20.2 for browser tests.
The machine's older Node 20.7.0 builds the app but cannot load this Playwright
runner; update Node or use the isolated command shown below.

```powershell
.\start.ps1 -Install
.\start.ps1
```

The install command creates/uses the repository's own `.venv`, installs backend
runtime constraints and development extras, runs `npm ci`, then exits.
It never uses the source repository's editable installation.

- Frontend: `http://localhost:3000` (also `/replay`)
- Backend/API docs: `http://localhost:8000/docs`
- Fresh database: `backend/data/db/trading_lab.sqlite`, created on first DB request

If those ports are occupied, do not stop unrelated services:

```powershell
.\start.ps1 -BackendPort 8100 -FrontendPort 3100
```

The launcher sets `NEXT_PUBLIC_API_BASE_URL` and `MRTL_FRONTEND_ORIGIN` consistently.
For manually launched processes, set those variables yourself. The two original
localhost/127.0.0.1 port-3000 CORS origins remain allowed by default.
Use `MRTL_DATABASE_URL` to select another SQLite file.

There is no sample-data button. Upload a validated CSV through
`POST /api/market/upload` in the API docs, then reload the frontend.
The test suite generates synthetic data in memory; no CSV fixture is required.
Do not expose either service to untrusted networks; authentication and multi-user
isolation are not provided. Dependency warnings are recorded in provenance.

## Database

The fresh baseline contains exactly seven application tables:

`ohlcv_bars`, `replay_sessions`, `orders`, `fills`, `positions`,
`portfolio_snapshots`, `risk_limits`.

Alembic additionally maintains `alembic_version`. Source database files and old
source revision IDs are not supported. Do not point the target at a source DB.

## Validation

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m alembic upgrade head
..\.venv\Scripts\python.exe -m ruff check .
..\.venv\Scripts\python.exe -m pytest
Set-Location ..\frontend
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install chromium
npm run e2e
```

E2E owns ports 3100/8100, never reuses existing servers, uses an isolated SQLite
file, uploads inline synthetic bars and shuts down its servers afterward.
It checks API docs, removed history access, browser execution, portfolio/risk
and completion. Its ignored database files can be removed after the servers stop.

For the older system Node observed during extraction, without changing global Node:

```powershell
npm exec --yes --package=node@20 -- node node_modules\playwright\cli.js test
```

The backend **85% coverage gate is unchanged**. Extraction validation:
**238 backend tests passed, 94.96% coverage; 4 frontend unit tests and 1 E2E passed.**
The six retained core anti-lookahead tests are AST-identical to the source.
Lock regeneration preserves the declared Python minimum:
`uv pip compile --universal --python-version 3.10 pyproject.toml -o requirements.lock`.

## Layout

```text
backend/
  alembic/          fresh target baseline
  src/
    api/           HTTP adapters
    services/      replay, orders, portfolio, risk, data loading
    core/          deterministic fill/P&L/risk functions
    models/        seven-table ORM
    schemas/       retained request/response contracts
    ai/ data/ eval/ README-only future-work placeholders
  tests/           retained regression tests
frontend/          minimal replay screen and tests
docs/              architecture, temporal integrity and fill rules
```
