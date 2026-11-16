# Architecture

```text
Next.js shared research shell
  -> Experiments / Inspector / Evaluation
     -> research provider / validated in-memory artifacts or explicit demo
     -> read-only operator instrument-summary adapter
  -> Replay
     -> cockpit / hooks / typed API
  -> FastAPI routes
  -> services
  -> deterministic core + ORM
  -> SQLite
```

The target retains the source's synchronous SQLAlchemy, polling-based React Query
workflow and application factory. There are no AI implementations yet.

The first [research interface milestone](research-interface.md) adds no backend
routes or tables. Its proposed versioned contract is independent of future
exporters. Imported artifacts never enter replay observations; explanations are
saved text or explicitly synthetic examples, not live generation.

| Layer | Responsibility |
| --- | --- |
| `backend/src/api` | HTTP validation, error mapping and service delegation |
| `backend/src/services` | Replay orchestration, execution, persistence and risk |
| `backend/src/core` | Deterministic fill rules, P&L and risk calculations; no I/O |
| `backend/src/models` | SQLAlchemy models backed by the fresh Alembic baseline |
| `backend/src/schemas` | Pydantic request/response contracts |
| `frontend/hooks` | Session queries, mutations and derived-query invalidation |
| `frontend/lib/api.ts` | Typed transport and API error handling |

Existing source exceptions were preserved rather than refactored: market listing
queries the ORM in its router; fill core consumes the market schema; services use
lazy imports to resolve orchestration cycles. No new business logic was added to
routers by extraction.

## Data flow

1. Upload validates an in-memory CSV and persists all bars or refuses the load.
2. Session creation validates available bars and creates the initial snapshot.
3. Order submission uses visible prices and risk limits, then queues the order.
4. A step advances one bar, processes eligible orders and updates the portfolio
   within the existing per-bar transaction.
5. The risk service can auto-pause on drawdown; completion cancels pending orders.
6. Frontend mutations invalidate visible bars, orders, portfolio and risk queries.

Only the replay engine advances `current_bar_index`. The browser chart receives
session-visible bars, never the unrestricted history of an instrument. The
full-history endpoint is not registered.

Reference fill prices lie inside OHLC; synthetic spread can move executed prices
outside it. Read [fill logic](fill_logic.md) before interpreting results.

## Database

Seven application tables: `ohlcv_bars`, `replay_sessions`, `orders`, `fills`,
`positions`, `portfolio_snapshots`, `risk_limits`; plus Alembic's version table.
The new initial revision supports fresh target databases, not source DB upgrades.

## Standalone operation

Defaults remain frontend :3000 and backend :8000. The optional
`MRTL_FRONTEND_ORIGIN` setting adds an explicitly selected frontend origin while
preserving the original localhost/127.0.0.1 :3000 origins.
The PowerShell launcher configures it alongside `NEXT_PUBLIC_API_BASE_URL`.
E2E uses :3100/:8100 and refuses to reuse an existing server.

Instrument listing remains an operator convenience, not an anonymized observation
interface. Future work and extraction boundaries are tracked in
[provenance](../PROVENANCE.md).
