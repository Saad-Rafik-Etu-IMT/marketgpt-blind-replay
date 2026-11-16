"use client";

import { useCallback, useEffect, useState } from "react";

import { CandlestickReplayChart } from "./CandlestickReplayChart";
import { OrderTicket } from "./OrderTicket";
import { PortfolioPanel } from "./PortfolioPanel";
import { ReplayControls } from "./ReplayControls";
import { RiskLimitsPanel } from "./RiskLimitsPanel";
import { RiskPanel } from "./RiskPanel";
import { TradeBlotter } from "./TradeBlotter";
import { useOrders } from "@/hooks/useOrders";
import { usePortfolio } from "@/hooks/usePortfolio";
import { useReplaySession } from "@/hooks/useReplaySession";
import { DEFAULT_INITIAL_CASH } from "@/lib/constants";
import { listInstruments } from "@/lib/api";
import type { CreateSessionRequest, InstrumentSummary } from "@/lib/types";
import { Alert, Button, Field, Input, Panel, Select, Skeleton } from "@/components/ui";

export function MarketReplayCockpit() {
  const { session, bars, start, step, jump, complete, reset, error, connectionLost } =
    useReplaySession();
  const sid = session?.session_id ?? null;
  const { orders, submit, cancel, error: orderError } = useOrders(sid);
  const { snapshot, metrics } = usePortfolio(sid);

  if (!session) {
    return <SessionStartPanel onStart={start} error={error} />;
  }

  return (
    <div className="p-4 space-y-3">
      <Button onClick={reset} variant="secondary">New session / change instrument</Button>
      {session.simulation_version === 1 && (
        <Alert tone="danger">Legacy simulation: execution is disabled. Create a new session.</Alert>
      )}
      {error && <Alert tone="danger">{error}</Alert>}
      {orderError && <Alert tone="danger">{orderError}</Alert>}
      {connectionLost && (
        <Alert tone="danger" title="Connection lost">
          The backend is unreachable. Showing the last known state; retrying automatically.
        </Alert>
      )}
      <h1 className="text-lg font-semibold">
        {session.instrument} · {session.start_date} → {session.end_date}
      </h1>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <main className="space-y-3 lg:col-span-2 overflow-x-auto">
          {bars.length === 0 ? (
            <Skeleton className="h-[420px] w-full" />
          ) : (
            <CandlestickReplayChart bars={bars} position={snapshot?.positions?.[0] ?? null} />
          )}
          <ReplayControls session={session} onStep={step} onJump={jump} onComplete={complete} />
          <TradeBlotter orders={orders} onCancel={cancel} />
        </main>
        <aside className="space-y-3">
          <OrderTicket session={session} onSubmit={submit} />
          <PortfolioPanel snapshot={snapshot} />
          <RiskPanel metrics={metrics} />
          <RiskLimitsPanel sessionId={sid} metrics={metrics} />
        </aside>
      </div>
    </div>
  );
}

function SessionStartPanel({
  onStart,
  error,
}: {
  onStart: (request: CreateSessionRequest) => Promise<void>;
  error: string | null;
}) {
  const [instruments, setInstruments] = useState<InstrumentSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [instrument, setInstrument] = useState("SPY");
  const [startDate, setStartDate] = useState("2023-01-03");
  const [endDate, setEndDate] = useState("2023-12-29");
  const [cash, setCash] = useState(DEFAULT_INITIAL_CASH);
  const [multiplier, setMultiplier] = useState(1);
  const [fee, setFee] = useState(0.01);
  const [spread, setSpread] = useState(1);

  function selectInstrument(row: InstrumentSummary) {
    setInstrument(row.instrument);
    setMultiplier(row.multiplier ?? 1);
    if (row.first_date) setStartDate(row.first_date);
    if (row.last_date) setEndDate(row.last_date);
  }

  const loadInstruments = useCallback(async () => {
    try {
      const rows = await listInstruments();
      setInstruments(rows);
      setLoadError(null);
      if (rows.length > 0) {
        selectInstrument(rows.find((row) => row.instrument === "SPY") ?? rows[0]);
      }
    } catch (e: unknown) {
      setLoadError(e instanceof Error ? e.message : "Failed to load instruments");
    }
  }, []);

  useEffect(() => {
    void loadInstruments();
  }, [loadInstruments]);

  return (
    <Panel title="Start a replay session" className="mx-auto mt-8 w-full max-w-xl" bodyClassName="space-y-3 text-sm">
      {instruments.length === 0 && (
        <Alert tone="info" title="No market data yet">
          No data is bundled. Upload a validated CSV through POST /api/market/upload
          in the backend API docs, then reload this page. Use synthetic data for checks.
        </Alert>
      )}
      <Field label="Instrument">
        {instruments.length > 0 ? (
          <Select value={instrument} aria-label="Instrument" onChange={(e) => {
            const row = instruments.find((r) => r.instrument === e.target.value);
            if (row) selectInstrument(row);
          }}>
            {instruments.map((row) => (
              <option key={row.instrument} value={row.instrument}>
                {row.instrument} · {row.name} ({row.bar_count} bars)
              </option>
            ))}
          </Select>
        ) : (
          <Input value={instrument} aria-label="Instrument symbol" onChange={(e) => setInstrument(e.target.value)} />
        )}
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Start date"><Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} /></Field>
        <Field label="End date"><Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} /></Field>
        <Field label="Initial cash"><Input type="number" value={cash} onChange={(e) => setCash(Number(e.target.value))} /></Field>
        <Field label="Multiplier">
          <Input type="number" min={0.0001} step={1} value={multiplier}
            disabled={instruments.find((r) => r.instrument === instrument)?.multiplier != null}
            onChange={(e) => setMultiplier(Math.max(0.0001, Number(e.target.value)))} />
        </Field>
        <Field label="Fee per unit / side (USD)"><Input aria-label="Fee per unit" type="number" min={0} step="0.01" value={fee} onChange={(e) => setFee(Number(e.target.value))} /></Field>
        <Field label="Synthetic full spread (bps)"><Input aria-label="Synthetic spread" type="number" min={0} max={1000} value={spread} onChange={(e) => setSpread(Number(e.target.value))} /></Field>
      </div>
      <p className="text-xs text-slate-400">
        1 bp = 0.01%. Costs are simulation assumptions, not broker quotes.
        Futures use contract P&amp;L without variation settlement or margin.
        Unknown instruments require a verified multiplier.
      </p>
      {error && <Alert tone="danger">{error}</Alert>}
      {loadError && <Alert tone="warning">Could not load instruments ({loadError}). Check the backend, then reload.</Alert>}
      <Button variant="primary" className="w-full"
        disabled={!instrument || cash <= 0 || fee < 0 || spread < 0 || spread > 1000}
        onClick={() => onStart({
          instrument, start_date: startDate, end_date: endDate, initial_cash: cash,
          multiplier, fee_per_unit: fee, spread_bps: spread,
        })}>
        Start session
      </Button>
    </Panel>
  );
}
