"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api, pct } from "./StudyWidgets";
import { LinePlot, plotColors, points, type Series } from "./ScientificPlot";
import { SensitivityPlot } from "./ResearchFigures";
import { studyLabel, type StudySummary } from "@/lib/research/study-summary";

type Portfolio = {
  bar_index: number;
  equity: number;
  cash: number;
  total_fees: number;
  total_realized_pnl: number;
  total_unrealized_pnl: number;
  positions: { quantity: number }[];
};
type Event = {
  decision: {
    decision_bar_index: number;
    observation?: { log_returns: number[]; bar_offsets: number[] };
    target: string;
    probabilities: { long: number; flat: number } | null;
    action: string;
    order_id: string | null;
    order_side: string | null;
    order_quantity: number | null;
    pre_trade_rejection: string | null;
    execution_context: {
      cash: number;
      position_quantity: number;
      reference_close: number;
    };
    execution_policy_version: string;
    risk_profile_id: string;
    risk_profile_sha256: string;
  };
  execution: {
    status: string;
    fill_bar: number | null;
    fill_price: number | null;
    fill_quantity: number | null;
    fee: number | null;
    rejection_reason: string | null;
  } | null;
  portfolio: Portfolio;
};
type Player = {
  name: string;
  run: string | null;
  sourceRun: string;
  playerIndex: number;
  checkpoint: string | null;
  macroF1: number | null;
  initial: Portfolio;
  events: Event[];
  occlusion?: number[][];
};
type Competition = {
  firstDecision: number;
  market: {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    adjusted_close: number;
  }[];
  profile: {
    profile_id: string;
    initial_cash: number;
    spread_bps: number;
    fee_per_unit: number;
    cash_reserve_fraction: number | null;
    risk_limits: Record<string, number>;
  };
  profileHash: string;
  accounting: string;
  dataset: string;
  origin: string;
  players: Player[];
  missing: string[];
};
type Run = {
  key: string;
  model: string;
  variant: string;
  seed: number;
  hasTrace?: boolean;
};
type Overview = {
  config: { source: string; seeds: number[]; variants: string[] };
  runs: Run[];
  status: { state: string; message: string };
};
const money = (value: number) =>
  value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
const actionText: Record<string, string> = {
  submit_buy: "Submit purchase",
  submit_sell: "Submit sale",
  none_already_long: "Already invested",
  none_already_flat: "Already in cash",
  none_pending_order: "Await pending order",
  none_zero_quantity: "Insufficient size",
  pre_trade_rejected: "Rejected by checks",
};
const targetText = (target: string) =>
  target === "long" ? "Invest" : "Stay in cash";
const playerText = (name: string) =>
  name === "buy_and_hold"
    ? "Buy & Hold"
    : name.startsWith("ma_crossover")
      ? "Moving Average 20/50"
      : name;

export function CompetitionReplay() {
  const params = useSearchParams();
  const router = useRouter();
  const study = params.get("study") || "";
  const seed = params.has("seed") ? Number(params.get("seed")) : null;
  const variant = params.get("variant") || "reference";
  const [selectionOpen, setSelectionOpen] = useState(false);
  function change(values: Record<string, string | null>) {
    const q = new URLSearchParams(params.toString());
    for (const [key, value] of Object.entries(values)) {
      if (value === null) q.delete(key);
      else q.set(key, value);
    }
    router.replace("/replay?" + q.toString(), { scroll: false });
  }
  const inventory = useQuery({
    queryKey: ["competition-inventory"],
    queryFn: () => api<{ studies: StudySummary[] }>(""),
    refetchInterval: 3500,
  });
  const latest = inventory.data?.studies.find(
    (s) => s.state === "complete" && s.hasTrace,
  );
  useEffect(() => {
    if (!params.has("study") && latest)
      router.replace(`/replay?study=${latest.id}`, { scroll: false });
  }, [params, latest, router]);
  const overview = useQuery({
    queryKey: ["competition-study", study],
    queryFn: () => api<Overview>(`/${study}`),
    enabled: !!study,
    refetchInterval: (q) =>
      ["complete", "failed", "interrupted"].includes(
        q.state.data?.status.state ?? "",
      )
        ? false
        : 2000,
  });
  const runs = overview.data?.runs.filter((r) => r.hasTrace) ?? [];
  const seeds = Array.from(new Set(runs.map((r) => r.seed))).sort(
    (a, b) => a - b,
  );
  const selectedSeed =
    seed !== null && seeds.includes(seed)
      ? seed
      : (runs.find((r) => r.key === params.get("run"))?.seed ?? seeds[0]);
  const variants = Array.from(
    new Set([
      "reference",
      ...runs
        .filter((r) => r.seed === selectedSeed && r.model === "Transformer")
        .map((r) => r.variant),
    ]),
  );
  const selectedVariant = variants.includes(variant) ? variant : "reference";
  const replay = useQuery({
    queryKey: ["competition", study, selectedSeed, selectedVariant],
    queryFn: () =>
      api<Competition>(
        `/${study}/competition?seed=${selectedSeed}&variant=${selectedVariant}`,
      ),
    enabled: !!study && selectedSeed !== undefined && runs.length > 0,
    refetchInterval: overview.data?.status.state === "complete" ? false : 3500,
  });
  const showSelection = selectionOpen || !replay.data;
  const studySummary = inventory.data?.studies.find((s) => s.id === study);
  return (
    <div className="research-stack">
      <section className="bench-panel competition-selection">
        <div className="workspace-controls replay-selection-summary">
          <div>
            <h2>AI competition replay</h2>
            {!study && (
              <p className="muted">
                Replay the recorded decisions, orders and portfolio snapshots
                from a saved validation study.
              </p>
            )}
            {replay.data && (
              <p className="muted compact-status">
                {studySummary
                  ? studyLabel(studySummary)
                  : overview.data?.config.source}{" "}
                · seed {selectedSeed}
                {variants.length > 1 ? ` · ${selectedVariant}` : ""}
              </p>
            )}
          </div>
          <div className="workspace-actions">
            {replay.data && (
              <button
                className="bench-button"
                aria-expanded={showSelection}
                aria-controls="replay-selection-fields"
                onClick={() => setSelectionOpen(!selectionOpen)}
              >
                {showSelection ? "Done" : "Change study"}
              </button>
            )}
            <Link
              className={`bench-button ${study ? "" : "primary-action"}`}
              href={
                study
                  ? `/studies?study=${study}&seed=${selectedSeed ?? 0}&variant=${selectedVariant}`
                  : "/studies?new=1"
              }
            >
              {study ? "Open study" : "Train competing models"}{" "}
              <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
        <div
          id="replay-selection-fields"
          className="workspace-controls replay-selection-fields"
          hidden={!showSelection}
        >
          <label className="study-field">
            Saved study
            <select
              value={study}
              onChange={(e) =>
                change({
                  study: e.target.value,
                  run: null,
                  seed: null,
                  variant: null,
                  case: null,
                })
              }
            >
              <option value="">Select a study</option>
              {inventory.data?.studies.map((s) => (
                <option key={s.id} value={s.id}>
                  {studyLabel(s)}
                </option>
              ))}
            </select>
          </label>
          {seeds.length > 0 && (
            <label className="study-field short-field">
              Seed
              <select
                value={selectedSeed}
                onChange={(e) =>
                  change({
                    seed: e.target.value,
                    run: null,
                    variant: null,
                    case: null,
                  })
                }
              >
                {seeds.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </label>
          )}
          {variants.length > 1 && (
            <label className="study-field short-field">
              Transformer variant
              <select
                value={selectedVariant}
                onChange={(e) =>
                  change({ variant: e.target.value, run: null, case: null })
                }
              >
                {variants.map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </label>
          )}
        </div>
        {(inventory.error || overview.error || replay.error) && (
          <p role="alert" className="bench-error">
            {String(inventory.error || overview.error || replay.error)}
          </p>
        )}
        {overview.data && !replay.data && (
          <p role="status" className="muted compact-status">
            {overview.data.config.source} · {overview.data.status.message}
            {!runs.length &&
              " · Engine traces are available for new studies as runs complete."}
          </p>
        )}
        {!study && inventory.data && !latest && (
          <p className="muted">
            Select a saved study, or train a synthetic MLP + Transformer study
            to compare both AI players and the two rule-based strategies.
          </p>
        )}
      </section>
      {study && !replay.data && !replay.error && overview.isPending && (
        <p role="status" className="bench-panel muted">
          Loading saved study…
        </p>
      )}
      {study && replay.isFetching && !replay.data && (
        <p role="status" className="bench-panel muted">
          Loading recorded replay…
        </p>
      )}
      {replay.data && (
        <Replay
          key={`${study}/${selectedSeed}/${selectedVariant}/${replay.data.players.length}/${params.get("case")}`}
          data={replay.data}
          study={study}
          start={Math.max(0, Math.trunc(Number(params.get("case")) || 0))}
        />
      )}
    </div>
  );
}

function Replay({
  data,
  study,
  start,
}: {
  data: Competition;
  study: string;
  start: number;
}) {
  const [cursor, setCursor] = useState(
    Math.min(start, data.players[0].events.length),
  ); // number of revealed next-session closes
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [pauseDisagree, setPauseDisagree] = useState(false);
  const [pauseFill, setPauseFill] = useState(false);
  const [selected, setSelected] = useState(0);
  const [view, setView] = useState<"Overview" | "Decision" | "Performance">(
    "Overview",
  );
  const contentRef = useRef<HTMLDivElement>(null);
  function showView(next: typeof view) {
    setView(next);
    // Keep a selected view in sight when navigating from a scrolled chart.
    requestAnimationFrame(() => {
      const content = contentRef.current;
      if (
        content &&
        content.getBoundingClientRect().top <
          (window.innerWidth > 760 ? 205 : 0)
      )
        content.scrollIntoView({ block: "start" });
    });
  }
  const [stage, setStage] = useState<number | null>(null);
  const [showPrevious, setShowPrevious] = useState(false);
  const [reason, setReason] = useState("");
  const length = data.players[0].events.length;
  const player = data.players[selected] ?? data.players[0];
  const stepIndex = Math.min(
    showPrevious && cursor > 0 ? cursor - 1 : cursor,
    length - 1,
  );
  const revealed = stepIndex < cursor;
  const event = player.events[stepIndex];
  const stageNames = [
    "Input & prediction",
    "Position decision",
    "Checks & order",
    "Next opening",
    "Portfolio valuation",
  ];
  const stageValues = [
    {
      observation:
        event.decision.observation ??
        "Exact observation was not stored in this older trace",
      probabilities: event.decision.probabilities,
    },
    {
      target: event.decision.target,
      context: event.decision.execution_context,
    },
    {
      action: event.decision.action,
      order_id: event.decision.order_id,
      side: event.decision.order_side,
      quantity: event.decision.order_quantity,
      pre_trade_rejection: event.decision.pre_trade_rejection,
      policy: event.decision.execution_policy_version,
      risk_profile: event.decision.risk_profile_id,
    },
    revealed ? event.execution : "Available after revealing the next session",
    revealed ? event.portfolio : "Available after revealing the next session",
  ];
  const stageButton = (i: number) => (
    <button
      className="stage-link"
      aria-pressed={stage === i}
      onClick={() => setStage(stage === i ? null : i)}
    >
      {stageNames[i]} <span aria-hidden="true">{stage === i ? "−" : "+"}</span>
    </button>
  );
  const snapshots = (p: Player) => [
    p.initial,
    ...p.events.slice(0, cursor).map((e) => e.portfolio),
  ];
  const current = (p: Player) =>
    cursor ? p.events[cursor - 1].portfolio : p.initial;
  const disagree = (i: number) => {
    const models = data.players.filter((p) => p.run);
    return (
      models.length > 1 &&
      new Set(models.map((p) => p.events[i].decision.target)).size > 1
    );
  };
  function seek(next: number) {
    setPlaying(false);
    setCursor(Math.max(0, Math.min(length, next)));
    setShowPrevious(false);
    setReason("");
  }
  useEffect(() => {
    if (!playing || cursor >= length) return;
    const timer = setTimeout(() => {
      const next = cursor + 1;
      setCursor(next);
      setShowPrevious(true);
      if (
        pauseFill &&
        data.players.some((p) => p.events[cursor].execution?.fill_quantity)
      ) {
        setPlaying(false);
        setReason("Paused after an executed order");
      } else if (pauseDisagree && next < length && disagree(next)) {
        setPlaying(false);
        setShowPrevious(false);
        setReason("Paused before a disagreement");
      } else if (next === length) {
        setPlaying(false);
        setReason("Competition complete");
      } else setShowPrevious(false);
    }, 1000 / speed);
    return () => clearTimeout(timer);
    // Disagreement is determined from this immutable replay artifact.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, cursor, speed, length, data, pauseFill, pauseDisagree]);
  const color = (i: number) => plotColors[i % plotColors.length];
  const series = (fn: (p: Player) => number[]): Series[] =>
    data.players.map((p, i) => ({
      name: playerText(p.name),
      points: points(fn(p)),
      color: color(i),
    }));
  const wealth = series((p) => snapshots(p).map((s) => s.equity));
  const pnl = series((p) =>
    snapshots(p).map((s) => s.equity - p.initial.equity),
  );
  const dd = series((p) => {
    let peak = p.initial.equity;
    return snapshots(p).map((s) => {
      peak = Math.max(peak, s.equity);
      return peak > 0 ? (peak - s.equity) / peak : 0;
    });
  });
  const exposure = series((p) =>
    snapshots(p).map((s) => (s.equity > 0 ? 1 - s.cash / s.equity : 0)),
  );
  const probability = data.players.flatMap((p, i) =>
    p.run
      ? [
          {
            name: p.name,
            color: color(i),
            points: p.events
              .slice(0, Math.min(cursor + 1, length))
              .flatMap((e, x) =>
                e.decision.probabilities
                  ? [{ x, y: e.decision.probabilities.long }]
                  : [],
              ),
          },
        ]
      : [],
  );
  const market = data.market.slice(
    data.firstDecision,
    data.firstDecision + cursor + 1,
  );
  const markers = data.players.flatMap((p, i) =>
    p.events.slice(0, cursor).flatMap((e) =>
      e.execution?.fill_bar != null && e.execution.fill_price != null
        ? [
            {
              x: e.execution.fill_bar - data.firstDecision,
              y: e.execution.fill_price,
              label: `${p.name} ${e.decision.order_side}`,
              color: color(i),
              direction:
                e.decision.order_side === "buy"
                  ? ("buy" as const)
                  : ("sell" as const),
            },
          ]
        : [],
    ),
  );
  const domain: [number, number] = [0, cursor];
  const displayDate = (i: number) => `Session ${i}`;
  return (
    <>
      <section className="bench-panel replay-console replay-toolbar">
        <div className="section-heading">
          <div>
            <h2>
              Session {cursor} / {length}
            </h2>
            <p>
              {data.origin === "synthetic"
                ? "Synthetic"
                : "Historical validation"}{" "}
              · {data.profile.profile_id} · {money(data.profile.initial_cash)}{" "}
              virtual starting capital per player
            </p>
          </div>
          <span className="bench-badge">
            {cursor === length ? "Complete" : playing ? "Playing" : "Paused"}
          </span>
        </div>
        {data.missing.length > 0 && (
          <p className="bench-notice">
            Awaiting recorded runs: {data.missing.join(", ")}
          </p>
        )}
        <div className="bench-actions replay-actions">
          <button
            className="bench-button primary-action"
            disabled={cursor === length}
            onClick={() => {
              setReason("");
              setPlaying(!playing);
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            className="bench-button"
            disabled={cursor === length}
            onClick={() => {
              seek(cursor + 1);
              setShowPrevious(true);
            }}
          >
            Advance one session
          </button>
          <button className="bench-button" onClick={() => seek(0)}>
            Restart
          </button>
          <label className="speed-control">
            Speed{" "}
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
            >
              {[0.5, 1, 2, 5, 10].map((x) => (
                <option key={x} value={x}>
                  {x}×
                </option>
              ))}
            </select>
          </label>
          <details className="replay-options">
            <summary>
              Auto-pause
              {pauseDisagree || pauseFill
                ? ` (${Number(pauseDisagree) + Number(pauseFill)})`
                : ""}
            </summary>
            <div className="replay-options-popover">
              <label>
                <input
                  type="checkbox"
                  checked={pauseDisagree}
                  onChange={(e) => setPauseDisagree(e.target.checked)}
                />{" "}
                Pause on AI disagreement
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={pauseFill}
                  onChange={(e) => setPauseFill(e.target.checked)}
                />{" "}
                Pause on fill
              </label>
            </div>
          </details>
        </div>
        <label className="replay-seek">
          Session{" "}
          <input
            aria-label="Seek replay session"
            type="range"
            min={0}
            max={length}
            value={cursor}
            onChange={(e) => seek(Number(e.target.value))}
          />
        </label>
        {reason && <p role="status">{reason}</p>}
        <nav className="replay-views" aria-label="Competition views">
          {(["Overview", "Decision", "Performance"] as const).map((name) => (
            <button
              key={name}
              className="bench-button"
              aria-pressed={view === name}
              onClick={() => showView(name)}
            >
              {name}
            </button>
          ))}
        </nav>
      </section>
      <div className="player-grid" aria-label="Select a player to inspect">
        {data.players.map((p, i) => (
          <button
            key={p.name}
            className="player-card"
            aria-pressed={selected === i}
            onClick={() => {
              setSelected(i);
              showView("Decision");
            }}
            style={{ borderTopColor: color(i) }}
          >
            <span>
              <i
                className="player-dot"
                style={{ background: color(i) }}
                aria-hidden="true"
              />
              {playerText(p.name)}
            </span>
            <strong>{money(current(p).equity)}</strong>
            <span>
              <span
                className={
                  current(p).equity >= p.initial.equity
                    ? "positive-value"
                    : "negative-value"
                }
              >
                {money(current(p).equity - p.initial.equity)} PnL
              </span>{" "}
              · {pct(current(p).equity / p.initial.equity - 1)}
            </span>
            <small>
              {cursor < length
                ? `Next target: ${targetText(p.events[cursor].decision.target)}`
                : "Final balance"}
            </small>
          </button>
        ))}
      </div>
      <div ref={contentRef} className="replay-view-content research-stack">
        {view === "Overview" && (
          <>
            <section className="competition-chart-grid replay-market-charts">
              <LinePlot
                title="Market close and executed orders"
                series={[
                  {
                    name: "Simulated close",
                    points: points(market.map((b) => b.close)),
                    color: "#819599",
                  },
                ]}
                markers={markers}
                xDomain={domain}
                formatX={displayDate}
                xLabel="Revealed session · ▲ buy · ▼ sell · actual fill prices"
                yLabel="Simulated price"
              />
              <LinePlot
                title="Portfolio equity comparison"
                series={wealth}
                xDomain={domain}
                formatX={displayDate}
                xLabel="Revealed session"
                yLabel="Virtual currency"
              />
            </section>
            <section className="bench-panel decision-preview">
              <div>
                <h2>
                  {playerText(player.name)} · Decision {stepIndex + 1}
                </h2>
                <p className="muted">
                  {revealed
                    ? "Next-session outcome revealed."
                    : "The next session has not yet been revealed."}
                </p>
              </div>
              <div className="decision-preview-stat">
                <span>Prediction</span>
                <strong>
                  {event.decision.probabilities
                    ? pct(event.decision.probabilities.long) + " up"
                    : "Rule-based"}
                </strong>
              </div>
              <div className="decision-preview-stat">
                <span>Target</span>
                <strong>{targetText(event.decision.target)}</strong>
              </div>
              <button
                className="bench-button"
                onClick={() => showView("Decision")}
              >
                Inspect decision <span aria-hidden="true">→</span>
              </button>
            </section>
          </>
        )}
        {view === "Decision" && (
          <section className="bench-panel">
            <div className="section-heading">
              <div>
                <h2>
                  {playerText(player.name)} · Decision {stepIndex + 1}
                </h2>
                <p className="muted">
                  {revealed
                    ? "Next-session outcome revealed."
                    : "The next session has not yet been revealed."}
                </p>
              </div>
              <button
                className="bench-button"
                disabled={cursor === 0 || cursor === length}
                onClick={() => setShowPrevious(!showPrevious)}
              >
                {showPrevious
                  ? "Inspect current decision"
                  : "Inspect last revealed decision"}
              </button>
            </div>
            <div className="engine-stages">
              <div>
                {stageButton(0)}
                <p>64 past returns</p>
                <strong>
                  {event.decision.probabilities
                    ? pct(event.decision.probabilities.long) + " up"
                    : "Rule-based"}
                </strong>
              </div>
              <div>
                {stageButton(1)}
                <strong>{targetText(event.decision.target)}</strong>
                <p>
                  Held {event.decision.execution_context.position_quantity}{" "}
                  units
                </p>
              </div>
              <div>
                {stageButton(2)}
                <strong>
                  {actionText[event.decision.action] ?? event.decision.action}
                </strong>
                <p>
                  {event.decision.order_quantity != null
                    ? `${event.decision.order_quantity} units`
                    : "No order size"}
                </p>
                <p>{event.decision.pre_trade_rejection}</p>
              </div>
              <div>
                {stageButton(3)}
                <strong>
                  {revealed
                    ? (event.execution?.status ?? "No order")
                    : "Hidden"}
                </strong>
                {revealed && event.execution && (
                  <p>
                    {event.execution.fill_quantity ?? "—"} units ·{" "}
                    {event.execution.fill_price != null
                      ? money(event.execution.fill_price)
                      : "—"}{" "}
                    · fee {event.execution.fee ?? "—"}
                    <br />
                    {event.execution.rejection_reason}
                  </p>
                )}
              </div>
              <div>
                {stageButton(4)}
                <strong>
                  {revealed ? money(event.portfolio.equity) : "Hidden"}
                </strong>
                {revealed && (
                  <p>
                    Cash {money(event.portfolio.cash)} · total fees{" "}
                    {money(event.portfolio.total_fees)}
                  </p>
                )}
              </div>
            </div>
            {stage !== null && (
              <div className="stage-record">
                <h3>{stageNames[stage]} · recorded details</h3>
                <pre className="engine-json">
                  {JSON.stringify(stageValues[stage], null, 2)}
                </pre>
              </div>
            )}
            <Explanation
              key={`${player.sourceRun}/${player.playerIndex}/${stepIndex}/${revealed}`}
              path={`/${study}/${player.sourceRun}/explain/${stepIndex}`}
              player={player.playerIndex}
              revealed={revealed}
            />
            <div className="comparison-figures">
              <LinePlot
                title="Visible input window"
                series={[
                  {
                    name: "Relative adjusted close",
                    points: points(
                      data.market
                        .slice(
                          event.decision.decision_bar_index - 64,
                          event.decision.decision_bar_index + 1,
                        )
                        .map(
                          (b, _, bars) =>
                            (100 * b.adjusted_close) / bars[0].adjusted_close,
                        ),
                    ),
                  },
                ]}
                xLabel="65 visible closes (64 returns)"
                yLabel="Index base 100"
              />
              {player.occlusion && (
                <SensitivityPlot values={player.occlusion[stepIndex]} />
              )}
            </div>
            <details>
              <summary>Exact engine records</summary>
              <pre className="engine-json">
                {JSON.stringify(revealed ? event : event.decision, null, 2)}
              </pre>
            </details>
            {player.run && (
              <Link
                className="bench-button"
                href={`/studies?study=${study}&run=${player.run}&case=${stepIndex}&tab=Inspector`}
              >
                Open research inspector →
              </Link>
            )}
          </section>
        )}
        {view === "Performance" && (
          <section className="competition-chart-grid replay-market-charts">
            <LinePlot
              title="Cumulative profit and loss"
              series={pnl}
              xDomain={domain}
              formatX={displayDate}
              xLabel="Revealed session"
              yLabel="Virtual currency"
            />
            <LinePlot
              title="Drawdown comparison"
              series={dd}
              xDomain={domain}
              formatX={displayDate}
              xLabel="Revealed session"
              formatY={pct}
            />
            <LinePlot
              title="Market exposure"
              series={exposure}
              xDomain={domain}
              formatX={displayDate}
              xLabel="Revealed session"
              formatY={pct}
            />
            <LinePlot
              title="AI prediction probability"
              series={probability}
              xDomain={domain}
              formatX={displayDate}
              yDomain={[0, 1]}
              xLabel="Decision session · only available predictions"
              formatY={pct}
            />
          </section>
        )}
      </div>
      <details className="bench-panel replay-provenance">
        <summary>Recorded execution settings and provenance</summary>
        <p>
          {data.accounting} · Spread {data.profile.spread_bps} bps · Fee per
          unit {data.profile.fee_per_unit} · Cash reserve{" "}
          {pct(data.profile.cash_reserve_fraction ?? 0)}
        </p>
        <p>
          Profile hash {data.profileHash}
          <br />
          Dataset {data.dataset}
        </p>
        <pre>{JSON.stringify(data.profile.risk_limits, null, 2)}</pre>
      </details>
    </>
  );
}

function Explanation({
  path,
  player,
  revealed,
}: {
  path: string;
  player: number;
  revealed: boolean;
}) {
  const [mode, setMode] = useState("standard");
  const [phase, setPhase] = useState("decision");
  const [format, setFormat] = useState("llm");
  const [result, setResult] = useState<{
    source: string;
    check: string;
    facts: string[];
  } | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function generate() {
    setBusy(true);
    setResult(null);
    setError("");
    try {
      setResult(
        await api(
          `${path}?mode=${mode}&phase=${phase}&player=${player}&output=${format}`,
          {},
        ),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="explanation-panel">
      <h3>Explain the decision</h3>
      <div className="bench-actions">
        <label>
          Format{" "}
          <select
            aria-label="Explanation format"
            value={format}
            disabled={busy}
            onChange={(e) => {
              setFormat(e.target.value);
              setResult(null);
            }}
          >
            <option value="llm">LLM explanation</option>
            <option value="facts">Fact table</option>
          </select>
        </label>
        <label>
          Language{" "}
          <select
            aria-label="Explanation language"
            disabled={busy}
            value={mode}
            onChange={(e) => {
              setMode(e.target.value);
              setResult(null);
            }}
          >
            <option value="standard">Accessible</option>
            <option value="scientific">Scientific</option>
          </select>
        </label>
        {revealed && (
          <label>
            Evidence{" "}
            <select
              aria-label="Evidence"
              disabled={busy}
              value={phase}
              onChange={(e) => {
                setPhase(e.target.value);
                setResult(null);
              }}
            >
              <option value="decision">At decision time</option>
              <option value="outcome">After the reveal</option>
            </select>
          </label>
        )}
        <button
          className="bench-button primary-action"
          disabled={busy}
          onClick={generate}
        >
          {busy ? "Generating…" : "Explain from measured facts"}
        </button>
      </div>
      {result && (
        <>
          <p className="muted">
            {result.source} · {result.check}
          </p>
          {format === "facts" ? (
            <table>
              <thead>
                <tr>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {result.facts.map((f) => (
                  <tr key={f}>
                    <td>{f}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            result.facts.map((f) => <p key={f}>{f}</p>)
          )}
        </>
      )}
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
