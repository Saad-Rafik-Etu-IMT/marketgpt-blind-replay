"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import {
  Evaluation,
  EvaluationEntry,
  laboratoryApi,
  labelText,
  numberText,
  Replay,
  Run,
} from "@/lib/research/laboratory";
import { LinePlot, PlotFrame, plotColors, ChartSync } from "./ScientificPlot";
import { MarketWorkbench } from "./MarketWorkbench";
import { PredictionExplorer } from "./PredictionExplorer";
import type { PredictionCase } from "@/lib/research/prediction-analysis";
import { ConfusionPlot } from "./ResearchFigures";
import {
  humanLabel,
  runLabel,
  entryKey,
  chooseEntry,
  scientificFinding,
  playbackFrame,
  replayCompatibility,
} from "@/lib/research/product-context";
import { FindingSummary } from "./ProductEvidence";

export function EvaluationView({
  id,
  part,
  entry,
  changeEntry,
  run,
  autoEvaluating = false,
  explanation = false,
  runs = [],
  onReplayCase,
}: {
  id: string;
  part: string;
  entry: string | null;
  changeEntry: (entry: EvaluationEntry) => void;
  run?: Run;
  autoEvaluating?: boolean;
  explanation?: boolean;
  runs?: Run[];
  onReplayCase?: (row: PredictionCase) => void;
}) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState("configuration");
  const query = useQuery({
    queryKey: ["lab-evaluation", id, part],
    retry: false,
    queryFn: () => laboratoryApi<Evaluation>(`/${id}/evaluation/${part}`),
    refetchInterval: autoEvaluating ? 1600 : false,
  });
  async function evaluate() {
    setBusy(true);
    setError("");
    try {
      await laboratoryApi(`/${id}/evaluate?part=${part}`, {});
      await query.refetch();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const report = query.data;
  const selected = chooseEntry(report?.results ?? [], entry, run);
  const matching =
    report?.results.filter((e) => !selected || e.target === selected.target) ??
    [];
  const filtered = matching
    .filter((e) =>
      [e.feature_profile, e.preset, e.target]
        .map(humanLabel)
        .join(" ")
        .toLowerCase()
        .includes(filter.toLowerCase()),
    )
    .sort((a, b) =>
      sort === "loss"
        ? a.ensemble.loss - b.ensemble.loss
        : entryKey(a).localeCompare(entryKey(b)),
    );
  return (
    <section className="bench-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Evidence</p>
          <h2>
            {explanation
              ? "Measured model sensitivity"
              : "Scientific evaluation"}
          </h2>
        </div>
        <span className="bench-badge">
          {humanLabel(part)} · ensemble statistics
        </span>
      </div>
      <p className="muted">
        Development results are not final test evidence. Negative or
        inconclusive outcomes are valid.
      </p>
      <button
        className="bench-button primary-action"
        disabled={busy || autoEvaluating}
        onClick={() => void evaluate()}
      >
        {busy || autoEvaluating
          ? "Evaluating…"
          : report
            ? "Recompute development report"
            : "Generate evaluation report"}
      </button>
      {error && (
        <p role="alert" className="bench-notice">
          {error}
        </p>
      )}
      {!report && (
        <p className="muted">
          {query.isPending
            ? "Loading saved evaluation…"
            : "No evaluation recorded for this partition."}
        </p>
      )}
      {report && (
        <label className="lab-field">
          Configuration and target
          <select
            value={selected ? entryKey(selected) : ""}
            onChange={(e) => {
              const next = report.results.find(
                (r) => entryKey(r) === e.target.value,
              );
              if (next) changeEntry(next);
            }}
          >
            <option value="" disabled>
              Choose a matching configuration
            </option>
            {report.results.map((r) => (
              <option key={entryKey(r)} value={entryKey(r)}>
                {humanLabel(r.feature_profile)} · {humanLabel(r.preset)} ·{" "}
                {humanLabel(r.target)}
              </option>
            ))}
          </select>
        </label>
      )}
      {report && !selected && (
        <p className="bench-notice">
          No evidence matches the selected run and target. Select a recorded
          configuration above; the workspace will switch to the matching run.
        </p>
      )}
      {selected && report && (
        <>
          <FindingSummary entry={selected} part={part} />
          {explanation ? (
            <Importance entry={selected} />
          ) : (
            <EvaluationFigures entry={selected} />
          )}
          {run?.state === "complete" && (
            <PredictionExplorer
              key={run.key + selected.target + part}
              id={id}
              run={run.key}
              part={part}
              target={selected.target}
              runs={runs}
              onReplay={onReplayCase}
            />
          )}
          {!explanation && (
            <>
              <h3>Feature ablation</h3>
              <p>{report.ablation.summary}</p>
              <div className="lab-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Profile</th>
                      <th>Target</th>
                      <th>Difference</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.ablation.findings.map((r, i) => (
                      <tr key={i}>
                        <td>
                          <button
                            className="bench-button"
                            disabled={
                              !report.results.some(
                                (e) =>
                                  e.feature_profile === r.feature_profile &&
                                  e.preset === r.preset &&
                                  e.target === r.target,
                              )
                            }
                            onClick={() => {
                              const found = report.results.find(
                                (e) =>
                                  e.feature_profile === r.feature_profile &&
                                  e.preset === r.preset &&
                                  e.target === r.target,
                              );
                              if (found) changeEntry(found);
                            }}
                          >
                            {humanLabel(r.feature_profile)} /{" "}
                            {humanLabel(r.preset)}
                          </button>
                        </td>
                        <td>{humanLabel(r.target)}</td>
                        <td>{numberText(r.comparison.mean_difference)}</td>
                        <td>{r.category}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <h3>Compare configurations · {humanLabel(selected.target)}</h3>
              <details className="supporting-details">
                <summary>
                  Configuration loss heatmap · same partition and target
                </summary>
                <div className="lab-table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Features / model</th>
                        {[...new Set(matching.map((e) => e.preset))].map(
                          (v) => (
                            <th key={v}>{humanLabel(v)}</th>
                          ),
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {[...new Set(matching.map((e) => e.feature_profile))].map(
                        (fp) => (
                          <tr key={fp}>
                            <th>{humanLabel(fp)}</th>
                            {[...new Set(matching.map((e) => e.preset))].map(
                              (ps) => {
                                const e = matching.find(
                                    (e) =>
                                      e.feature_profile === fp &&
                                      e.preset === ps,
                                  ),
                                  losses = matching.map((e) => e.ensemble.loss),
                                  lo = Math.min(...losses),
                                  hi = Math.max(...losses);
                                return (
                                  <td key={ps}>
                                    {e ? (
                                      <button
                                        className="bench-button"
                                        style={{
                                          background: `color-mix(in srgb, var(--accent) ${15 + 55 * (1 - (e.ensemble.loss - lo) / (hi - lo || 1))}%, var(--panel))`,
                                        }}
                                        onClick={() => changeEntry(e)}
                                      >
                                        {numberText(e.ensemble.loss)}
                                      </button>
                                    ) : (
                                      "—"
                                    )}
                                  </td>
                                );
                              },
                            )}
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
                <p className="muted">
                  Darker cells indicate lower observed loss; not a promotion
                  decision. Compare paired confidence intervals before
                  interpreting differences.
                </p>
              </details>
              <div className="lab-two">
                <label>
                  Filter models and features
                  <input
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    placeholder="e.g. Transformer"
                  />
                </label>
                <label>
                  Sort configurations
                  <select
                    value={sort}
                    onChange={(e) => setSort(e.target.value)}
                  >
                    <option value="configuration">Configuration</option>
                    <option value="loss">
                      Measured loss (not model promotion)
                    </option>
                  </select>
                </label>
              </div>
              <div className="lab-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Configuration</th>
                      <th>Seeds</th>
                      <th>{selected.loss}</th>
                      <th>Compared with strongest baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((e) => (
                      <tr key={entryKey(e)}>
                        <td>
                          <button
                            className="bench-button"
                            aria-pressed={entryKey(e) === entryKey(selected)}
                            onClick={() => changeEntry(e)}
                          >
                            {humanLabel(e.feature_profile)} ·{" "}
                            {humanLabel(e.preset)}
                          </button>
                        </td>
                        <td>{e.seeds.length}</td>
                        <td>{numberText(e.ensemble.loss)}</td>
                        <td>{scientificFinding(e).headline}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {report.notes.map((note) => (
                <p className="muted" key={note}>
                  {note}
                </p>
              ))}
            </>
          )}
        </>
      )}
    </section>
  );
}

function EvaluationFigures({ entry: e }: { entry: EvaluationEntry }) {
  const names = ["Model ensemble", ...Object.keys(e.baselines)];
  const losses = [
    e.ensemble.loss,
    ...Object.values(e.baselines).map((x) => x.loss),
  ];
  return (
    <>
      <LinePlot
        title="Model versus statistical baselines · lower is better"
        xLabel="Candidate"
        yLabel={e.loss}
        formatX={(x) => (names[Math.round(x)] ? String(Math.round(x) + 1) : "")}
        series={[
          {
            name: "Model",
            color: plotColors[0],
            scatter: true,
            dots: true,
            points: [{ x: 0, y: losses[0] }],
          },
          {
            name: "Baselines",
            color: plotColors[1],
            scatter: true,
            dots: true,
            points: losses.slice(1).map((y, i) => ({ x: i + 1, y })),
          },
        ]}
      />
      <ConfidenceIntervals entry={e} />
      <div className="lab-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Candidate</th>
              <th>Loss</th>
              <th>95% paired interval (model − baseline)</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1 · Model ensemble</td>
              <td>{numberText(e.ensemble.loss)}</td>
              <td>—</td>
              <td>—</td>
            </tr>
            {Object.entries(e.baselines).map(([name, b], i) => (
              <tr key={name}>
                <td>
                  {i + 2} · {labelText(name)}
                </td>
                <td>{numberText(b.loss)}</td>
                <td>
                  [{numberText(b.comparison.interval_95[0])},{" "}
                  {numberText(b.comparison.interval_95[1])}]
                </td>
                <td>{b.comparison.verdict}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details className="supporting-details">
        <summary>
          Predictive diagnostics · seeds, calibration and classification
        </summary>
        <div className="lab-two">
          <LinePlot
            title="Seed variability"
            xLabel="Seed"
            yLabel={e.loss}
            series={[
              {
                name: "Per-seed loss",
                color: plotColors[0],
                dots: true,
                scatter: true,
                points: Object.entries(e.per_seed_loss).map(([x, y]) => ({
                  x: Number(x),
                  y,
                })),
              },
            ]}
          />
          {e.calibration && (
            <LinePlot
              title="Probability reliability"
              xLabel="Predicted probability"
              yLabel="Observed frequency"
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              series={[
                {
                  name: "Perfect calibration",
                  dashed: true,
                  points: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                  ],
                },
                {
                  name: "Observed bins",
                  color: plotColors[0],
                  dots: true,
                  points: e.calibration.reliability_up.map((x) => ({
                    x: x.confidence,
                    y: x.frequency,
                  })),
                },
              ]}
            />
          )}
        </div>
        {e.curves?.roc.length ? (
          <div className="lab-two">
            <LinePlot
              title="ROC curve"
              xLabel="False positive rate"
              yLabel="True positive rate"
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              series={[
                {
                  name: "Random ranking",
                  dashed: true,
                  points: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                  ],
                },
                {
                  name: "Model",
                  points: e.curves.roc.map(([x, y]) => ({ x, y })),
                },
              ]}
            />
            <LinePlot
              title="Precision-recall curve"
              xLabel="Recall"
              yLabel="Precision"
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              series={[
                {
                  name: "Model",
                  points: e.curves.precision_recall.map(([x, y]) => ({ x, y })),
                },
              ]}
            />
          </div>
        ) : null}
        {Array.isArray(e.ensemble.metrics.confusion) &&
          e.ensemble.metrics.confusion.length === 2 && (
            <ConfusionPlot
              cells={e.ensemble.metrics.confusion.flat() as number[]}
            />
          )}
        <h3>Predictive metrics</h3>
        <div className="metric-strip lab-metrics">
          {Object.entries(e.ensemble.metrics)
            .filter(([, v]) => typeof v === "number")
            .map(([name, value]) => (
              <div key={name}>
                <span>{labelText(name)}</span>
                <strong>{numberText(value)}</strong>
              </div>
            ))}
        </div>
      </details>
      <details className="supporting-details">
        <summary>Robustness · market conditions and transaction costs</summary>
        {Object.entries(e.regimes).map(([kind, rows]) => (
          <LinePlot
            key={kind}
            title={`Regime loss · ${humanLabel(kind)}`}
            xLabel="Regime"
            yLabel={e.loss}
            formatX={(x) => humanLabel(Object.keys(rows)[Math.round(x)] ?? "")}
            series={[
              {
                name: "Model",
                points: Object.values(rows).map((r, x) => ({
                  x,
                  y: r.model_loss,
                })),
                dots: true,
                scatter: true,
              },
              {
                name: "Strongest baseline",
                points: Object.values(rows).map((r, x) => ({
                  x,
                  y: r.strongest_baseline_loss,
                })),
                dots: true,
                scatter: true,
              },
            ]}
          />
        ))}
        {e.cost_sensitivity?.length ? (
          <>
            <LinePlot
              title="Transaction cost sensitivity · simplified return proxy"
              xLabel="Cost per position change (bps)"
              yLabel="Equal-weight cumulative log return"
              series={[
                {
                  name: "Model",
                  color: plotColors[0],
                  points: e.cost_sensitivity.map((x) => ({
                    x: x.cost_bps_per_change,
                    y: x.model_ensemble.total_log_return_equal_weight,
                  })),
                },
                {
                  name: "Logistic regression",
                  color: plotColors[1],
                  points: e.cost_sensitivity.map((x) => ({
                    x: x.cost_bps_per_change,
                    y: x.logistic_regression.total_log_return_equal_weight,
                  })),
                },
                {
                  name: "Always long",
                  color: plotColors[2],
                  points: e.cost_sensitivity.map((x) => ({
                    x: x.cost_bps_per_change,
                    y: x.always_long.total_log_return_equal_weight,
                  })),
                },
              ]}
            />
            <p className="muted">
              This is a development-only vectorized proxy, not the broker replay
              or final test evidence.
            </p>
          </>
        ) : null}
        <h3>Market-regime robustness</h3>
        <div className="lab-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Regime</th>
                <th>Rows</th>
                <th>Model loss</th>
                <th>Baseline loss</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(e.regimes).flatMap(([kind, regimes]) =>
                Object.entries(regimes).map(([name, r]) => (
                  <tr key={kind + name}>
                    <td>
                      {labelText(kind)} · {labelText(name)}
                    </td>
                    <td>{r.rows}</td>
                    <td>{numberText(r.model_loss)}</td>
                    <td>{numberText(r.strongest_baseline_loss)}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

export function ConfidenceIntervals({ entry }: { entry: EvaluationEntry }) {
  const rows = Object.entries(entry.baselines).flatMap(([name, b]) => {
    const [lo, hi] = b.comparison.interval_95;
    return lo !== null &&
      hi !== null &&
      Number.isFinite(lo) &&
      Number.isFinite(hi) &&
      Number.isFinite(b.comparison.mean_difference)
      ? [{ name, lo, hi, mean: b.comparison.mean_difference }]
      : [];
  });
  if (!rows.length)
    return (
      <p className="muted">Paired confidence intervals are unavailable.</p>
    );
  const min = Math.min(0, ...rows.map((r) => r.lo)),
    max = Math.max(0, ...rows.map((r) => r.hi));
  const x = (v: number) => 190 + (465 * (v - min)) / (max - min || 1);
  const height = 75 + rows.length * 28;
  return (
    <PlotFrame
      title="Paired 95% confidence intervals · model minus baseline"
      height={height}
    >
      <line
        x1={x(0)}
        x2={x(0)}
        y1={24}
        y2={height - 27}
        stroke="var(--muted)"
        strokeDasharray="4 4"
      />
      {rows.map((r, i) => (
        <g key={r.name}>
          <text
            x={180}
            y={44 + i * 28}
            textAnchor="end"
            fontSize="11"
            fill="var(--fg)"
          >
            {humanLabel(r.name)}
          </text>
          <line
            x1={x(r.lo)}
            x2={x(r.hi)}
            y1={40 + i * 28}
            y2={40 + i * 28}
            stroke={plotColors[0]}
            strokeWidth={3}
          />
          <circle cx={x(r.mean)} cy={40 + i * 28} r={4} fill={plotColors[0]}>
            <title>
              {numberText(r.mean)} [{numberText(r.lo)}, {numberText(r.hi)}]
            </title>
          </circle>
        </g>
      ))}
      <text x={190} y={height - 9} fill="var(--muted)" fontSize="10">
        {numberText(min)}
      </text>
      <text
        x={655}
        y={height - 9}
        textAnchor="end"
        fill="var(--muted)"
        fontSize="10"
      >
        {numberText(max)}
      </text>
    </PlotFrame>
  );
}

function Importance({ entry }: { entry: EvaluationEntry }) {
  const data = entry.importance ?? [];
  return (
    <>
      <p className="muted">
        Circular shifts perturb the temporal alignment of one feature within
        each asset. A positive loss increase indicates measured predictive
        association, not causation.
      </p>
      {!data.length ? (
        <p>
          No importance measurements for this target; select the primary target.
        </p>
      ) : (
        <>
          <LinePlot
            title="Feature sensitivity · change in loss"
            xLabel="Rank (features below)"
            yLabel="Loss increase"
            series={[
              {
                name: "Measured perturbation",
                color: plotColors[0],
                scatter: true,
                dots: true,
                points: data
                  .slice(0, 15)
                  .map((v, i) => ({ x: i + 1, y: v.loss_increase })),
              },
            ]}
          />
          <div className="lab-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Columns</th>
                  <th>Loss increase</th>
                </tr>
              </thead>
              <tbody>
                {data.map((v, i) => (
                  <tr key={v.feature}>
                    <td>
                      {i + 1} · {labelText(v.feature)}
                    </td>
                    <td>{v.columns}</td>
                    <td>{numberText(v.loss_increase)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <p className="muted">
        For human explanation studies and the legacy Transformer demonstration,
        use <Link href="/studies">saved research studies</Link>.
      </p>
    </>
  );
}

export function ReplayView({
  id,
  run,
  part,
  asset,
  initialCursor = 0,
  onCursor,
  runs = [],
  opponent = "",
  onOpponent,
  inspectDate,
}: {
  id: string;
  run: string;
  part: string;
  asset?: string;
  initialCursor?: number;
  onCursor?: (value: number) => void;
  runs?: Run[];
  opponent?: string;
  onOpponent?: (value: string) => void;
  inspectDate?: string;
}) {
  const [requestedCursor, setCursor] = useState(initialCursor),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [pauseDisagreement, setPauseDisagreement] = useState(true);
  const [bookmarks, setBookmarks] = useState<{ bar: number; note: string }[]>(
    [],
  );
  const [note, setNote] = useState("");
  const [guesses, setGuesses] = useState<Record<number, string>>({});
  const [blindHuman, setBlindHuman] = useState(false);
  const bookmarkKey = `blind-replay.bookmarks.${id}.${run}.${part}.${asset ?? ""}`;
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(bookmarkKey) ?? "[]");
      if (Array.isArray(saved))
        setBookmarks(
          saved
            .filter(
              (b) => Number.isInteger(b?.bar) && typeof b?.note === "string",
            )
            .slice(0, 200),
        );
    } catch {}
  }, [bookmarkKey]);
  const [explanation, setExplanation] = useState<{
    source: string;
    facts: string[];
  }>();
  const explanationRequest = useRef(0);
  const cursorCallback = useRef(onCursor);
  useEffect(() => {
    cursorCallback.current = onCursor;
  }, [onCursor]);
  const suffix = `?part=${part}${asset ? "&asset=" + encodeURIComponent(asset) : ""}`;
  const path = `/${id}/runs/${run}/replay${suffix}`;
  const query = useQuery({
    queryKey: ["lab-replay", id, run, part, asset],
    enabled: !!run,
    retry: false,
    queryFn: () => laboratoryApi<Replay>(path),
  });
  async function generate() {
    setBusy(true);
    setError("");
    try {
      await laboratoryApi<Replay>(path, {});
      seek(0);
      await query.refetch();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const replay = query.data;
  useEffect(() => {
    if (!inspectDate || !replay) return;
    const match =
      replay.players[0]?.events.findIndex(
        (e) =>
          replay.market[e.decision.decision_bar_index]?.date === inspectDate,
      ) ?? -1;
    if (match >= 0) {
      setCursor(match);
      cursorCallback.current?.(match);
      setPlaying(false);
    } else setError("The evaluated session is not recorded in this replay.");
  }, [inspectDate, replay]);
  const model = replay?.players[0];
  const frame = replay ? playbackFrame(replay, requestedCursor) : undefined;
  const {
    cursor = 0,
    index = 0,
    length = 0,
    event,
    end: marketEnd = 0,
  } = frame ?? {};
  const comparison = useQuery({
    queryKey: ["lab-replay", id, opponent, part, asset],
    enabled: !!opponent && opponent !== run,
    retry: false,
    queryFn: () =>
      laboratoryApi<Replay>(`/${id}/runs/${opponent}/replay${suffix}`),
  });
  const incompatibility =
    replay && comparison.data
      ? replayCompatibility(replay, comparison.data)
      : null;
  const players =
    replay?.players.map((p, i) => ({
      ...p,
      name: i === 0 ? runLabel(run) : humanLabel(p.name),
    })) ?? [];
  if (
    opponent &&
    comparison.data &&
    !incompatibility &&
    comparison.data.players[0]
  )
    players.push({ ...comparison.data.players[0], name: runLabel(opponent) });
  const opponentEvents = !incompatibility
    ? comparison.data?.players[0]?.events
    : undefined;
  const disagreement =
    !!event &&
    !!opponentEvents?.[index] &&
    opponentEvents[index].decision.target !== event.decision.target;
  const hideDecision = blindHuman && cursor < length && !guesses[index];
  const revealedGuesses = Object.entries(guesses).flatMap(([k, guess]) => {
    const e = model?.events[Number(k)],
      bar = e?.decision.decision_bar_index;
    if (bar === undefined || bar + 2 > marketEnd) return [];
    const first = replay?.market[bar + 1]?.open,
      second = replay?.market[bar + 2]?.open;
    if (first === undefined || second === undefined) return [];
    const label = second > first ? "long" : "flat";
    return [{ human: guess === label, model: e?.decision.target === label }];
  });
  function saveBookmark(bar: number, note: string) {
    const next = [
      ...bookmarks.filter((b) => b.bar !== bar),
      { bar, note },
    ].slice(-200);
    setBookmarks(next);
    try {
      localStorage.setItem(bookmarkKey, JSON.stringify(next));
    } catch {}
  }
  function reportDownload() {
    const data = players.map((p) => {
      const snapshots = [
        p.initial,
        ...p.events
          .filter((e) => e.portfolio.bar_index <= marketEnd)
          .map((e) => e.portfolio),
      ];
      let peak = p.initial.equity,
        drawdown = 0;
      for (const s of snapshots) {
        peak = Math.max(peak, s.equity);
        drawdown = Math.max(drawdown, peak > 0 ? 1 - s.equity / peak : 0);
      }
      const last = snapshots.at(-1)!;
      return {
        name: p.name,
        initial_equity: p.initial.equity,
        revealed_equity: last.equity,
        pnl: last.equity - p.initial.equity,
        return_fraction: p.initial.equity
          ? last.equity / p.initial.equity - 1
          : null,
        fees: last.total_fees,
        drawdown,
        events: snapshots.length - 1,
      };
    });
    const a = document.createElement("a"),
      url = URL.createObjectURL(
        new Blob(
          [
            JSON.stringify(
              {
                asset: replay?.asset,
                partition: part,
                revealed_cursor: cursor,
                note: "Revealed virtual broker results only; no future events included.",
                players: data,
                human_comparison: blindHuman
                  ? {
                      resolved: revealedGuesses.length,
                      human_correct: revealedGuesses.filter((r) => r.human)
                        .length,
                      model_correct: revealedGuesses.filter((r) => r.model)
                        .length,
                    }
                  : undefined,
              },
              null,
              2,
            ),
          ],
          { type: "application/json" },
        ),
      );
    a.href = url;
    a.download = "revealed-replay-analysis.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const nextDisagreement =
    !!model?.events[cursor + 1] &&
    !!opponentEvents?.[cursor + 1] &&
    model.events[cursor + 1].decision.target !==
      opponentEvents[cursor + 1].decision.target;
  useEffect(() => {
    if (!playing || cursor >= length) return;
    const timeout = setTimeout(() => {
      explanationRequest.current++;
      setExplanation(undefined);
      setCursor(cursor + 1);
      cursorCallback.current?.(cursor + 1);
      if (cursor + 1 >= length || (pauseDisagreement && nextDisagreement))
        setPlaying(false);
    }, 1000 / speed);
    return () => clearTimeout(timeout);
  }, [playing, cursor, length, speed, pauseDisagreement, nextDisagreement]);
  async function explain(generate: boolean) {
    const request = ++explanationRequest.current;
    try {
      const value = await laboratoryApi<{ source: string; facts: string[] }>(
        `/${id}/runs/${run}/replay/explain${suffix}&index=${index}&outcome=${cursor === length}&generate=${generate}`,
        {},
      );
      if (request === explanationRequest.current) setExplanation(value);
    } catch (e) {
      if (request === explanationRequest.current) setError(String(e));
    }
  }
  function seek(value: number) {
    explanationRequest.current++;
    setCursor(value);
    onCursor?.(value);
    setPlaying(false);
    setExplanation(undefined);
  }
  return (
    <section className="bench-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Deterministic simulator</p>
          <h2>Registered model replay</h2>
        </div>
        <button
          className="bench-button primary-action"
          disabled={!run || busy}
          onClick={() => void generate()}
        >
          {busy
            ? "Running broker replay…"
            : replay
              ? "Regenerate replay"
              : "Generate replay"}
        </button>
      </div>
      <p className="muted">
        Binary direction head only. The trained checkpoint and its exact feature
        contract drive the same broker, orders, risk controls and portfolio
        ledger as the original cockpit.
      </p>
      {!run && <p>Train and select a completed run first.</p>}
      {onOpponent && run && (
        <label className="lab-field">
          Compare an additional completed run
          <select value={opponent} onChange={(e) => onOpponent(e.target.value)}>
            <option value="">No additional model</option>
            {runs
              .filter((r) => r.state === "complete" && r.key !== run)
              .map((r) => (
                <option key={r.key} value={r.key}>
                  {runLabel(r)}
                </option>
              ))}
          </select>
        </label>
      )}
      {opponent && comparison.isError && (
        <div className="bench-actions">
          <p className="muted">
            No saved comparison replay for this asset and partition.
          </p>
          <button
            className="bench-button"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await laboratoryApi(
                  `/${id}/runs/${opponent}/replay${suffix}`,
                  {},
                );
                await comparison.refetch();
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            Generate comparison replay
          </button>
        </div>
      )}
      {incompatibility && (
        <p className="bench-notice">
          Comparison unavailable: {incompatibility}
        </p>
      )}
      {error && <p role="alert">{error}</p>}
      {replay && model && (
        <>
          <p className="bench-notice">
            {replay.asset} · {replay.accounting} · {replay.label}. Recorded
            predictions are disclosed at their decision close, while the
            next-close portfolio is disclosed after execution.
          </p>
          <div className="bench-actions">
            <label>
              <input
                type="checkbox"
                checked={pauseDisagreement}
                onChange={(e) => setPauseDisagreement(e.target.checked)}
              />{" "}
              Pause on model disagreement
            </label>
            <button
              className="bench-button"
              disabled={cursor === length || blindHuman}
              onClick={() => setPlaying((v) => !v)}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <label>
              <input
                type="checkbox"
                checked={blindHuman}
                onChange={(e) => {
                  setBlindHuman(e.target.checked);
                  setPlaying(false);
                  setExplanation(undefined);
                }}
              />{" "}
              Human blind prediction mode
            </label>
            <label>
              Speed{" "}
              <select
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
              >
                {[0.5, 1, 2, 4].map((v) => (
                  <option key={v} value={v}>
                    {v}×
                  </option>
                ))}
              </select>
            </label>
            <button className="bench-button" onClick={() => seek(0)}>
              Restart
            </button>
            <button
              className="bench-button"
              disabled={cursor === 0}
              onClick={() => seek(cursor - 1)}
            >
              Back
            </button>
            <button
              className="bench-button primary-action"
              disabled={cursor === length || hideDecision}
              onClick={() => seek(cursor + 1)}
            >
              Reveal next session
            </button>
            <label>
              Revealed decisions: {cursor}/{length}
              <input
                type="range"
                min={0}
                max={blindHuman ? cursor : length}
                value={cursor}
                onChange={(e) => seek(Number(e.target.value))}
              />
            </label>
          </div>
          <div className="metric-strip">
            {players
              .map((p) => ({
                p,
                latest:
                  p.events
                    .filter((e) => e.portfolio.bar_index <= marketEnd)
                    .at(-1)?.portfolio ?? p.initial,
              }))
              .sort(
                (a, b) =>
                  b.latest.equity / b.p.initial.equity -
                  a.latest.equity / a.p.initial.equity,
              )
              .map(({ p, latest }, rank) => {
                return (
                  <div key={p.name}>
                    <span>
                      #{rank + 1} · {labelText(p.name)}
                    </span>
                    <strong>{numberText(latest.equity, 2)}</strong>
                    <small>
                      {numberText(
                        (latest.equity / p.initial.equity - 1) * 100,
                        2,
                      )}
                      % · P&amp;L{" "}
                      {numberText(latest.equity - p.initial.equity, 2)}
                    </small>
                  </div>
                );
              })}
          </div>
          {event && (
            <article className="finding-card replay-decision">
              <p className="eyebrow">
                Decision at market close ·{" "}
                {replay.market[event.decision.decision_bar_index]?.date}
              </p>
              <h3>
                {cursor === length
                  ? "Last revealed decision"
                  : "Current blind decision"}
              </h3>
              {hideDecision ? (
                <p className="bench-notice">
                  Choose your direction before revealing the registered AI
                  prediction.
                </p>
              ) : (
                <div className="metric-strip">
                  <div>
                    <span>Probability of positive executed return</span>
                    <strong>
                      {event.decision.probabilities?.long === undefined
                        ? "—"
                        : numberText(
                            event.decision.probabilities.long * 100,
                            1,
                          ) + "%"}
                    </strong>
                  </div>
                  <div>
                    <span>Policy target</span>
                    <strong>{humanLabel(event.decision.target)}</strong>
                  </div>
                  <div>
                    <span>Order action</span>
                    <strong>{humanLabel(event.decision.action)}</strong>
                  </div>
                </div>
              )}
              <p className="muted">
                {cursor === length
                  ? "The final execution outcome is revealed."
                  : "The next market session and its execution outcome remain hidden."}
              </p>
              {!hideDecision && disagreement && (
                <p className="bench-notice">
                  Model disagreement: selected model{" "}
                  {humanLabel(event.decision.target)} · opponent{" "}
                  {humanLabel(opponentEvents![index].decision.target)}
                </p>
              )}
              <div className="bench-actions">
                <label>
                  Annotation{" "}
                  <input
                    maxLength={150}
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="Note on this decision"
                  />
                </label>
                <button
                  className="bench-button"
                  onClick={() => {
                    saveBookmark(
                      event.decision.decision_bar_index,
                      note.trim() || "Bookmarked decision",
                    );
                    setNote("");
                  }}
                >
                  Bookmark
                </button>
              </div>
              {bookmarks.length > 0 && (
                <div className="bench-actions">
                  {bookmarks
                    .filter((b) => b.bar <= marketEnd)
                    .map((b) => (
                      <button
                        key={b.bar}
                        className="bench-button"
                        onClick={() =>
                          seek(
                            model.events.findIndex(
                              (e) => e.decision.decision_bar_index === b.bar,
                            ),
                          )
                        }
                      >
                        {replay.market[b.bar].date} · {b.note}
                      </button>
                    ))}
                </div>
              )}
              {blindHuman && (
                <>
                  <p className="muted">
                    Human predictions lock when submitted. Scoring waits until
                    both execution-horizon opens have been revealed. This is a
                    local educational exercise, not a protected benchmark.
                  </p>
                  <div className="bench-actions">
                    {["long", "flat"].map((g) => (
                      <button
                        key={g}
                        className="bench-button"
                        aria-pressed={guesses[index] === g}
                        disabled={cursor === length || !!guesses[index]}
                        onClick={() =>
                          setGuesses((v) => ({ ...v, [index]: g }))
                        }
                      >
                        {g === "long"
                          ? "Predict positive"
                          : "Predict non-positive"}
                      </button>
                    ))}
                  </div>
                  <p>
                    Resolved guesses: {revealedGuesses.length} · Human correct:{" "}
                    {revealedGuesses.filter((r) => r.human).length} · AI
                    correct: {revealedGuesses.filter((r) => r.model).length}
                  </p>
                </>
              )}
              {cursor > 0 && (
                <p>
                  Previous revealed execution:{" "}
                  {humanLabel(
                    model.events[cursor - 1]?.execution?.status ?? "No order",
                  )}{" "}
                  · fill{" "}
                  {numberText(
                    model.events[cursor - 1]?.execution?.fill_price,
                    2,
                  )}{" "}
                  · fee{" "}
                  {numberText(model.events[cursor - 1]?.execution?.fee, 2)}
                </p>
              )}
              <div className="bench-actions">
                <button
                  className="bench-button"
                  disabled={hideDecision}
                  onClick={() => void explain(false)}
                >
                  Measured facts
                </button>
                <button
                  className="bench-button"
                  disabled={hideDecision}
                  onClick={() => void explain(true)}
                >
                  Explain with local LLM
                </button>
              </div>
              {explanation && (
                <div>
                  <p>{explanation.source}</p>
                  <ul>
                    {explanation.facts.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </article>
          )}
          <ChartSync>
            <div className="lab-two">
              {replay.market[0]?.open !== undefined ? (
                <MarketWorkbench
                  replay={replay}
                  end={marketEnd}
                  hiddenBar={
                    hideDecision
                      ? event?.decision.decision_bar_index
                      : undefined
                  }
                  annotations={[
                    ...bookmarks,
                    ...(opponentEvents ?? []).flatMap((e, i) =>
                      model.events[i] &&
                      i < cursor &&
                      e.decision.target !== model.events[i].decision.target
                        ? [
                            {
                              bar: e.decision.decision_bar_index,
                              note: "Model disagreement",
                            },
                          ]
                        : [],
                    ),
                  ]}
                  onSelect={(bar) => {
                    const nearest = model.events
                      .slice(0, Math.min(cursor + 1, length))
                      .reduce(
                        (best, e, i) =>
                          Math.abs(e.decision.decision_bar_index - bar) <
                          Math.abs(
                            model.events[best].decision.decision_bar_index -
                              bar,
                          )
                            ? i
                            : best,
                        0,
                      );
                    if (length) seek(nearest);
                  }}
                />
              ) : (
                <LinePlot
                  title="Revealed market prices and fills"
                  sync
                  xDomain={[0, Math.max(1, marketEnd)]}
                  xLabel="Market bar"
                  yLabel="Index units"
                  series={[
                    {
                      name: "Adjusted close",
                      points: replay.market
                        .slice(0, marketEnd + 1)
                        .map((b, x) => ({ x, y: b.close })),
                    },
                  ]}
                  markers={model.events
                    .filter(
                      (e) =>
                        e.execution &&
                        e.execution.fill_bar !== null &&
                        e.execution.fill_price !== null &&
                        e.portfolio.bar_index <= marketEnd,
                    )
                    .map((e) => ({
                      x: e.execution!.fill_bar!,
                      y: e.execution!.fill_price!,
                      label: e.decision.action,
                      color: plotColors[0],
                    }))}
                />
              )}
              <LinePlot
                title="Marked equity · same market clock"
                sync
                xDomain={[0, Math.max(1, marketEnd)]}
                formatX={(v) => replay.market[Math.round(v)]?.date ?? ""}
                xLabel="Market bar"
                yLabel="Virtual currency"
                series={players.map((p, i) => ({
                  name: p.name,
                  color: plotColors[i % plotColors.length],
                  points: [
                    { x: p.initial.bar_index, y: p.initial.equity },
                    ...p.events
                      .filter((e) => e.portfolio.bar_index <= marketEnd)
                      .map((e) => ({
                        x: e.portfolio.bar_index,
                        y: e.portfolio.equity,
                      })),
                  ],
                }))}
              />
            </div>
            <details className="product-details">
              <summary>
                Portfolio diagnostics · cumulative P&amp;L and drawdown
              </summary>
              <div className="lab-two">
                <LinePlot
                  title="Cumulative P&L"
                  sync
                  xDomain={[0, Math.max(1, marketEnd)]}
                  formatX={(v) => replay.market[Math.round(v)]?.date ?? ""}
                  xLabel="Market bar"
                  yLabel="Virtual currency"
                  series={players.map((p, i) => ({
                    name: p.name,
                    color: plotColors[i % plotColors.length],
                    points: [
                      { x: p.initial.bar_index, y: 0 },
                      ...p.events
                        .filter((e) => e.portfolio.bar_index <= marketEnd)
                        .map((e) => ({
                          x: e.portfolio.bar_index,
                          y: e.portfolio.equity - p.initial.equity,
                        })),
                    ],
                  }))}
                />
                <LinePlot
                  title="Drawdown"
                  sync
                  xDomain={[0, Math.max(1, marketEnd)]}
                  formatX={(v) => replay.market[Math.round(v)]?.date ?? ""}
                  xLabel="Market bar"
                  yLabel="Fraction of peak"
                  series={players.map((p, i) => {
                    let peak = p.initial.equity;
                    return {
                      name: p.name,
                      color: plotColors[i % plotColors.length],
                      points: [
                        { x: p.initial.bar_index, y: 0 },
                        ...p.events
                          .filter((e) => e.portfolio.bar_index <= marketEnd)
                          .map((e) => {
                            peak = Math.max(peak, e.portfolio.equity);
                            return {
                              x: e.portfolio.bar_index,
                              y: peak > 0 ? 1 - e.portfolio.equity / peak : 0,
                            };
                          }),
                      ],
                    };
                  })}
                />
              </div>
            </details>
          </ChartSync>
          <button className="bench-button" onClick={reportDownload}>
            Download revealed replay analysis
          </button>
          <details className="product-details">
            <summary>Revealed execution and portfolio ledger</summary>
            <div className="lab-table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Session</th>
                    <th>Action</th>
                    <th>Fill</th>
                    <th>Fees</th>
                    <th>Equity</th>
                    <th>Cash</th>
                  </tr>
                </thead>
                <tbody>
                  {model.events
                    .filter((e) => e.portfolio.bar_index <= marketEnd)
                    .slice(-100)
                    .reverse()
                    .map((e) => (
                      <tr key={e.decision.decision_bar_index}>
                        <td>
                          <button
                            className="bench-button"
                            onClick={() => seek(model.events.indexOf(e))}
                          >
                            {replay.market[e.decision.decision_bar_index]?.date}
                          </button>
                        </td>
                        <td>{humanLabel(e.decision.action)}</td>
                        <td>{numberText(e.execution?.fill_price)}</td>
                        <td>{numberText(e.execution?.fee)}</td>
                        <td>{numberText(e.portfolio.equity, 2)}</td>
                        <td>{numberText(e.portfolio.cash, 2)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </details>
          {cursor === length && (
            <p className="muted">
              Recorded run ended: {model.state}. A risk pause is reported, not
              silently bypassed.
            </p>
          )}
        </>
      )}
      <p className="muted">
        For the original AI competition and manual simulator,{" "}
        <Link href="/replay">open legacy replay</Link>.
      </p>
    </section>
  );
}
