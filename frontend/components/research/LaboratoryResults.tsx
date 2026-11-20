"use client";

import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import {
  Evaluation,
  EvaluationEntry,
  laboratoryApi,
  labelText,
  numberText,
  Replay,
} from "@/lib/research/laboratory";
import { LinePlot, plotColors } from "./ScientificPlot";
import { ConfusionPlot } from "./ResearchFigures";

export function EvaluationView({
  id,
  part,
  changePart,
  entry,
  changeEntry,
  explanation = false,
}: {
  id: string;
  part: string;
  changePart: (part: string) => void;
  entry: number;
  changeEntry: (index: number) => void;
  explanation?: boolean;
}) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const query = useQuery({
    queryKey: ["lab-evaluation", id, part],
    retry: false,
    queryFn: () => laboratoryApi<Evaluation>(`/${id}/evaluation/${part}`),
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
  const selected = report?.results[entry] ?? report?.results[0];
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
        <label>
          Evaluation partition{" "}
          <select
            value={part}
            onChange={(e) => {
              changePart(e.target.value);
              changeEntry(0);
            }}
          >
            <option value="inner_validation">
              Inner validation · development
            </option>
            <option value="validation">
              Historical validation · recorded read
            </option>
          </select>
        </label>
      </div>
      <p className="muted">
        Development results are not final test evidence. Negative or
        inconclusive outcomes are valid.
      </p>
      <button
        className="bench-button primary-action"
        disabled={busy}
        onClick={() => void evaluate()}
      >
        {busy
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
      {selected && report && (
        <>
          <label className="lab-field">
            Configuration and target
            <select
              value={report.results.indexOf(selected)}
              onChange={(e) => changeEntry(Number(e.target.value))}
            >
              {report.results.map((r, i) => (
                <option key={i} value={i}>
                  {r.feature_profile} · {r.preset} · {r.target}
                </option>
              ))}
            </select>
          </label>
          {explanation ? (
            <Importance entry={selected} />
          ) : (
            <EvaluationFigures entry={selected} />
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
                          {labelText(r.feature_profile)} / {r.preset}
                        </td>
                        <td>{r.target}</td>
                        <td>{numberText(r.comparison.mean_difference)}</td>
                        <td>{r.category}</td>
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
  const baseline = e.baselines[e.strongest_baseline];
  const names = ["Model ensemble", ...Object.keys(e.baselines)];
  const losses = [
    e.ensemble.loss,
    ...Object.values(e.baselines).map((x) => x.loss),
  ];
  const comparison = baseline?.comparison;
  return (
    <>
      <div className="metric-strip">
        <div>
          <span>Evaluated decisions</span>
          <strong>{e.rows}</strong>
        </div>
        <div>
          <span>Seeds</span>
          <strong>{e.seeds.length}</strong>
        </div>
        <div>
          <span>Model loss</span>
          <strong>{numberText(e.ensemble.loss)}</strong>
        </div>
        <div>
          <span>Verdict vs strongest baseline</span>
          <strong>{comparison?.verdict ?? "—"}</strong>
        </div>
      </div>
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
    </>
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
}: {
  id: string;
  run: string;
  part: string;
}) {
  const [cursor, setCursor] = useState(0),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [explanation, setExplanation] = useState<{
    source: string;
    facts: string[];
  }>();
  const explanationRequest = useRef(0);
  const path = `/${id}/runs/${run}/replay?part=${part}`;
  const query = useQuery({
    queryKey: ["lab-replay", id, run, part],
    enabled: !!run,
    retry: false,
    queryFn: () => laboratoryApi<Replay>(path),
  });
  async function generate() {
    setBusy(true);
    setError("");
    try {
      await laboratoryApi<Replay>(path, {});
      setCursor(0);
      await query.refetch();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const replay = query.data;
  const model = replay?.players[0];
  const length = model?.events.length ?? 0;
  const index = Math.min(cursor, Math.max(0, length - 1));
  const event = model?.events[index];
  const marketEnd =
    cursor < length
      ? (event?.decision.decision_bar_index ?? 0)
      : (model?.events.at(-1)?.portfolio.bar_index ??
        replay?.first_decision ??
        0);
  async function explain(generate: boolean) {
    const request = ++explanationRequest.current;
    try {
      const value = await laboratoryApi<{ source: string; facts: string[] }>(
        `/${id}/runs/${run}/replay/explain?part=${part}&index=${index}&outcome=${cursor === length}&generate=${generate}`,
        {},
      );
      if (request === explanationRequest.current) setExplanation(value);
    } catch (e) {
      setError(String(e));
    }
  }
  function seek(value: number) {
    explanationRequest.current++;
    setCursor(value);
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
      {error && <p role="alert">{error}</p>}
      {replay && model && (
        <>
          <p className="bench-notice">
            {replay.asset} · {replay.accounting} · {replay.label}. Recorded
            predictions are disclosed at their decision close, while the
            next-close portfolio is disclosed after execution.
          </p>
          <div className="bench-actions">
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
              disabled={cursor === length}
              onClick={() => seek(cursor + 1)}
            >
              Reveal next session
            </button>
            <label>
              Revealed decisions: {cursor}/{length}
              <input
                type="range"
                min={0}
                max={length}
                value={cursor}
                onChange={(e) => seek(Number(e.target.value))}
              />
            </label>
          </div>
          <div className="metric-strip">
            {replay.players.map((p) => {
              const visible = p.events.filter(
                (x) => x.portfolio.bar_index <= marketEnd,
              );
              return (
                <div key={p.name}>
                  <span>{labelText(p.name)}</span>
                  <strong>
                    {numberText(
                      visible.at(-1)?.portfolio.equity ?? p.initial.equity,
                      2,
                    )}
                  </strong>
                </div>
              );
            })}
          </div>
          <div className="lab-two">
            <LinePlot
              title="Revealed market prices and fills"
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
            <LinePlot
              title="Marked equity · same market clock"
              xLabel="Market bar"
              yLabel="Virtual currency"
              series={replay.players.map((p, i) => ({
                name: labelText(p.name),
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
          <div className="lab-two">
            <LinePlot
              title="Cumulative P&L"
              xLabel="Market bar"
              yLabel="Virtual currency"
              series={replay.players.map((p, i) => ({
                name: labelText(p.name),
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
              xLabel="Market bar"
              yLabel="Fraction of peak"
              series={replay.players.map((p, i) => {
                let peak = p.initial.equity;
                return {
                  name: labelText(p.name),
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
          {cursor === length && (
            <p className="muted">
              Recorded run ended: {model.state}. A risk pause is reported, not
              silently bypassed.
            </p>
          )}
          {event && (
            <article className="bench-panel">
              <h3>
                {cursor === length
                  ? "Last revealed decision"
                  : "Current blind decision"}
              </h3>
              <div className="metric-strip">
                <div>
                  <span>Probability of positive executed return</span>
                  <strong>
                    {numberText(
                      (event.decision.probabilities?.long ?? 0) * 100,
                      1,
                    )}
                    %
                  </strong>
                </div>
                <div>
                  <span>Policy target</span>
                  <strong>{event.decision.target}</strong>
                </div>
                <div>
                  <span>Order action</span>
                  <strong>{labelText(event.decision.action)}</strong>
                </div>
              </div>
              <div className="bench-actions">
                <button
                  className="bench-button"
                  onClick={() => void explain(false)}
                >
                  Measured facts
                </button>
                <button
                  className="bench-button"
                  onClick={() => void explain(true)}
                >
                  Explain with local LLM
                </button>
              </div>
              {explanation && (
                <div>
                  <p>{explanation.source}</p>
                  <ul>
                    {explanation.facts.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </article>
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
