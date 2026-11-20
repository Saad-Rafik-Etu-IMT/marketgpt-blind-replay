"use client";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { laboratoryApi, numberText, type Run } from "@/lib/research/laboratory";
import {
  confusion,
  matchCases,
  type PredictionCase,
} from "@/lib/research/prediction-analysis";
import { humanLabel, runLabel } from "@/lib/research/product-context";
import { LinePlot, PlotFrame, plotColors } from "./ScientificPlot";

type CaseReport = {
  cases: PredictionCase[];
  dataset: string;
  run: string;
  target: string;
  part: string;
};
type Inspection = {
  index: number;
  probability: number;
  reference: number;
  sequence: boolean;
  method: string;
  features: { column: string; raw: number; scaled: number; delta: number }[];
  counterfactual?: { column: string; replacement: number; probability: number };
  activations?: { layer: number; values: number[]; width: number }[];
};
const pct = (v: number | null | undefined) =>
  v == null ? "—" : numberText(v * 100, 1) + "%";
function ContributionWaterfall({ data }: { data: Inspection }) {
  const first = data.features.slice(0, 10);
  const steps = [
    ...first.map((f) => ({ name: f.column, delta: f.delta })),
    ...(data.features.length > 10
      ? [
          {
            name: "Remaining columns (net)",
            delta: data.features.slice(10).reduce((n, f) => n + f.delta, 0),
          },
        ]
      : []),
  ];
  let from = data.reference;
  const bars = steps.map((s) => {
    const start = from;
    from += s.delta;
    return { ...s, start, end: from };
  });
  const x = (v: number) => 175 + 440 * v,
    height = 80 + bars.length * 20;
  return (
    <PlotFrame
      title="Sequential feature contribution waterfall · probability change"
      height={height}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line
            x1={x(v)}
            x2={x(v)}
            y1="30"
            y2={height - 27}
            stroke="var(--line)"
          />
          <text
            x={x(v)}
            y="25"
            textAnchor="middle"
            fontSize="10"
            fill="var(--muted)"
          >
            {v * 100}%
          </text>
        </g>
      ))}
      {bars.map((b, i) => (
        <g key={b.name}>
          <text
            x="165"
            y={49 + i * 20}
            fontSize="10"
            textAnchor="end"
            fill="var(--fg)"
          >
            {b.name.slice(0, 26)}
          </text>
          <rect
            x={x(Math.min(b.start, b.end))}
            y={38 + i * 20}
            width={Math.max(1, 440 * Math.abs(b.delta))}
            height="14"
            fill={b.delta < 0 ? "#b95061" : plotColors[0]}
          >
            <title>
              {b.name}: {pct(b.start)} → {pct(b.end)}
            </title>
          </rect>
          <text x="621" y={49 + i * 20} fontSize="10" fill="var(--fg)">
            {(b.delta * 100).toFixed(2)} pp
          </text>
        </g>
      ))}
      <text
        x="350"
        y={height - 8}
        textAnchor="middle"
        fontSize="10"
        fill="var(--muted)"
      >
        Reference {pct(data.reference)} → actual {pct(data.probability)}
      </text>
    </PlotFrame>
  );
}
export function PredictionExplorer({
  id,
  run,
  part,
  target,
  runs = [],
  onReplay,
  initialDate,
}: {
  id: string;
  run: string;
  part: string;
  target: string;
  runs?: Run[];
  onReplay?: (row: PredictionCase) => void;
  initialDate?: string;
}) {
  const [open, setOpen] = useState(false);
  const [threshold, setThreshold] = useState(0.5),
    [cell, setCell] = useState<number | null>(null);
  const [asset, setAsset] = useState(""),
    [regime, setRegime] = useState(""),
    [sort, setSort] = useState("date"),
    [page, setPage] = useState(0);
  const [selected, setSelected] = useState<number | null>(null),
    [candidate, setCandidate] = useState("");
  const [column, setColumn] = useState(0),
    [replacement, setReplacement] = useState(0),
    [scenario, setScenario] = useState<{
      column: number;
      replacement: number;
    } | null>(null);
  const suffix = `?part=${part}&target=${encodeURIComponent(target)}`;
  const path = `/${id}/runs/${run}/predictions`;
  const report = useQuery({
    queryKey: ["prediction-cases", id, run, part, target],
    enabled: open && !!run,
    retry: false,
    queryFn: () => laboratoryApi<CaseReport>(path + suffix),
  });
  const comparison = useQuery({
    queryKey: ["prediction-cases", id, candidate, part, target],
    enabled: open && !!candidate,
    retry: false,
    queryFn: () =>
      laboratoryApi<CaseReport>(
        `/${id}/runs/${candidate}/predictions` + suffix,
      ),
  });
  const cases = useMemo(() => report.data?.cases ?? [], [report.data]);
  const scope = useMemo(
    () =>
      cases.filter(
        (r) =>
          (!asset || r.asset === asset) &&
          (!regime || r.volatility === regime || r.trend === regime),
      ),
    [cases, asset, regime],
  );
  const stats = confusion(scope, threshold);
  const filtered = scope
    .filter(
      (r) =>
        cell === null ||
        r.actual * 2 + Number(r.probability >= threshold) === cell,
    )
    .sort((a, b) =>
      sort === "wrong"
        ? Math.abs(b.actual - b.probability) -
          Math.abs(a.actual - a.probability)
        : sort === "confidence"
          ? Math.abs(b.probability - 0.5) - Math.abs(a.probability - 0.5)
          : a.date.localeCompare(b.date) || a.asset.localeCompare(b.asset),
    );
  const picked =
    cases.find((r) => r.index === selected) ??
    (initialDate ? cases.find((r) => r.date === initialDate) : undefined);
  const detail = useQuery({
    queryKey: [
      "prediction-detail",
      id,
      run,
      part,
      target,
      picked?.index,
      scenario,
    ],
    enabled: open && !!picked,
    retry: false,
    queryFn: () =>
      laboratoryApi<Inspection>(
        path +
          "/" +
          picked!.index +
          suffix +
          (scenario
            ? `&column=${scenario.column}&replacement=${scenario.replacement}`
            : ""),
      ),
  });
  const explanation = useQuery({
    queryKey: ["prediction-explanation", id, run, part, target, picked?.index],
    enabled: false,
    retry: false,
    queryFn: () =>
      laboratoryApi<{ source: string; facts: string[]; check: string }>(
        path + "/" + picked!.index + "/explain" + suffix,
        {},
      ),
  });
  const paired = matchCases(scope, comparison.data?.cases ?? []);
  const pairedRow = picked
    ? paired.find((r) => r.first.index === picked.index)
    : undefined;
  const histogram = Array.from({ length: 10 }, (_, i) => ({
    x: (i + 0.5) / 10,
    y: scope.filter((r) => Math.min(9, Math.floor(r.probability * 10)) === i)
      .length,
  }));
  const dates = [...new Set(scope.map((r) => r.date))].sort();
  const dateStats = new Map<string, { correct: number; total: number }>();
  for (const r of scope) {
    const d = dateStats.get(r.date) ?? { correct: 0, total: 0 };
    d.total++;
    d.correct += Number(Number(r.probability >= threshold) === r.actual);
    dateStats.set(r.date, d);
  }
  const rolling = dates.flatMap((d, x) => {
    if (x < 19) return [];
    const ds = dates.slice(x - 19, x + 1).map((k) => dateStats.get(k)!);
    return [
      {
        x,
        y:
          ds.reduce((n, v) => n + v.correct, 0) /
          ds.reduce((n, v) => n + v.total, 0),
      },
    ];
  });
  function choose(row: PredictionCase) {
    setSelected(row.index);
    setScenario(null);
    setColumn(0);
  }
  return (
    <section className="prediction-explorer supporting-details">
      <h3>Interactive prediction laboratory · selected registered seed</h3>
      <p className="muted">
        Exact checkpoint decisions, not seed-ensemble metrics. Threshold changes
        are exploratory only and never change the registered policy or final
        test.
      </p>
      <button
        className="bench-button"
        disabled={!run}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Close prediction explorer" : "Open prediction explorer"}
      </button>
      {open && (
        <>
          {report.isPending ? (
            <p>Loading registered predictions…</p>
          ) : report.error ? (
            <p role="alert">{String(report.error)}</p>
          ) : (
            <>
              <div className="context-fields">
                <label>
                  Decision threshold {pct(threshold)}
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={threshold}
                    onChange={(e) => {
                      setThreshold(Number(e.target.value));
                      setPage(0);
                    }}
                  />
                </label>
                <label>
                  Asset
                  <select
                    value={asset}
                    onChange={(e) => {
                      setAsset(e.target.value);
                      setPage(0);
                    }}
                  >
                    <option value="">All assets</option>
                    {[...new Set(cases.map((r) => r.asset))].map((a) => (
                      <option key={a}>{a}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Regime
                  <select
                    value={regime}
                    onChange={(e) => {
                      setRegime(e.target.value);
                      setPage(0);
                    }}
                  >
                    <option value="">All regimes</option>
                    {[
                      ...new Set(cases.flatMap((r) => [r.volatility, r.trend])),
                    ].map((a) => (
                      <option key={a} value={a}>
                        {humanLabel(a)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Compare another checkpoint
                  <select
                    value={candidate}
                    onChange={(e) => setCandidate(e.target.value)}
                  >
                    <option value="">None</option>
                    {runs
                      .filter((r) => r.state === "complete" && r.key !== run)
                      .map((r) => (
                        <option key={r.key} value={r.key}>
                          {runLabel(r)}
                        </option>
                      ))}
                  </select>
                </label>
              </div>
              <div className="metric-strip">
                <div>
                  <span>Accuracy</span>
                  <strong>{pct(stats.accuracy)}</strong>
                </div>
                <div>
                  <span>Precision</span>
                  <strong>{pct(stats.precision)}</strong>
                </div>
                <div>
                  <span>Recall</span>
                  <strong>{pct(stats.recall)}</strong>
                </div>
                <div>
                  <span>Observations</span>
                  <strong>{scope.length}</strong>
                </div>
              </div>
              <div className="lab-two">
                <div>
                  <h4>Confusion matrix · actual × predicted</h4>
                  <div className="case-confusion">
                    {stats.cells.map((n, i) => (
                      <button
                        className="bench-button"
                        key={i}
                        aria-pressed={cell === i}
                        onClick={() => {
                          setCell(cell === i ? null : i);
                          setPage(0);
                        }}
                      >
                        <span>
                          {
                            [
                              "True negative",
                              "False positive",
                              "False negative",
                              "True positive",
                            ][i]
                          }
                        </span>
                        <strong>{n}</strong>
                      </button>
                    ))}
                  </div>
                  <button
                    className="bench-button"
                    onClick={() => {
                      setCell(null);
                      setPage(0);
                    }}
                  >
                    All prediction cases
                  </button>
                </div>
                <LinePlot
                  title="Predicted positive-class probability histogram"
                  xLabel="Probability"
                  yLabel="Count"
                  xDomain={[0, 1]}
                  series={[{ name: "Count", points: histogram, dots: true }]}
                />
              </div>
              <LinePlot
                title="Rolling accuracy · 20 observed dates"
                xLabel="Session"
                yLabel="Accuracy"
                yDomain={[0, 1]}
                formatX={(v) => dates[Math.round(v)] ?? ""}
                series={[{ name: "Accuracy", points: rolling }]}
              />
              {comparison.error && (
                <p role="alert">{String(comparison.error)}</p>
              )}
              {comparison.data && (
                <p className="bench-notice">
                  Matched {paired.length} identical asset/date/label
                  observations; disagreement at current threshold:{" "}
                  {
                    paired.filter(
                      (r) =>
                        r.first.probability >= threshold !==
                        r.second.probability >= threshold,
                    ).length
                  }
                  . Unmatched rows excluded.
                </p>
              )}
              <label className="lab-field">
                Browse cases
                <select
                  value={sort}
                  onChange={(e) => {
                    setSort(e.target.value);
                    setPage(0);
                  }}
                >
                  <option value="date">Chronological</option>
                  <option value="wrong">Highest prediction error</option>
                  <option value="confidence">Highest confidence</option>
                </select>
              </label>
              <div className="lab-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Session</th>
                      <th>Asset</th>
                      <th>Probability</th>
                      <th>Actual</th>
                      <th>Regime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.slice(page * 20, page * 20 + 20).map((r) => (
                      <tr key={r.index}>
                        <td>
                          <button
                            className="bench-button"
                            aria-pressed={picked?.index === r.index}
                            onClick={() => choose(r)}
                          >
                            {r.date}
                          </button>
                        </td>
                        <td>{r.asset}</td>
                        <td>{pct(r.probability)}</td>
                        <td>{r.actual === 1 ? "Positive" : "Negative"}</td>
                        <td>{humanLabel(r.volatility)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="bench-actions">
                <button
                  className="bench-button"
                  disabled={!page}
                  onClick={() => setPage((v) => v - 1)}
                >
                  Previous
                </button>
                <span>
                  {filtered.length
                    ? Math.min(page * 20 + 1, filtered.length)
                    : 0}
                  –{Math.min((page + 1) * 20, filtered.length)} /{" "}
                  {filtered.length}
                </span>
                <button
                  className="bench-button"
                  disabled={(page + 1) * 20 >= filtered.length}
                  onClick={() => setPage((v) => v + 1)}
                >
                  Next
                </button>
              </div>
              {picked && (
                <article className="finding-card">
                  <h4>
                    {picked.asset} · {picked.date}
                  </h4>
                  <p>
                    Prediction {pct(picked.probability)} · observed label{" "}
                    {picked.actual === 1 ? "Positive" : "Negative"}
                    {pairedRow &&
                      ` · comparison checkpoint ${pct(pairedRow.second.probability)}`}
                  </p>
                  {onReplay && target === "direction_binary_v1" && (
                    <button
                      className="bench-button primary-action"
                      onClick={() => onReplay(picked)}
                    >
                      Open this historical session in replay
                    </button>
                  )}
                  {detail.error && <p role="alert">{String(detail.error)}</p>}
                  {detail.isPending && (
                    <p>Measuring checkpoint feature contributions…</p>
                  )}
                  {detail.data && (
                    <>
                      <p>
                        {detail.data.method} Reference{" "}
                        {pct(detail.data.reference)} → measured prediction{" "}
                        {pct(detail.data.probability)}.
                      </p>
                      <ContributionWaterfall data={detail.data} />
                      <button
                        className="bench-button"
                        disabled={explanation.isFetching}
                        onClick={() => void explanation.refetch()}
                      >
                        Explain measured evidence with local LLM
                      </button>
                      {explanation.error && (
                        <p role="alert">{String(explanation.error)}</p>
                      )}
                      {explanation.data && (
                        <div className="finding-card">
                          <strong>{explanation.data.source}</strong>
                          <p className="muted">{explanation.data.check}</p>
                          <ul>
                            {explanation.data.facts.map((f, i) => (
                              <li key={i}>{f}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div className="context-fields">
                        <label>
                          Counterfactual input
                          <select
                            value={column}
                            onChange={(e) => setColumn(Number(e.target.value))}
                          >
                            {detail.data.features.map((f, i) => (
                              <option key={f.column} value={i}>
                                {f.column}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Replacement on model-input scale: {replacement}
                          <input
                            type="range"
                            min={-3}
                            max={3}
                            step={0.1}
                            value={replacement}
                            onChange={(e) =>
                              setReplacement(Number(e.target.value))
                            }
                          />
                        </label>
                        <button
                          className="bench-button"
                          onClick={() => setScenario({ column, replacement })}
                        >
                          Measure counterfactual
                        </button>
                      </div>
                      {detail.data.counterfactual && (
                        <p>
                          Measured probability:{" "}
                          {pct(detail.data.counterfactual.probability)} ·
                          replaces {detail.data.counterfactual.column} across
                          the input window. Not an attainable market scenario or
                          causal claim.
                        </p>
                      )}
                      <details>
                        <summary>
                          Exact inputs ·{" "}
                          {detail.data.sequence
                            ? "last token shown"
                            : "recorded feature vector"}
                        </summary>
                        <div className="lab-table-scroll">
                          <table>
                            <thead>
                              <tr>
                                <th>Column</th>
                                <th>Raw</th>
                                <th>Scaled</th>
                                <th>Contribution (pp)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {detail.data.features.map((f) => (
                                <tr key={f.column}>
                                  <td>{f.column}</td>
                                  <td>{numberText(f.raw)}</td>
                                  <td>{numberText(f.scaled)}</td>
                                  <td>{numberText(f.delta * 100)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </details>
                      {detail.data.activations?.length ? (
                        <details>
                          <summary>Measured MLP neuron activations</summary>
                          {detail.data.activations.map((l) => (
                            <LinePlot
                              key={l.layer}
                              title={`Layer ${l.layer} · first ${l.values.length} of ${l.width} activations`}
                              xLabel="Neuron"
                              series={[
                                {
                                  name: "Activation",
                                  points: l.values.map((y, x) => ({ x, y })),
                                  dots: true,
                                  scatter: true,
                                },
                              ]}
                            />
                          ))}
                        </details>
                      ) : null}
                    </>
                  )}
                </article>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
