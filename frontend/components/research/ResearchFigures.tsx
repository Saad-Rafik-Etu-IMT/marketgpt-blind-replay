"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, pct } from "@/lib/research/study-api";
import { LinePlot, PlotFrame, plotColors, type Series } from "./ScientificPlot";

type Run = {
  key: string;
  model: string;
  variant: string;
  seed: number;
  f1: number;
  accuracy: number;
  curves: { step: number; trainingLoss: number; validationLoss: number }[];
};
export function ModelFigures({
  runs,
  seed,
  onSeed,
}: {
  runs: Run[];
  seed: number;
  onSeed: (seed: number) => void;
}) {
  const [variant, setVariant] = useState("no-position");
  const seeds = Array.from(new Set(runs.map((r) => r.seed))).sort(
    (a, b) => a - b,
  );
  const paired = seeds.flatMap((s) => {
    const base = runs.find(
      (r) =>
        r.model === "Transformer" && r.variant === "reference" && r.seed === s,
    );
    const other = runs.find(
      (r) => r.model === "Transformer" && r.variant === variant && r.seed === s,
    );
    return base && other ? [{ seed: s, base: base.f1, other: other.f1 }] : [];
  });
  const mean = paired.length
    ? paired.reduce((sum, p) => sum + p.base - p.other, 0) / paired.length
    : 0;
  const matched = runs.filter(
    (r) => r.variant === "reference" && r.seed === seed,
  );
  const xmax = Math.max(
    1,
    ...matched.flatMap((r) => r.curves.map((c) => c.step)),
  );
  const ymax = Math.max(
    0.001,
    ...matched.flatMap((r) =>
      r.curves.flatMap((c) => [c.trainingLoss, c.validationLoss]),
    ),
  );
  return (
    <div className="research-stack">
      <div className="study-controls">
        <label>
          Ablation
          <select value={variant} onChange={(e) => setVariant(e.target.value)}>
            <option value="no-position">Without positional information</option>
            <option value="no-dropout">Without dropout</option>
            <option value="no-decay">Without weight decay</option>
            <option value="fixed-lr">Fixed learning rate</option>
          </select>
        </label>
        <label>
          Comparison seed
          <select value={seed} onChange={(e) => onSeed(Number(e.target.value))}>
            {seeds.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>
      {paired.length ? (
        <>
          <LinePlot
            title="Paired Transformer ablation"
            series={paired.map((p, i) => ({
              name: `Seed ${p.seed}`,
              points: [
                { x: 0, y: p.base },
                { x: 1, y: p.other },
              ],
              dots: true,
              color: plotColors[i % plotColors.length],
            }))}
            xDomain={[0, 1]}
            yDomain={[0, 1]}
            formatX={(v) => (v === 0 ? "Reference" : variant)}
            xLabel="Same seed · same validation data"
            yLabel="Macro-F1"
          />
          <p>
            {paired.length === 1
              ? "One paired seed"
              : `${paired.length} paired seeds`}{" "}
            · mean reference minus ablation {mean.toFixed(3)} ·{" "}
            {seeds.length - paired.length} seed(s) without a complete pair.
          </p>
          <details>
            <summary>Exact paired values</summary>
            <table>
              <thead>
                <tr>
                  <th>Seed</th>
                  <th>Reference</th>
                  <th>{variant}</th>
                  <th>Difference</th>
                </tr>
              </thead>
              <tbody>
                {paired.map((p) => (
                  <tr key={p.seed}>
                    <td>{p.seed}</td>
                    <td>{p.base.toFixed(3)}</td>
                    <td>{p.other.toFixed(3)}</td>
                    <td>{(p.base - p.other).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      ) : (
        <p className="bench-notice">
          Paired ablation · no matched reference and {variant} runs yet. Enable
          ablations when training.
        </p>
      )}
      {matched.length > 0 && (
        <>
          <h3>Learning curves on shared axes</h3>
          <div className="comparison-figures">
            {matched.map((r) => {
              const best = r.curves.reduce(
                (a, b) => (b.validationLoss < a.validationLoss ? b : a),
                r.curves[0],
              );
              return (
                <div key={r.key}>
                  <LinePlot
                    title={`${r.model} training and validation loss`}
                    series={[
                      {
                        name: "Training",
                        points: r.curves.map((c) => ({
                          x: c.step,
                          y: c.trainingLoss,
                        })),
                      },
                      {
                        name: "Validation",
                        points: r.curves.map((c) => ({
                          x: c.step,
                          y: c.validationLoss,
                        })),
                        dashed: true,
                      },
                    ]}
                    markers={
                      best
                        ? [
                            {
                              x: best.step,
                              y: best.validationLoss,
                              color: "#d07822",
                              label: "Retained checkpoint",
                            },
                          ]
                        : []
                    }
                    xDomain={[1, xmax]}
                    yDomain={[0, ymax]}
                    xLabel="Epoch"
                    yLabel="Cross-entropy"
                  />
                  {best && (
                    <p>
                      Retained checkpoint: epoch {best.step} · validation loss{" "}
                      {best.validationLoss.toFixed(4)}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
          <LinePlot
            title="Reference model performance"
            series={matched.map((r, i) => ({
              name: r.model,
              points: [
                { x: 0, y: r.accuracy },
                { x: 1, y: r.f1 },
              ],
              dots: true,
              color: plotColors[i % plotColors.length],
            }))}
            xDomain={[0, 1]}
            yDomain={[0, 1]}
            formatX={(v) => (v === 0 ? "Accuracy" : "Macro-F1")}
            xLabel="Same seed and validation windows"
            formatY={pct}
          />
        </>
      )}
    </div>
  );
}

type Sample = { index: number; p: number; up: boolean };
export function ReliabilityPlot({
  samples,
  inspect,
}: {
  samples: Sample[];
  inspect: (i: number) => void;
}) {
  const bins = Array.from({ length: 5 }, (_, i) =>
    samples.filter((s) => Math.min(4, Math.floor(5 * s.p)) === i),
  );
  const values = bins.flatMap((b, i) =>
    b.length
      ? [
          {
            i,
            count: b.length,
            predicted: b.reduce((sum, s) => sum + s.p, 0) / b.length,
            observed: b.filter((s) => s.up).length / b.length,
          },
        ]
      : [],
  );
  const [selected, setSelected] = useState<number | null>(null);
  return (
    <div>
      <LinePlot
        title="Probability reliability diagram"
        series={[
          {
            name: "Perfect calibration",
            points: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ],
            dashed: true,
            color: "#819599",
          },
          {
            name: "Observed frequency",
            scatter: true,
            points: values.map((v) => ({ x: v.predicted, y: v.observed })),
            dots: true,
            color: "#148b82",
          },
        ]}
        xDomain={[0, 1]}
        yDomain={[0, 1]}
        formatX={pct}
        formatY={pct}
        xLabel="Mean predicted chance of rising"
        yLabel="Observed fraction of rising sessions"
      />
      <p className="muted">
        {samples.length} decisions in the selected period. Perfect calibration
        follows the diagonal.
      </p>
      <div className="calibration-bins">
        {bins.map((b, i) => (
          <button
            key={i}
            className="bench-button"
            disabled={!b.length}
            aria-pressed={selected === i}
            onClick={() => setSelected(i)}
          >
            {i * 20}–{(i + 1) * 20}% · {b.length}
          </button>
        ))}
      </div>
      {selected !== null && bins[selected].length > 0 && (
        <label>
          Inspect case in bin{" "}
          <select
            value=""
            onChange={(e) => {
              if (e.target.value !== "") inspect(Number(e.target.value));
            }}
          >
            <option value="">Choose a case</option>
            {bins[selected].map((s) => (
              <option key={s.index} value={s.index}>
                Case {s.index + 1} · {pct(s.p)} · {s.up ? "up" : "not up"}
              </option>
            ))}
          </select>
        </label>
      )}
      <details>
        <summary>Exact bin values</summary>
        <table>
          <thead>
            <tr>
              <th>Bin</th>
              <th>Count</th>
              <th>Predicted</th>
              <th>Observed</th>
            </tr>
          </thead>
          <tbody>
            {bins.map((b, i) => {
              const v = values.find((v) => v.i === i);
              return (
                <tr key={i}>
                  <td>
                    {i * 20}–{(i + 1) * 20}%
                  </td>
                  <td>{b.length}</td>
                  <td>{v ? pct(v.predicted) : "—"}</td>
                  <td>{v ? pct(v.observed) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </div>
  );
}

export function ConfusionPlot({ cells }: { cells: number[] }) {
  const max = Math.max(1, ...cells);
  return (
    <PlotFrame title="Confusion matrix heatmap" height={195}>
      <text x="420" y="28" fill="var(--fg)" textAnchor="middle" fontSize="12">
        Predicted class
      </text>
      {["Not up", "Up"].map((label, i) => (
        <g key={label}>
          <text
            x={308 + i * 225}
            y="45"
            textAnchor="middle"
            fontSize="12"
            fill="var(--muted)"
          >
            {label}
          </text>
          <text
            x="177"
            y={86 + i * 58}
            textAnchor="end"
            fontSize="12"
            fill="var(--muted)"
          >
            {label}
          </text>
        </g>
      ))}
      <text
        transform="translate(60 110) rotate(-90)"
        textAnchor="middle"
        fontSize="12"
        fill="var(--fg)"
      >
        Actual class
      </text>
      {cells.map((v, i) => (
        <g key={i}>
          <rect
            x={198 + (i % 2) * 225}
            y={55 + Math.floor(i / 2) * 58}
            width="220"
            height="54"
            rx="4"
            fill={i % 3 === 0 ? "#148b82" : "#d07822"}
            opacity={0.15 + (0.65 * v) / max}
          />
          <text
            x={308 + (i % 2) * 225}
            y={88 + Math.floor(i / 2) * 58}
            fontWeight="bold"
            fontSize="18"
            fill="var(--fg)"
            textAnchor="middle"
          >
            {v}
          </text>
        </g>
      ))}
    </PlotFrame>
  );
}

export function SensitivityPlot({ values }: { values: number[] }) {
  const max = Math.max(0.000001, ...values.map(Math.abs));
  return (
    <PlotFrame title="Occlusion sensitivity by eight-return block" height={128}>
      {values.map((v, i) => {
        const delta = -100 * v;
        return (
          <g key={i}>
            <rect
              x={45 + i * 78}
              y="27"
              width="76"
              height="48"
              rx="3"
              fill={delta >= 0 ? "#148b82" : "#c95680"}
              opacity={0.1 + (0.65 * Math.abs(v)) / max}
            />
            <text
              x={83 + i * 78}
              y="55"
              textAnchor="middle"
              fill="var(--fg)"
              fontSize="13"
              fontWeight="bold"
            >
              {delta > 0 ? "+" : ""}
              {delta.toFixed(1)}
            </text>
            <text
              x={83 + i * 78}
              y="91"
              fill="var(--muted)"
              textAnchor="middle"
              fontSize="11"
            >
              {8 * i - 63}…{8 * i - 56}
            </text>
          </g>
        );
      })}
      <text
        x="350"
        y="115"
        textAnchor="middle"
        fontSize="11"
        fill="var(--muted)"
      >
        Change in P(up), percentage points · offsets oldest → newest
      </text>
    </PlotFrame>
  );
}

type Timeline = {
  dataset: string;
  split: {
    train_end: string;
    validation_start: string;
    validation_end: string;
    test_start: string;
    embargo: number;
    label_horizon: number;
  };
  points: { date: string; split: string; value: number }[];
  first: string;
  last: string;
  gapBars: number;
};
export function DatasetFigures({
  study,
  dataset,
}: {
  study: string;
  dataset: string;
}) {
  const query = useQuery({
    queryKey: ["timeline", study],
    queryFn: () => api<Timeline>(`/${study}/timeline`),
    retry: false,
  });
  const t = query.data;
  if (!t)
    return (
      <p className="muted">
        {query.error
          ? "This saved study has no recorded timeline. Train a new study to inspect development history."
          : "Loading data provenance…"}
      </p>
    );
  if (t.dataset !== dataset)
    return (
      <p role="alert">Timeline fingerprint does not match the selected run.</p>
    );
  const splits = ["train", "gap", "validation"];
  const series: Series[] = splits.map((split, i) => ({
    name: split,
    points: t.points
      .filter((p) => p.split === split)
      .map((p) => ({ x: Date.parse(p.date), y: p.value })),
    color: plotColors[i],
  }));
  const scale = Math.max(0.001, ...t.points.map((p) => Math.abs(p.value)));
  const hist = Array.from(
    { length: 20 },
    (_, i) => -scale + ((i + 0.5) * scale) / 10,
  );
  const distribution = ["train", "validation"].map((split, i) => {
    const rows = t.points.filter((p) => p.split === split);
    const counts = Array(20).fill(0) as number[];
    rows.forEach(
      (p) =>
        counts[
          Math.min(
            19,
            Math.max(0, Math.floor(((p.value + scale) / scale) * 10)),
          )
        ]++,
    );
    return {
      name: split,
      color: plotColors[i === 0 ? 0 : 2],
      points: hist.map((x, j) => ({
        x,
        y: rows.length ? counts[j] / rows.length : 0,
      })),
    };
  });
  return (
    <>
      <div className="split-timeline">
        <div>
          <b>Training</b>
          <span>
            {t.first} → {t.split.train_end}
          </span>
        </div>
        <div>
          <b>Gap</b>
          <span>{t.gapBars} observed bars</span>
        </div>
        <div>
          <b>Validation</b>
          <span>
            {t.split.validation_start} → {t.split.validation_end}
          </span>
          <small>Data through {t.last}</small>
        </div>
        <div>
          <b>Sealed test</b>
          <span>From {t.split.test_start}</span>
          <small>No test values shown</small>
        </div>
      </div>
      <p className="muted">
        Required embargo: {t.split.embargo} bars plus {t.split.label_horizon}{" "}
        label bar. The first gap is counted from observed dates; the test gap
        uses configured dates only.
      </p>
      <LinePlot
        title="Chronological log returns through validation"
        series={series}
        formatX={(v) => new Date(v).toISOString().slice(0, 10)}
        formatY={pct}
        xLabel="Observation date"
        yLabel="Daily log return"
      />
      <LinePlot
        title="Return distribution by partition"
        series={distribution}
        formatX={pct}
        formatY={pct}
        xLabel="Log return (20 equal-width bins)"
        yLabel="Share of partition"
      />
    </>
  );
}
