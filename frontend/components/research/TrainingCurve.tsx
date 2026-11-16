import type { ResearchRun } from "@/lib/research/contract";

export function TrainingCurve({ run }: { run: ResearchRun }) {
  const points = run.curves;
  const values = points.flatMap(p => [p.trainingLoss, p.validationLoss]).filter((v): v is number => v !== null);
  const max = Math.max(1, ...values);
  const first = points[0]?.step ?? 0;
  const span = Math.max(1, (points.at(-1)?.step ?? 1) - first);
  const x = (step: number) => 60 + (step - first) / span * 590;
  const y = (value: number) => 190 - value / max * 155;
  const tickStride = Math.max(1, Math.ceil(points.length / 6));
  function path(key: "trainingLoss" | "validationLoss") {
    let previous = false;
    return points.map(p => {
      const value = p[key];
      if (value === null) { previous = false; return ""; }
      const segment = `${previous ? "L" : "M"}${x(p.step)},${y(value)}`;
      previous = true;
      return segment;
    }).join(" ");
  }
  return <section className="bench-panel" aria-labelledby="curve-title">
    <div className="section-heading"><div><p className="eyebrow">Learning trace</p><h2 id="curve-title">Training & validation loss</h2></div><span className="bench-badge">{run.lossName}</span></div>
    {values.length ? <>
      <div className="chart-scroll" tabIndex={0} role="region" aria-label="Training curve, scroll horizontally on narrow screens">
      <svg className="research-chart" viewBox="0 0 700 240" role="img" aria-label={`${run.lossName} against training step. Solid line: training. Dashed line: validation. Exact values in the table below.`}>
        {[0, max / 2, max].map(v => <g key={v}><line x1="60" x2="650" y1={y(v)} y2={y(v)} stroke="var(--line)" /><text x="48" y={y(v) + 4} textAnchor="end">{v.toFixed(2)}</text></g>)}
        <path d={path("trainingLoss")} fill="none" stroke="var(--accent)" strokeWidth="3" />
        <path d={path("validationLoss")} fill="none" stroke="var(--fg)" strokeWidth="2" strokeDasharray="7 5" />
        {points.map((p, i) => <g key={p.step}>
          {(i % tickStride === 0 || i === points.length - 1) && <text x={x(p.step)} y="213" textAnchor="middle">{p.step}</text>}
          {(["trainingLoss", "validationLoss"] as const).map(key => p[key] !== null && <circle key={key} cx={x(p.step)} cy={y(p[key])} r="4" fill={key === "trainingLoss" ? "var(--accent)" : "var(--fg)"}><title>{`Step ${p.step} · ${key}: ${p[key]}`}</title></circle>)}
        </g>)}
        <text x="355" y="237" textAnchor="middle">Training step</text>
      </svg>
      </div>
      <p className="muted">Solid blue: training · Dashed neutral: validation. Missing values break the curve; they are not zero. These are not held-out test metrics. On narrow screens, scroll the chart or open the values below.</p>
      <details><summary>Loss values (accessible table)</summary><div className="table-scroll" tabIndex={0} role="region" aria-label="Loss values"><table><thead><tr><th>Step</th><th>Training</th><th>Validation</th></tr></thead><tbody>{points.map(p => <tr key={p.step}><td>{p.step}</td><td>{p.trainingLoss ?? "Unavailable"}</td><td>{p.validationLoss ?? "Unavailable"}</td></tr>)}</tbody></table></div></details>
    </> : <p className="muted">Training curves unavailable in this artifact.</p>}
  </section>;
}
