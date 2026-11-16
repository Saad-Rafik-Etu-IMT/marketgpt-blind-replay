"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { loadResearchInventory } from "@/lib/research/api-adapter";
import { useResearch } from "./ResearchProvider";
import { TrainingCurve } from "./TrainingCurve";

function Empty({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="bench-panel empty-state"><p className="eyebrow">Integration boundary</p><h2>{title}</h2><p className="muted">{children}</p></section>;
}
function SelectionError() {
  return <p className="bench-notice" role="alert">Selection unavailable or incompatible with this bundle. Choose a run or sample below. Shared links require the same artifact to be loaded; imports disappear on reload.</p>;
}
function Inventory() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["research", "instrument-inventory"], queryFn: loadResearchInventory });
  return <section className="bench-panel">
    <p className="eyebrow">Real API · operator metadata</p><h2>Available market history</h2>
    <p className="muted">Loaded instruments only, not model inputs or evidence of a trained run. No price history or test distributions are fetched here.</p>
    {isPending ? <p role="status">Loading instrument inventory…</p> : isError ? <p role="alert">Instrument inventory unavailable. <button className="bench-button" onClick={() => void refetch()}>Retry inventory</button></p> : !data?.length ? <p>No instruments loaded. Use the existing Replay upload workflow.</p> :
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Loaded instruments"><table><thead><tr><th>Instrument</th><th>Bars</th><th>First date</th><th>Last date</th></tr></thead><tbody>{data.map(i => <tr key={i.instrument}><td>{i.instrument}</td><td>{i.bar_count}</td><td>{i.first_date ?? "Unavailable"}</td><td>{i.last_date ?? "Unavailable"}</td></tr>)}</tbody></table></div>}
  </section>;
}
export function Experiments() {
  const { bundle } = useResearch();
  const params = useSearchParams();
  const requested = params.get("run");
  const run = bundle?.runs.find(r => r.id === requested) ?? (!requested ? bundle?.runs[0] : undefined);
  if (!bundle) return <div className="research-stack"><Empty title="No research runs loaded">Training and evaluation exporters are not implemented. Import a version-1 research bundle or explicitly load the synthetic demo to explore the connected workflow.</Empty><Inventory /></div>;
  return <div className="research-stack">
    {requested && !run && <SelectionError />}
    <section className="bench-panel"><div className="section-heading"><div><p className="eyebrow">01 / Experiment</p><h2>Run ledger</h2></div><span className="bench-badge">{bundle.runs.length} loaded</span></div>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Research runs"><table><thead><tr><th>Run / model</th><th>Dataset / split</th><th>Seed</th><th>Checkpoint</th><th>Inspect</th></tr></thead><tbody>
        {bundle.runs.map(r => <tr key={r.id}><td><Link href={`/experiments?run=${encodeURIComponent(r.id)}`}>{r.id}</Link><small>{r.model}</small></td><td>{r.dataset}<small>{r.split}</small></td><td>{r.seed}</td><td>{r.checkpoint}</td><td><Link href={`/inspector?run=${encodeURIComponent(r.id)}`}>Inspect prediction →</Link></td></tr>)}
      </tbody></table></div>
      {!bundle.runs.length && <p>No runs in this bundle.</p>}
    </section>
    {run && <><TrainingCurve run={run} /><section className="bench-panel"><h2>Reproducibility boundary</h2><p>Selected run: <code>{run.id}</code> · checkpoint: <code>{run.checkpoint}</code></p><p className="muted">Artifact contract v1 · dataset and split identifiers are exporter declarations, not verified fingerprints. Code commit, preprocessing, tokenizer and recorded integrity checks are unavailable in this thin contract.</p></section></>}
  </div>;
}
export function Inspector() {
  const { bundle } = useResearch();
  const params = useSearchParams();
  const runId = params.get("run");
  const sampleId = params.get("sample");
  const samples = bundle?.samples.filter(s => !runId || s.runId === runId) ?? [];
  const sample = sampleId ? samples.find(s => s.id === sampleId) : samples[0];
  const run = bundle?.runs.find(r => r.id === sample?.runId);
  if (!bundle) return <Empty title="No saved prediction loaded">The backend has no model prediction or explanation service. Replay bars are not substituted for actual model inputs. Import a research bundle or explicitly load the synthetic demo.</Empty>;
  const explanation = sample?.explanation;
  return <div className="research-stack">
    <section className="bench-panel"><p className="eyebrow">02 / Decision-time snapshot</p><h2>Saved predictions</h2><div className="bench-actions">{samples.map(s => <Link className="bench-button" key={s.id} href={`/inspector?run=${encodeURIComponent(s.runId)}&sample=${encodeURIComponent(s.id)}`} aria-current={sample?.id === s.id ? "true" : undefined}>{s.id}</Link>)}</div>
      {(sampleId && !sample || runId && !bundle.runs.some(r => r.id === runId)) && <SelectionError />}
      {!samples.length && <p>No matching samples. <Link href="/experiments">Choose a loaded run</Link>.</p>}
    </section>
    {sample && run && <><p className="selection-line"><code>{run.id}</code> / {run.model} / <code>{run.checkpoint}</code></p>
      <div className="inspector-grid"><div className="research-stack">
        <section className="bench-panel"><div className="section-heading"><div><p className="eyebrow">Prediction · {sample.id}</p><h2>Model probabilities</h2></div><span className="bench-badge">Selected: {sample.predicted}</span></div>
          <div className="probability-grid">{(["long", "flat", "short"] as const).map(label => <div key={label}><span>P({label})</span><strong>{(sample.probabilities[label] * 100).toFixed(1)}%</strong><meter min={0} max={1} value={sample.probabilities[label]} aria-label={`Model probability ${label}`} /></div>)}</div>
          <p className="muted">Validated finite probabilities summing to 1 (tolerance 0.000001). Not guaranteed confidence or a probability of profit.</p>
        </section>
        <section className="bench-panel"><h2>Input window</h2><p className="muted">Decision boundary: bar {sample.decisionBar}, fully visible. Only the saved features below are supplied; no future bars are requested.</p>
          <div className="table-scroll" tabIndex={0} role="region" aria-label="Saved model inputs"><table><thead><tr><th>Bar</th><th>Feature</th><th>Value</th></tr></thead><tbody>{sample.inputs.map((input, i) => <tr key={i}><td>{input.bar}{input.bar === sample.decisionBar ? " · current" : ""}</td><td>{input.feature}</td><td>{input.value} {input.unit}</td></tr>)}</tbody></table></div>
          {!sample.inputs.length && <p>Input features unavailable in this artifact.</p>}
        </section>
        <section className="bench-panel"><p className="eyebrow">LLM explanation interface</p><h2>{bundle.origin === "synthetic" ? "Synthetic explanation example" : "Saved explanation"}</h2>
          <p className="muted">{bundle.origin === "synthetic" ? "Authored synthetic example, not a live LLM output." : "Imported text, not generated or independently validated by this interface."}</p>
          {explanation ? <><p className="explanation-text">{explanation.text}</p><details><summary>Template baseline</summary><p>{explanation.template}</p></details><p className="muted">Model: {explanation.modelVersion}<br />Prompt: {explanation.promptVersion}</p></> : <p>Explanation unavailable. No template or LLM response is silently substituted.</p>}
        </section>
      </div><aside className="research-stack">
        <section className="bench-panel"><p className="eyebrow">Evidence / not causal proof</p><h2>Fact sheet & checks</h2>
          {explanation ? <><ul className="evidence-list">{explanation.facts.map(f => <li key={f.id}><code>{f.id}</code> · bar {f.bar}<p>{f.text}</p></li>)}</ul><p>Referenced facts: {explanation.references.join(", ") || "None"}</p>
            <h3>Recorded automated checks</h3><p className="muted">Exporter-reported outcomes; not verification of the whole explanation.</p>
            {explanation.checks.length ? explanation.checks.map((c, i) => <div className="check-row" key={i}><strong>{c.passed ? "Automated check passed" : "Check failed"} · {c.name}</strong><p>{c.detail}</p></div>) : <p>Not evaluated.</p>}</> : <p>No explanation evidence loaded.</p>}
        </section>
        <section className="bench-panel"><h2>Execution is separate</h2><dl className="fact-grid"><dt>Model prediction</dt><dd>{sample.predicted}</dd><dt>Target position</dt><dd>Unavailable</dd><dt>Risk decision</dt><dd>Unavailable</dd><dt>Order / actual fill</dt><dd>Unavailable</dd></dl><p className="muted">A prediction is not an order. Manual decisions at N execute at N+1 or later in Replay.</p></section>
        <section className="bench-panel"><h2>Occlusion sensitivity</h2><p className="muted">No attribution artifact in this milestone. Sensitivity under an intervention would not prove causal reasoning.</p></section>
      </aside></div>
      <Link className="bench-button next-link" href={`/evaluation?run=${encodeURIComponent(run.id)}`}>Review saved evaluation →</Link>
    </>}
  </div>;
}
export function Evaluation() {
  const { bundle } = useResearch();
  const params = useSearchParams();
  const runId = params.get("run");
  const reportId = params.get("report");
  const reports = bundle?.reports.filter(r => !runId || r.runId === runId) ?? [];
  const report = reportId ? reports.find(r => r.id === reportId) : reports[0];
  if (!bundle) return <Empty title="No evaluation report loaded">Predictive evaluation, explanation assessment and machine-readable test reports are not exposed by the backend. No synthetic scores are substituted.</Empty>;
  return <div className="research-stack">
    <section className="bench-panel"><p className="eyebrow">03 / Retrospective evaluation</p><h2>Saved reports</h2><div className="bench-actions">{reports.map(r => <Link className="bench-button" key={r.id} href={`/evaluation?run=${encodeURIComponent(r.runId)}&report=${encodeURIComponent(r.id)}`}>{r.id}</Link>)}</div>
      {(reportId && !report || runId && !bundle.runs.some(r => r.id === runId)) && <SelectionError />}
      {!reports.length && <p>No matching evaluation reports. <Link href="/experiments">Choose a run</Link>.</p>}
    </section>
    {report && <>
      <section className="bench-panel"><h2>Predictive model results</h2><p className="selection-line"><code>{report.runId}</code> / <code>{report.id}</code></p><dl className="fact-grid"><dt>Evaluation split</dt><dd>{report.split}</dd><dt>Outcome cutoff</dt><dd>Bar {report.cutoffBar}</dd><dt>Sample count</dt><dd>{report.sampleCount}</dd><dt>Macro-F1</dt><dd>{report.macroF1 ?? "Unavailable"}</dd></dl><p className="muted">Saved report only. No calibration, confidence intervals or aggregate metrics are computed in the browser.</p></section>
      <section className="bench-panel"><div className="section-heading"><div><p className="eyebrow">Failure explorer</p><h2>Where the prediction failed</h2></div><span className="bench-badge">{report.failures.length} listed / {report.sampleCount} evaluated</span></div>
        <p className="muted">Retrospective outcomes below are not part of the decision-time input or explanation.</p>
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Evaluation failures"><table><thead><tr><th>Sample</th><th>Failure</th><th>Observed outcome</th><th>Saved evidence</th></tr></thead><tbody>{report.failures.map(f => <tr key={f.sampleId}><td><Link href={`/inspector?run=${encodeURIComponent(report.runId)}&sample=${encodeURIComponent(f.sampleId)}`}>{f.sampleId} → Inspector</Link></td><td>Misclassification</td><td>{f.observed} · bar {f.outcomeBar}</td><td>{f.detail}</td></tr>)}</tbody></table></div>
        {!report.failures.length && <p>No failures listed in this report.</p>}
      </section>
    </>}
    <div className="two-column"><section className="bench-panel"><h2>Explanation evaluation</h2><p className="muted">Aggregate numeric/reference pass rates, human ratings, fallback rates and occlusion comparisons are unavailable. Individual recorded checks are shown in Inspector, not combined into a trust score.</p></section><section className="bench-panel"><h2>Integrity & reproducibility</h2><p>No test report loaded.</p><p className="muted">Historical test results would require a timestamp and code commit; they would not certify current system health.</p></section></div>
    <section className="bench-panel"><h2>Financial outcomes · secondary</h2><p className="muted">No financial evaluation artifact loaded. <Link href="/replay">Replay</Link> retains its existing portfolio and risk outputs. Trade win rate is not directional classification accuracy.</p></section>
  </div>;
}
