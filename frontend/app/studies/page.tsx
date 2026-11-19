"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import { Decision, api, pct, Plot } from "@/components/research/StudyWidgets";
import { TrainingCurve } from "@/components/research/TrainingCurve";

type State = { state: string; message: string; done: number; total: number };
type Inventory = {
  sources: string[];
  transformer: boolean;
  studies: ({ id: string } & State)[];
};
type Run = {
  key: string;
  model: string;
  variant: string;
  seed: number;
  f1: number;
  loss: number;
  curves: { step: number; trainingLoss: number; validationLoss: number }[];
};
type Study = {
  config: { source: string; epochs: number };
  status: State;
  runs: Run[];
};
type Sample = { index: number; up: boolean; p: number; period: string };
type Strategy = {
  name: string;
  gain: number;
  dd: number;
  equity: number[];
  exposure: number[];
  fees: number[];
};
type Report = {
  samples: Sample[];
  trainEnd: string;
  validationStart: string;
  validationEnd: string;
  trainCount: number;
  validationUp: number;
  trainUp: number;
  missing: number;
  rows: number;
  bins: number[];
  normalization: { mean: number; std: number };
  dataset: string;
  parameters: number;
  checkpoint: string;
  commit: string;
  calibration: { count: number; predicted: number; observed: number }[];
  interval: number[];
  strategies: Strategy[];
  accounting: string;
  rejections?: number[];
};
const tabs = [
  "Experiment",
  "Data",
  "Inspector",
  "Blind replay",
  "Evaluation",
  "Portfolio",
  "Presentation",
];
function Launcher({
  inv,
  open,
}: {
  inv: Inventory;
  open: (id: string) => void;
}) {
  const [source, s] = useState("synthetic"),
    [models, m] = useState("MLP"),
    [seeds, f] = useState("0"),
    [epochs, e] = useState(8),
    [sweep, a] = useState(false),
    [busy, b] = useState(false),
    [err, x] = useState("");
  async function submit() {
    b(true);
    x("");
    try {
      if (!/^\d+(,\s*\d+){0,4}$/.test(seeds))
        throw Error("Enter one to five integer seeds separated by commas.");
      const r = await api<{ id: string }>("", {
        source,
        seeds: seeds.split(",").map(Number),
        epochs,
        models: models.split("+"),
        variants: sweep
          ? ["reference", "no-position", "no-dropout", "no-decay", "fixed-lr"]
          : ["reference"],
      });
      open(r.id);
    } catch (ex) {
      x(String(ex));
    } finally {
      b(false);
    }
  }
  return (
    <details className="bench-panel" open={!inv.studies.length}>
      <summary>New validation study</summary>
      <div className="bench-actions">
        <label>
          Dataset{" "}
          <select value={source} onChange={(v) => s(v.target.value)}>
            {inv.sources.map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <label>
          Models{" "}
          <select value={models} onChange={(v) => m(v.target.value)}>
            <option>MLP</option>
            {inv.transformer && (
              <>
                <option>Transformer</option>
                <option>MLP+Transformer</option>
              </>
            )}
          </select>
        </label>
        <label>
          Seeds{" "}
          <input size={9} value={seeds} onChange={(v) => f(v.target.value)} />
        </label>
        <label>
          Epochs{" "}
          <input
            type="number"
            min="1"
            max="60"
            value={epochs}
            onChange={(v) => e(Number(v.target.value))}
          />
        </label>
        {models.includes("Transformer") && (
          <label>
            <input
              type="checkbox"
              checked={sweep}
              onChange={(v) => a(v.target.checked)}
            />{" "}
            Include four ablations
          </label>
        )}
        <button className="bench-button" disabled={busy} onClick={submit}>
          Start training
        </button>
      </div>
      {!inv.transformer && (
        <p>
          Enable Transformer runs by installing CPU PyTorch in the backend
          environment.
        </p>
      )}
      {err && <p role="alert">{err}</p>}
    </details>
  );
}
function drawdowns(equity: number[]) {
  let peak = equity[0];
  return equity.map((value) => {
    peak = Math.max(peak, value);
    return peak > 0 ? (peak - value) / peak : 0;
  });
}
function Results({ r, inspect }: { r: Report; inspect: (i: number) => void }) {
  const [period, set] = useState("all"),
    [filter, f] = useState("errors");
  const rows = r.samples.filter((x) => period === "all" || x.period === period);
  const cells = [0, 0, 0, 0];
  rows.forEach((x) => cells[Number(x.up) * 2 + Number(x.p >= 0.5)]++);
  const modelHits = cells[0] + cells[3];
  const alwaysUpHits = cells[2] + cells[3];
  const accuracy = (hits: number) => rows.length ? pct(hits / rows.length) : "—";
  const cases = rows
    .filter((x) => filter === "all" || (x.p >= 0.5) !== x.up)
    .sort((a, b) => Math.abs(b.p - 0.5) - Math.abs(a.p - 0.5));
  return (
    <section className="bench-panel">
      <h2>Validation results</h2>
      <label>
        Period{" "}
        <select value={period} onChange={(v) => set(v.target.value)}>
          {["all", ...Array.from(new Set(r.samples.map((x) => x.period)))].map(
            (v) => (
              <option key={v}>{v}</option>
            ),
          )}
        </select>
      </label>
      <p>
        {rows.length} cases · Model accuracy {accuracy(modelHits)} · Always-up
        accuracy {accuracy(alwaysUpHits)}
      </p>
      <table>
        <caption>Confusion matrix</caption>
        <thead>
          <tr>
            <th>Actual / predicted</th>
            <th>Not up</th>
            <th>Up</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>Not up</th>
            <td>{cells[0]}</td>
            <td>{cells[1]}</td>
          </tr>
          <tr>
            <th>Up</th>
            <td>{cells[2]}</td>
            <td>{cells[3]}</td>
          </tr>
        </tbody>
      </table>
      <h3>Full-period calibration</h3>
      {r.calibration.map((v, i) => (
        <p key={i}>
          <meter min={0} max={1} value={v.observed} /> Predicted{" "}
          {pct(v.predicted)} · observed {pct(v.observed)} ({v.count})
        </p>
      ))}
      <p>
        Paired 95% interval for accuracy difference vs always up:{" "}
        {r.interval.map(pct).join(" to ")}.
      </p>
      <label>
        Browse{" "}
        <select value={filter} onChange={(v) => f(v.target.value)}>
          <option value="errors">Errors by confidence</option>
          <option value="all">All predictions</option>
        </select>
      </label>{" "}
      <select
        aria-label="Inspect prediction"
        value=""
        onChange={(v) => inspect(Number(v.target.value))}
      >
        <option value="">Select from {cases.length} cases</option>
        {cases.map((x) => (
          <option value={x.index} key={x.index}>
            Case {x.index + 1} · P(up) {pct(x.p)}
          </option>
        ))}
      </select>
    </section>
  );
}
function Guide({ go }: { go: (name: string) => void }) {
  const team = useQuery({
    queryKey: ["research-contributions"],
    queryFn: () =>
      api<{ area: string; authors: { name: string; commit: string }[] }[]>(
        "/contributions",
      ),
  });
  const [start, set] = useState<number | null>(() =>
      typeof window === "undefined"
        ? null
        : Number(sessionStorage.getItem("rehearsal-start")) || null,
    ),
    [elapsed, tick] = useState(0);
  useEffect(() => {
    if (start === null) return;
    const h = setInterval(
      () => tick(Math.floor((Date.now() - start) / 1000)),
      500,
    );
    return () => clearInterval(h);
  }, [start]);
  return (
    <section className="bench-panel">
      <h2>Three-minute defense</h2>
      <p>Can a model trained on 64 past returns predict the next move better than a simple baseline on unseen dates?</p>
      <p>Use a single saved run throughout. Ask the jury to vote on one window, then compare its outcome with the full validation period.</p>
      <button
        className="bench-button"
        onClick={() => {
          sessionStorage.setItem("rehearsal-start", String(Date.now()));
          set(Date.now());
          tick(0);
        }}
      >
        Start rehearsal
      </button>
      <p>
        {Math.floor(elapsed / 60)}m {elapsed % 60}s
      </p>
      {[
        ["0:00 · Data source, chronological split and class balance (3/20)", "Data"],
        ["0:30 · Course relevance (3/20), architecture and training (3/20)", "Experiment"],
        ["1:00 · Execution (5/20). Cast a vote and reveal the broker fills", "Blind replay"],
        ["1:35 · Results (5/20). Compare with always-up and inspect an error", "Evaluation"],
        ["2:10 · Explain the decision", "Inspector"],
        ["2:40 · Portfolio limits and team ownership", "Portfolio"],
      ].map(([label, tab]) => (
        <p key={tab}>
          <button className="bench-button" onClick={() => go(tab)}>
            {label}
          </button>
        </p>
      ))}
      <h3>Task distribution (1/20)</h3>
      <p><a href="https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/blob/main/docs/work-plan.md#proposed-six-member-allocation">Proposed responsibilities for the six members</a></p>
      <p>Recorded commits by area</p>
      {team.data?.map((area) => (
        <p key={area.area}>
          {area.area}:{" "}
          {area.authors.map((a) => (
            <a
              key={a.name}
              href={
                "https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/commit/" +
                a.commit
              }
            >
              {a.name}{" "}
            </a>
          ))}
        </p>
      ))}
      <a href="https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/graphs/contributors">
        Review team contributions on GitHub
      </a>
    </section>
  );
}
export default function Studies() {
  const params = useSearchParams(),
    router = useRouter();
  const id = params.get("study") || "",
    tab = params.get("tab") || "Experiment",
    index = Math.max(0, Number(params.get("case")) || 0);
  function nav(change: Record<string, string>) {
    const q = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(change)) q.set(k, v);
    router.push("/studies?" + q.toString(), { scroll: false });
  }
  const inv = useQuery({
    queryKey: ["studies"],
    queryFn: () => api<Inventory>(""),
    refetchInterval: 3500,
  });
  const study = useQuery({
    queryKey: ["study", id],
    queryFn: () => api<Study>("/" + id),
    enabled: !!id,
    refetchInterval: (q) =>
      ["complete", "failed", "interrupted"].includes(
        q.state.data?.status.state || "",
      )
        ? false
        : 2000,
  });
  const s = study.data,
    run = s?.runs.find((r) => r.key === params.get("run")) || s?.runs[0],
    base = "/" + id + "/" + run?.key;
  const report = useQuery({
    queryKey: ["report", base],
    queryFn: () => api<Report>(base + "/evaluation"),
    enabled:
      !!run && ["Portfolio", "Evaluation", "Data", "Experiment"].includes(tab),
  });
  const catalog = useQuery({
    queryKey: ["study-catalog", s?.status.state],
    queryFn: () =>
      api<
        {
          source: string;
          key: string;
          id: string;
          f1: number;
          dataset: string;
          epochs: number;
        }[]
      >("/results"),
    enabled: tab === "Evaluation",
  });
  const [asset, filterAsset] = useState("all");
  const r = report.data;
  if (!inv.data)
    return (
      <p role="status">
        {inv.error
          ? String(inv.error)
          : "Connecting to the local research service…"}
      </p>
    );
  return (
    <div className="research-stack">
      <Launcher
        inv={inv.data}
        open={(v) => nav({ study: v, run: "", tab: "Experiment", case: "0" })}
      />
      <section className="bench-panel">
        <label>
          Saved study{" "}
          <select
            value={id}
            onChange={(v) => nav({ study: v.target.value, run: "", case: "0" })}
          >
            <option value="">Choose a study</option>
            {inv.data.studies.map((x) => (
              <option key={x.id} value={x.id}>
                {x.id.slice(0, 8)} · {x.state}
              </option>
            ))}
          </select>
        </label>
        {s && (
          <>
            <p role="status">
              {s.config.source} · {s.status.message} · {s.status.done}/
              {s.status.total} runs
            </p>
            {run && (
              <label>
                Active run{" "}
                <select
                  value={run.key}
                  onChange={(v) => nav({ run: v.target.value, case: "0" })}
                >
                  {s.runs.map((x) => (
                    <option key={x.key}>{x.key}</option>
                  ))}
                </select>
              </label>
            )}
          </>
        )}
      </section>
      {study.error && <p role="alert">{String(study.error)}</p>}
      {run && s && (
        <>
          <nav aria-label="Study views" className="bench-actions">
            {tabs.map((x) => (
              <button
                key={x}
                className="bench-button"
                aria-pressed={tab === x}
                onClick={() => nav({ tab: x })}
              >
                {x}
              </button>
            ))}
          </nav>
          {report.error && tab !== "Blind replay" && (
            <p role="alert">{String(report.error)}</p>
          )}
          {tab === "Experiment" && (
            <section className="bench-panel">
              <h2>Model and ablation comparison</h2>
              <table>
                <thead>
                  <tr>
                    <th>Model / configuration</th>
                    <th>Seeds</th>
                    <th>Mean macro-F1</th>
                    <th>Range</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from(
                    new Set(s.runs.map((x) => x.model + " / " + x.variant)),
                  ).map((group) => {
                    const rr = s.runs.filter(
                        (x) => x.model + " / " + x.variant === group,
                      ),
                      scores = rr.map((x) => x.f1);
                    return (
                      <tr key={group}>
                        <th>{group}</th>
                        <td>{rr.map((x) => x.seed).join(", ")}</td>
                        <td>
                          {(
                            scores.reduce((a, b) => a + b) / scores.length
                          ).toFixed(3)}
                        </td>
                        <td>
                          {Math.min(...scores).toFixed(3)}–
                          {Math.max(...scores).toFixed(3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <TrainingCurve
                run={{
                  id: run.key,
                  model: run.model,
                  dataset: s.config.source,
                  split: "validation",
                  checkpoint: r?.checkpoint || "",
                  seed: run.seed,
                  lossName: "Cross-entropy",
                  curves: run.curves,
                }}
              />
              {r && (
                <p>
                  {run.model === "MLP"
                    ? "64 normalized returns → 32 ReLU units → two classes"
                    : "64 quantile tokens → 16-dimensional embeddings → two attention heads → mean pooling → two classes"}{" "}
                  · {r.parameters} parameters · {s.config.epochs} epochs
                </p>
              )}
            </section>
          )}
          {tab === "Data" && r && (
            <section className="bench-panel">
              <h2>{s.config.source} · Dataset and preprocessing</h2>
              <p>
                {r.rows} source rows through validation · {r.missing} missing
                values · training through {r.trainEnd} · validation{" "}
                {r.validationStart} to {r.validationEnd}
              </p>
              <p>
                {r.trainCount} train windows ({pct(r.trainUp)} up) ·{" "}
                {r.samples.length} validation windows ({pct(r.validationUp)}{" "}
                up). Each window uses 65 adjusted closes and one next-day label.
              </p>
              <p>
                Training mean {r.normalization.mean.toPrecision(4)} · standard
                deviation {r.normalization.std.toPrecision(4)}
              </p>
              <Plot values={r.bins} label="Train-fitted quantile boundaries" />
              <details>
                <summary>Reproduction identifiers</summary>
                <p style={{ overflowWrap: "anywhere" }}>
                  Data {r.dataset}
                  <br />
                  Checkpoint {r.checkpoint}
                  <br />
                  Commit {r.commit}
                </p>
              </details>
            </section>
          )}
          {["Inspector", "Blind replay"].includes(tab) && (
            <Decision
              key={base + index + tab}
              base={base}
              index={index}
              blind={tab === "Blind replay"}
            />
          )}
          {tab === "Evaluation" && r && (
            <Results
              r={r}
              inspect={(i) => nav({ case: String(i), tab: "Inspector" })}
            />
          )}
          {tab === "Evaluation" && catalog.data && (
            <details className="bench-panel">
              <summary>Compare results across datasets</summary>
              <label>
                Source{" "}
                <select
                  value={asset}
                  onChange={(e) => filterAsset(e.target.value)}
                >
                  {[
                    "all",
                    ...Array.from(new Set(catalog.data.map((v) => v.source))),
                  ].map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              </label>
              <table>
                <thead>
                  <tr>
                    <th>Dataset</th>
                    <th>Run</th>
                    <th>Epochs</th>
                    <th>Macro-F1</th>
                  </tr>
                </thead>
                <tbody>
                  {catalog.data
                    .filter((x) => asset === "all" || x.source === asset)
                    .map((x) => (
                      <tr key={x.id + x.key}>
                        <td>
                          {x.source} · {x.dataset.slice(0, 8)}
                        </td>
                        <td>
                          <button
                            className="bench-button"
                            onClick={() =>
                              nav({ study: x.id, run: x.key, case: "0" })
                            }
                          >
                            {x.key}
                          </button>
                        </td>
                        <td>{x.epochs}</td>
                        <td>{x.f1.toFixed(3)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </details>
          )}
          {tab === "Portfolio" && r && (
            <section className="bench-panel">
              <h2>Portfolio and execution</h2>
              <p>
                Accounting: {r.accounting}. Same initial cash, costs and
                decision dates for each strategy.
              </p>
              {r.strategies.map((x, i) => (
                <details key={x.name} open={i === 0}>
                  <summary>
                    {x.name} · return {pct(x.gain)} · max drawdown {pct(x.dd)}
                  </summary>
                  <Plot values={x.equity} label="Marked equity" />
                  <Plot values={drawdowns(x.equity)} label="Drawdown from running peak" />
                  <Plot
                    values={x.exposure}
                    label="Fraction of equity exposed"
                  />
                  <p>
                    Fees {x.fees.at(-1)?.toFixed(2)} · rejected decisions{" "}
                    {r.rejections?.[i] ?? 0}
                  </p>
                </details>
              ))}
            </section>
          )}
          {tab === "Presentation" && <Guide go={(t) => nav({ tab: t })} />}
        </>
      )}
    </div>
  );
}
