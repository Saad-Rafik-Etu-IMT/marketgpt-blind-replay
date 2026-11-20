"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Decision, api, pct } from "@/components/research/StudyWidgets";
import { SeriesPlot } from "@/components/research/SeriesPlot";
import { TrainingCurve } from "@/components/research/TrainingCurve";
import {
  ModelFigures,
  ReliabilityPlot,
  ConfusionPlot,
  DatasetFigures,
} from "@/components/research/ResearchFigures";
import { HumanEvaluation } from "@/components/research/HumanEvaluation";
import { studyLabel, type StudySummary } from "@/lib/research/study-summary";

type State = { state: string; message: string; done: number; total: number };
type Inventory = {
  sources: string[];
  transformer: boolean;
  studies: StudySummary[];
};
type Run = {
  key: string;
  model: string;
  variant: string;
  seed: number;
  f1: number;
  accuracy: number;
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
  "Explanations",
  "Portfolio",
  "Presentation",
];
function Launcher({
  inv,
  open,
  visible,
}: {
  inv: Inventory;
  open: (id: string) => void;
  visible: boolean;
}) {
  const [source, s] = useState("synthetic"),
    [models, m] = useState("MLP"),
    [seeds, f] = useState("0"),
    [epochs, e] = useState(8),
    [sweep, a] = useState(false),
    [busy, b] = useState(false),
    [err, x] = useState("");
  async function submit() {
    if (
      busy ||
      inv.studies.some((s) => ["queued", "running"].includes(s.state))
    )
      return;
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
    <form
      id="new-study-form"
      className="study-form"
      hidden={!visible}
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <div className="study-controls">
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
        <button
          className="bench-button primary-action"
          disabled={
            busy ||
            inv.studies.some((s) => ["queued", "running"].includes(s.state))
          }
          type="submit"
        >
          {busy ? "Starting…" : "Start training"}
        </button>
      </div>
      {!inv.transformer && (
        <p className="muted">
          Enable Transformer runs by installing CPU PyTorch in the backend
          environment.
        </p>
      )}
      {err && <p role="alert">{err}</p>}
    </form>
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
  const accuracy = (hits: number) =>
    rows.length ? pct(hits / rows.length) : "—";
  const cases = rows
    .filter((x) => filter === "all" || x.p >= 0.5 !== x.up)
    .sort((a, b) => Math.abs(b.p - 0.5) - Math.abs(a.p - 0.5));
  return (
    <section className="bench-panel">
      <div className="section-heading">
        <h2>Validation results</h2>
        <label className="study-field short-field">
          Period{" "}
          <select value={period} onChange={(v) => set(v.target.value)}>
            {[
              "all",
              ...Array.from(new Set(r.samples.map((x) => x.period))),
            ].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="metric-strip">
        <div>
          <span>Decisions</span>
          <strong>{rows.length}</strong>
        </div>
        <div>
          <span>Model accuracy</span>
          <strong>{accuracy(modelHits)}</strong>
        </div>
        <div>
          <span>Always-up baseline</span>
          <strong>{accuracy(alwaysUpHits)}</strong>
        </div>
      </div>
      <div className="comparison-figures evaluation-figures">
        <div>
          <ConfusionPlot cells={cells} />
          <details className="supporting-details">
            <summary>Confusion matrix</summary>
            <table>
              <caption>Exact confusion counts</caption>
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
          </details>
        </div>
        <ReliabilityPlot key={period} samples={rows} inspect={inspect} />
      </div>
      <p className="muted compact-status">
        Full-period paired 95% interval for accuracy difference vs always up:{" "}
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
function Guide({
  go,
  replayHref,
}: {
  go: (name: string) => void;
  replayHref: string;
}) {
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
      <p>
        Can a model trained on 64 past returns predict the next move better than
        a simple baseline on unseen dates?
      </p>
      <p>
        Use a single saved run throughout. Ask the jury to vote on one window,
        then compare its outcome with the full validation period.
      </p>
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
      <Link className="bench-button primary-action" href={replayHref}>
        Watch recorded AI competition
      </Link>
      {[
        [
          "0:00 · Data source, chronological split and class balance (3/20)",
          "Data",
        ],
        [
          "0:30 · Course relevance (3/20), architecture and training (3/20)",
          "Experiment",
        ],
        [
          "1:00 · Execution (5/20). Cast a vote and reveal the broker fills",
          "Blind replay",
        ],
        [
          "1:35 · Results (5/20). Compare with always-up and inspect an error",
          "Evaluation",
        ],
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
      <p>
        <a href="https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/blob/main/docs/work-plan.md#proposed-six-member-allocation">
          Proposed responsibilities for the six members
        </a>
      </p>
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
    tab = tabs.includes(params.get("tab") || "")
      ? params.get("tab")!
      : "Experiment",
    index = Math.max(0, Math.trunc(Number(params.get("case")) || 0));
  const [creating, setCreating] = useState<boolean | null>(null);
  function nav(change: Record<string, string | null>) {
    const q = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(change)) {
      if (v === null) q.delete(k);
      else q.set(k, v);
    }
    router.push("/studies?" + q.toString(), { scroll: false });
  }
  const inv = useQuery({
    queryKey: ["studies"],
    queryFn: () => api<Inventory>(""),
    refetchInterval: 3500,
  });
  const latest =
    inv.data?.studies.find((s) => ["queued", "running"].includes(s.state)) ??
    inv.data?.studies.find((s) => s.state === "complete") ??
    inv.data?.studies[0];
  useEffect(() => {
    if (!params.has("study") && !params.has("new") && latest)
      router.replace(`/studies?study=${latest.id}&tab=Experiment`, {
        scroll: false,
      });
  }, [params, latest, router]);
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
    run =
      s?.runs.find((r) => r.key === params.get("run")) ||
      s?.runs.find(
        (r) =>
          params.has("seed") &&
          r.seed === Number(params.get("seed")) &&
          r.variant === (params.get("variant") || "reference"),
      ) ||
      s?.runs.find(
        (r) => params.has("seed") && r.seed === Number(params.get("seed")),
      ) ||
      s?.runs[0],
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
      <section className="bench-panel study-workspace">
        <div className="workspace-controls">
          <label className="study-field">
            Saved study{" "}
            <select
              value={id}
              onChange={(v) =>
                nav({
                  study: v.target.value,
                  run: null,
                  seed: null,
                  variant: null,
                  new: null,
                  case: "0",
                })
              }
            >
              <option value="">Choose a study</option>
              {inv.data.studies.map((x) => (
                <option key={x.id} value={x.id}>
                  {studyLabel(x)}
                </option>
              ))}
            </select>
          </label>
          {run && (
            <label className="study-field">
              Active run{" "}
              <select
                value={run.key}
                onChange={(v) =>
                  nav({
                    run: v.target.value,
                    seed: null,
                    variant: null,
                    case: "0",
                  })
                }
              >
                {s?.runs.map((x) => (
                  <option key={x.key} value={x.key}>
                    {x.model} · {x.variant} · Seed {x.seed}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="workspace-actions">
            <button
              type="button"
              className="bench-button"
              aria-expanded={
                creating ?? (params.has("new") || !inv.data.studies.length)
              }
              aria-controls="new-study-form"
              onClick={() =>
                setCreating(
                  !(
                    creating ??
                    (params.has("new") || !inv.data.studies.length)
                  ),
                )
              }
            >
              New validation study
            </button>
            {run && (
              <Link
                className="bench-button primary-action"
                href={`/replay?study=${id}&run=${run.key}&seed=${run.seed}&variant=${run.variant}&case=${index}`}
              >
                Watch AI competition <span aria-hidden="true">→</span>
              </Link>
            )}
          </div>
        </div>
        {s && (
          <div className="study-status" role="status">
            <span>
              {s.config.source} · {s.status.message}
            </span>
            <span>
              {s.status.done}/{s.status.total} runs
            </span>
            {["queued", "running"].includes(s.status.state) && (
              <progress
                value={s.status.done}
                max={Math.max(1, s.status.total)}
                aria-label="Training progress"
              />
            )}
          </div>
        )}
        <Launcher
          visible={creating ?? (params.has("new") || !inv.data.studies.length)}
          inv={inv.data}
          open={(v) => {
            setCreating(false);
            nav({
              study: v,
              run: null,
              seed: null,
              variant: null,
              tab: "Experiment",
              case: "0",
              new: null,
            });
          }}
        />
      </section>
      {study.error && <p role="alert">{String(study.error)}</p>}
      {run && s && (
        <>
          <nav aria-label="Study views" className="study-nav">
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
              <ModelFigures
                runs={s.runs}
                seed={run.seed}
                onSeed={(seed) => {
                  const match =
                    s.runs.find(
                      (r) =>
                        r.seed === seed &&
                        r.model === run.model &&
                        r.variant === run.variant,
                    ) ?? s.runs.find((r) => r.seed === seed);
                  if (match) nav({ run: match.key, case: "0" });
                }}
              />
              <details className="supporting-details">
                <summary>Exact scores by model and configuration</summary>
                <div className="table-scroll">
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
                </div>
              </details>
              <details className="supporting-details">
                <summary>
                  Selected run architecture and exact loss trace
                </summary>
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
              </details>
            </section>
          )}
          {tab === "Data" && r && (
            <section className="bench-panel">
              <h2>{s.config.source} · Dataset and preprocessing</h2>
              <DatasetFigures study={id} dataset={r.dataset} />
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
              <SeriesPlot
                values={r.bins}
                label="Train-fitted quantile boundaries"
                startLabel="Lowest boundary"
                endLabel="Highest boundary"
              />
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
              key={base}
              r={r}
              inspect={(i) => nav({ case: String(i), tab: "Inspector" })}
            />
          )}
          {tab === "Explanations" && <HumanEvaluation key={base} base={base} />}
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
                  <SeriesPlot values={x.equity} label="Marked equity" />
                  <SeriesPlot
                    values={drawdowns(x.equity)}
                    percent
                    label="Drawdown from running peak"
                  />
                  <SeriesPlot
                    values={x.exposure}
                    percent
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
          {tab === "Presentation" && (
            <Guide
              go={(t) => nav({ tab: t })}
              replayHref={`/replay?study=${id}&run=${run.key}&seed=${run.seed}&variant=${run.variant}`}
            />
          )}
        </>
      )}
    </div>
  );
}
