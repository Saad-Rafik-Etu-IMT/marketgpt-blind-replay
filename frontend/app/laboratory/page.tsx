"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Campaign,
  Catalogue,
  Detail,
  laboratoryApi,
  laboratoryUrl,
  labelText,
  Run,
  Source,
  numberText,
} from "@/lib/research/laboratory";
import {
  CampaignForm,
  TrainingMonitor,
} from "@/components/research/LaboratoryTraining";
import {
  EvaluationView,
  ReplayView,
} from "@/components/research/LaboratoryResults";
import { LinePlot, PlotFrame } from "@/components/research/ScientificPlot";

const tabs = [
  "Overview",
  "Data",
  "Training",
  "Evaluation",
  "Replay",
  "Explainability",
] as const;
type Tab = (typeof tabs)[number];
const running = (state?: string) => state === "queued" || state === "running";

export default function Laboratory() {
  const router = useRouter(),
    params = useSearchParams(),
    client = useQueryClient();
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  const active = (tabs.find((x) => x === params.get("tab")) ??
    "Overview") as Tab;
  const [part, setPart] = useState("inner_validation"),
    [entry, setEntry] = useState(0);
  const [error, setError] = useState("");
  const catalogue = useQuery({
    queryKey: ["lab-catalog"],
    queryFn: () => laboratoryApi<Catalogue>("/catalog"),
    staleTime: 60000,
  });
  const sources = useQuery({
    queryKey: ["lab-sources"],
    queryFn: () => laboratoryApi<Source[]>("/sources"),
    staleTime: 30000,
  });
  const inventory = useQuery({
    queryKey: ["lab-inventory"],
    queryFn: () => laboratoryApi<Campaign[]>(),
    refetchInterval: 2500,
  });
  const id = params.get("campaign") ?? inventory.data?.[0]?.id ?? "";
  const detail = useQuery({
    queryKey: ["lab-detail", id],
    enabled: !!id,
    queryFn: () => laboratoryApi<Detail>(`/${id}`),
    refetchInterval: (q) =>
      running(q.state.data?.status.state) ||
      q.state.data?.status.evaluation_state === "running"
        ? 1500
        : 4000,
  });
  const runs = useQuery({
    queryKey: ["lab-runs", id],
    enabled: !!id,
    queryFn: () => laboratoryApi<Run[]>(`/${id}/runs`),
    refetchInterval: 2500,
  });
  const run =
    params.get("run") ??
    runs.data?.find((r) => r.state === "complete")?.key ??
    "";
  const selectedRun =
    runs.data?.find((r) => r.key === run && r.state === "complete")?.key ?? "";
  function navigate(tab: Tab, campaign = id, key = run) {
    const q = new URLSearchParams();
    q.set("tab", tab);
    if (campaign) q.set("campaign", campaign);
    if (key) q.set("run", key);
    router.replace("/laboratory?" + q.toString(), { scroll: false });
  }
  function refresh() {
    void client.invalidateQueries({ queryKey: ["lab-inventory"] });
    void client.invalidateQueries({ queryKey: ["lab-detail", id] });
    void client.invalidateQueries({ queryKey: ["lab-runs", id] });
  }
  async function dispatch() {
    try {
      await laboratoryApi("/queue/dispatch", {});
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }
  return (
    <div className="research-stack laboratory">
      <section className="bench-panel lab-context">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Research workspace</p>
            <h2>One campaign. One evidence trail.</h2>
          </div>
          <span className="bench-badge">Protected test locked</span>
        </div>
        <nav className="study-nav" aria-label="Laboratories">
          {tabs.map((t) => (
            <button
              key={t}
              className="bench-button"
              disabled={!ready}
              aria-pressed={t === active}
              onClick={() => navigate(t)}
            >
              {t}
            </button>
          ))}
        </nav>
        <div className="lab-two">
          <label>
            Selected campaign
            <select
              value={id}
              onChange={(e) => {
                setEntry(0);
                navigate(active, e.target.value, "");
              }}
            >
              <option value="">No campaign selected</option>
              {inventory.data?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} · {c.state} ({c.done}/{c.total})
                </option>
              ))}
            </select>
          </label>
          <label>
            Selected run
            <select
              value={run}
              disabled={!id}
              onChange={(e) => navigate(active, id, e.target.value)}
            >
              <option value="">Select a run</option>
              {runs.data?.map((r) => (
                <option key={r.key} value={r.key}>
                  {r.key} · {r.state}
                </option>
              ))}
            </select>
          </label>
        </div>
        {(inventory.error ||
          catalogue.error ||
          sources.error ||
          detail.error ||
          error) && (
          <p role="alert" className="bench-notice">
            {String(
              inventory.error ||
                catalogue.error ||
                sources.error ||
                detail.error ||
                error,
            )}
          </p>
        )}
        {detail.data && (
          <div className="lab-context-meta">
            <span>{detail.data.spec.profile}</span>
            <span>{detail.data.spec.sources.join(", ")}</span>
            <span>
              {detail.data.status.state} · {detail.data.status.done}/
              {detail.data.status.total} runs
            </span>
            <span>Spec {detail.data.fingerprint.slice(0, 12)}</span>
          </div>
        )}
      </section>
      {active === "Overview" && (
        <>
          <section className="bench-panel">
            <p className="eyebrow">Research question</p>
            <h2>Does financial AI add measurable predictive value?</h2>
            <p>
              Compare causal model inputs and architectures against simple
              statistical baselines, validate over chronological folds, and
              replay the exact registered model through the deterministic
              broker.
            </p>
            <div className="metric-strip">
              <div>
                <span>Saved campaigns</span>
                <strong>{inventory.data?.length ?? 0}</strong>
              </div>
              <div>
                <span>Available feature profiles</span>
                <strong>
                  {catalogue.data?.feature_profiles.filter((x) => x.available)
                    .length ?? "—"}
                </strong>
              </div>
              <div>
                <span>Prediction tasks</span>
                <strong>{catalogue.data?.targets.length ?? "—"}</strong>
              </div>
              <div>
                <span>Final holdout</span>
                <strong>Locked</strong>
              </div>
            </div>
            <div className="lab-roadmap">
              {[
                "Data → causal features",
                "Training → selection",
                "Evaluation → baselines",
                "Replay → accounting",
                "Explanations → facts",
              ].map((x, i) => (
                <div key={x}>
                  <strong>0{i + 1}</strong>
                  <span>{x}</span>
                </div>
              ))}
            </div>
            <div className="bench-actions">
              <button
                className="bench-button primary-action"
                onClick={() => navigate("Training")}
              >
                Configure training
              </button>
              <button
                className="bench-button"
                onClick={() => navigate("Evaluation")}
              >
                Inspect evidence
              </button>
              <Link className="bench-button" href="/studies">
                Legacy MLP / Transformer studies
              </Link>
            </div>
          </section>
          <section className="bench-panel">
            <div className="section-heading">
              <h2>Campaign registry</h2>
              <button className="bench-button" onClick={() => void dispatch()}>
                Dispatch waiting queue
              </button>
            </div>
            {!inventory.data?.length ? (
              <p className="muted">
                No campaigns yet. Start with a synthetic positive or negative
                control to verify the pipeline.
              </p>
            ) : (
              <div className="lab-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Profile</th>
                      <th>State</th>
                      <th>Runs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inventory.data.map((c) => (
                      <tr key={c.id}>
                        <td>
                          <button
                            className="bench-button"
                            onClick={() => navigate("Training", c.id, "")}
                          >
                            {c.name}
                          </button>
                        </td>
                        <td>{c.profile}</td>
                        <td>{labelText(c.state)}</td>
                        <td>
                          {c.done}/{c.total}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          {id && <Holdout key={id} id={id} run={selectedRun} />}
          {id && (
            <section className="bench-panel">
              <h2>Export evidence</h2>
              <div className="bench-actions">
                {["md", "csv", "json"].map((k) => (
                  <a
                    className="bench-button"
                    key={k}
                    href={`${laboratoryUrl}/${id}/export/${k}`}
                  >
                    Download {k.toUpperCase()}
                  </a>
                ))}
              </div>
              <p className="muted">
                Exports contain recorded artifacts only; no retraining or
                protected data access.
              </p>
            </section>
          )}
        </>
      )}
      {active === "Data" && catalogue.data && sources.data && (
        <DataView catalogue={catalogue.data} sources={sources.data} id={id} />
      )}
      {active === "Training" && catalogue.data && sources.data && (
        <>
          {detail.data && (
            <TrainingMonitor
              id={id}
              status={detail.data.status}
              runs={runs.data ?? []}
              run={run}
              onRun={(key) => navigate("Training", id, key)}
              onRefresh={refresh}
            />
          )}
          <CampaignForm
            catalogue={catalogue.data}
            sources={sources.data}
            onCreated={(newId) => {
              refresh();
              navigate("Training", newId, "");
            }}
          />
        </>
      )}
      {(active === "Evaluation" || active === "Explainability") &&
        (id ? (
          <EvaluationView
            key={id + active}
            id={id}
            part={part}
            changePart={setPart}
            entry={entry}
            changeEntry={setEntry}
            explanation={active === "Explainability"}
          />
        ) : (
          <section className="bench-panel">
            <h2>No campaign selected</h2>
            <p>Train a model before opening evaluation evidence.</p>
          </section>
        ))}
      {active === "Replay" && (
        <>
          <label className="lab-field">
            Replay partition
            <select value={part} onChange={(e) => setPart(e.target.value)}>
              <option value="inner_validation">Inner validation</option>
              <option value="validation">Historical validation</option>
            </select>
          </label>
          <ReplayView
            key={id + selectedRun + part}
            id={id}
            run={selectedRun}
            part={part}
          />
        </>
      )}
    </div>
  );
}

function DataView({
  catalogue,
  sources,
  id,
}: {
  catalogue: Catalogue;
  sources: Source[];
  id: string;
}) {
  const [source, setSource] = useState("synthetic"),
    [feature, setFeature] = useState("rsi_14");
  const description = catalogue.features.find((f) => f.id === feature);
  const preview = useQuery({
    queryKey: ["lab-preview", source, feature],
    queryFn: () =>
      laboratoryApi<{
        dates: string[];
        columns: string[];
        values: (number | null)[][];
      }>(`/sources/${source}/preview?feature=${feature}&last=250`),
  });
  const correlation = useQuery({
    queryKey: ["lab-correlation", source],
    queryFn: () =>
      laboratoryApi<{
        columns: string[];
        rows: number;
        matrix: (number | null)[][];
      }>(`/sources/${source}/correlation`),
  });
  const manifests = useQuery({
    queryKey: ["lab-manifests", id],
    enabled: !!id,
    queryFn: () =>
      laboratoryApi<
        {
          spec: { profile: string; target: string };
          parts: Record<
            string,
            {
              rows: number;
              dropped_undefined: number;
              first: string | null;
              last: string | null;
            }
          >;
        }[]
      >(`/${id}/datasets`),
  });
  const data = preview.data;
  const values =
    data?.values.flatMap((v, i) => (v[0] == null ? [] : [{ x: i, y: v[0] }])) ??
    [];
  const min = Math.min(...values.map((v) => v.y)),
    max = Math.max(...values.map((v) => v.y));
  const bins = values.length
    ? Array.from({ length: 12 }, (_, i) => ({
        x: min + ((max - min) * (i + 0.5)) / 12,
        y: values.filter(
          (v) =>
            Math.min(
              11,
              Math.floor(((v.y - min) / Math.max(max - min, 1e-9)) * 12),
            ) === i,
        ).length,
      }))
    : [];
  return (
    <>
      <section className="bench-panel">
        <h2>Dataset inventory and provenance</h2>
        <div className="lab-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Type</th>
                <th>Coverage</th>
                <th>Rows</th>
                <th>Provenance</th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s) => (
                <tr key={s.source}>
                  <td>{s.source}</td>
                  <td>{s.kind}</td>
                  <td>
                    {s.first ?? "Generated"} – {s.last ?? "Development"}
                  </td>
                  <td>{numberText(s.rows, 0)}</td>
                  <td>
                    {s.status}
                    {s.provenance_match === true
                      ? " · verified"
                      : s.provenance_match === false
                        ? " · mismatch"
                        : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="bench-panel">
        <h2>Point-in-time feature explorer</h2>
        <div className="lab-two">
          <label>
            Data source
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              {sources
                .filter((s) => s.status === "ok")
                .map((s) => (
                  <option key={s.source}>{s.source}</option>
                ))}
            </select>
          </label>
          <label>
            Feature
            <select
              value={feature}
              onChange={(e) => setFeature(e.target.value)}
            >
              {catalogue.features
                .filter((f) => f.blind_safe)
                .map((f) => (
                  <option key={f.id}>{f.id}</option>
                ))}
            </select>
          </label>
        </div>
        <p>{description?.description}</p>
        <p className="muted">
          {description?.definition} · {description?.lookback} bars of history ·{" "}
          {description?.units}
        </p>
        <p className="bench-notice">
          Preview stops at the development validation boundary. Protected test
          observations are not loaded into the feature explorer.
        </p>
        {preview.error && <p role="alert">{String(preview.error)}</p>}
        {data && (
          <div className="lab-two">
            <LinePlot
              title={`${data.columns[0]} · development history`}
              xLabel="Session"
              yLabel={description?.units}
              formatX={(v) => data.dates[Math.round(v)]?.slice(5) ?? ""}
              series={[{ name: data.columns[0], points: values }]}
            />
            <LinePlot
              title="Observed feature distribution · 12 bins"
              xLabel="Feature value"
              yLabel="Count"
              series={[{ name: "Frequency", points: bins, dots: true }]}
            />
          </div>
        )}
        {correlation.data && (
          <>
            <h3>Training-only feature redundancy</h3>
            <p className="muted">
              Spearman correlation over {correlation.data.rows} complete
              training rows; representative scalar features from the core
              profile. Blue = positive, red = negative.
            </p>
            <PlotFrame
              title="Training feature correlation heatmap"
              height={410}
            >
              {correlation.data.columns.map((c, i) => (
                <text
                  key={c}
                  x={186 + i * 40}
                  y={116}
                  textAnchor="start"
                  fontSize="9"
                  fill="var(--fg)"
                  transform={`rotate(-45 ${186 + i * 40} 116)`}
                >
                  {c}
                </text>
              ))}
              {correlation.data.columns.map((c, i) => (
                <text
                  key={c}
                  x={170}
                  y={126 + i * 22 + 15}
                  textAnchor="end"
                  fontSize="10"
                  fill="var(--fg)"
                >
                  {c}
                </text>
              ))}
              {correlation.data.matrix.flatMap((row, i) =>
                row.map((v, j) => (
                  <g key={i + ":" + j}>
                    <rect
                      x={178 + j * 40}
                      y={126 + i * 22}
                      width={39}
                      height={21}
                      fill={
                        v === null
                          ? "var(--line)"
                          : v < 0
                            ? `rgba(180,65,75,${0.15 + Math.abs(v) * 0.8})`
                            : `rgba(38,116,158,${0.15 + Math.abs(v) * 0.8})`
                      }
                    />
                    <text
                      x={197 + j * 40}
                      y={140 + i * 22}
                      fontSize="8"
                      textAnchor="middle"
                      fill={
                        v !== null && Math.abs(v) > 0.68 ? "white" : "var(--fg)"
                      }
                    >
                      {v === null ? "—" : v.toFixed(2)}
                    </text>
                  </g>
                )),
              )}
            </PlotFrame>
          </>
        )}
        {manifests.data?.length ? (
          <>
            <h3>Selected campaign partitions</h3>
            <div className="lab-table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Profile / target</th>
                    <th>Partition</th>
                    <th>Rows</th>
                    <th>Dropped</th>
                    <th>Period</th>
                  </tr>
                </thead>
                <tbody>
                  {manifests.data.flatMap((m, i) =>
                    Object.entries(m.parts).map(([name, p]) => (
                      <tr key={i + name}>
                        <td>
                          {m.spec.profile} / {m.spec.target}
                        </td>
                        <td>{name}</td>
                        <td>{p.rows}</td>
                        <td>{p.dropped_undefined}</td>
                        <td>
                          {p.first ?? "—"} – {p.last ?? "—"}
                        </td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
        <h3>Versioned feature profiles</h3>
        <div className="lab-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Profile</th>
                <th>Inputs</th>
                <th>Lookback</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {catalogue.feature_profiles.map((p) => (
                <tr key={p.id}>
                  <td>{labelText(p.id)}</td>
                  <td>{p.column_count ?? "—"}</td>
                  <td>{p.lookback ?? "—"}</td>
                  <td>{p.available ? p.description : p.unavailable_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function Holdout({ id, run }: { id: string; run: string }) {
  const [start, setStart] = useState(""),
    [end, setEnd] = useState(""),
    [error, setError] = useState("");
  const query = useQuery({
    queryKey: ["lab-holdout", id],
    queryFn: () =>
      laboratoryApi<{
        plan: {
          start: string;
          end: string;
          candidate: string;
          authorization_id: string;
        } | null;
        consumed: boolean;
        result_available: boolean;
      }>(`/${id}/holdout`),
  });
  async function freeze() {
    setError("");
    try {
      await laboratoryApi(`/${id}/holdout/lock`, { key: run, start, end });
      await query.refetch();
    } catch (e) {
      setError(String(e));
    }
  }
  const plan = query.data?.plan;
  return (
    <section className="bench-panel">
      <div className="section-heading">
        <h2>Protected final evaluation</h2>
        <span className="bench-badge">
          {plan ? "Protocol frozen" : "Not frozen"}
        </span>
      </div>
      <p>
        Historical development data has been examined. Establish a prospective
        period before it starts. Freezing a candidate does not reveal or execute
        the final benchmark.
      </p>
      {plan ? (
        <>
          <p>
            Candidate: {plan.candidate} · {plan.start} — {plan.end}
          </p>
          <p className="muted">
            {query.data?.consumed
              ? "Authorization consumed"
              : "Awaiting explicit local authorization after end date"}
          </p>
          <details>
            <summary>Authorization identity</summary>
            <code>{plan.authorization_id}</code>
          </details>
        </>
      ) : (
        <div className="lab-two">
          <label>
            Prospective start
            <input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
            />
          </label>
          <label>
            Prospective end
            <input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
            />
          </label>
          <button
            className="bench-button"
            disabled={!run || !start || !end}
            onClick={() => void freeze()}
          >
            Freeze selected candidate
          </button>
        </div>
      )}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
