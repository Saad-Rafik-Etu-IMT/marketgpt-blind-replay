"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Evaluation,
  EvaluationEntry,
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
  laboratories,
  type Laboratory as Tab,
  labHref,
  humanLabel,
  runLabel,
  entryKey,
  chooseEntry,
} from "@/lib/research/product-context";
import { FindingSummary } from "@/components/research/ProductEvidence";
import { Presentation } from "@/components/research/LaboratoryPresentation";
import {
  CampaignForm,
  TrainingMonitor,
} from "@/components/research/LaboratoryTraining";
import {
  EvaluationView,
  ReplayView,
} from "@/components/research/LaboratoryResults";
import { LinePlot, PlotFrame } from "@/components/research/ScientificPlot";

const running = (state?: string) => state === "queued" || state === "running";

export default function Laboratory() {
  const router = useRouter(),
    params = useSearchParams(),
    client = useQueryClient();
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  const active =
    laboratories.find((x) => x === params.get("tab")) ?? "Overview";
  const part =
    params.get("part") === "validation" ? "validation" : "inner_validation";
  const entry = params.get("entry");
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
  const requestedId = params.get("campaign");
  const id =
    requestedId === "none"
      ? ""
      : (requestedId ?? inventory.data?.[0]?.id ?? "");
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
  const requestedRun = params.get("run");
  const run =
    requestedRun ?? runs.data?.find((r) => r.state === "complete")?.key ?? "";
  const runRecord = runs.data?.find((r) => r.key === run);
  const evidenceRun =
    runRecord ??
    (requestedRun ? { key: requestedRun, state: "unknown" } : undefined);
  const selectedRun =
    runs.data?.find((r) => r.key === run && r.state === "complete")?.key ?? "";
  function patch(changes: Record<string, string | number | null | undefined>) {
    router.replace(labHref(params, changes), { scroll: false });
  }
  function navigate(tab: Tab, campaign = id, key = run) {
    patch({
      tab,
      campaign: campaign || "none",
      run: key || null,
      mode: null,
      step: null,
    });
  }
  useEffect(() => {
    if (!requestedId && id)
      router.replace(labHref(params, { campaign: id }), { scroll: false });
  }, [requestedId, id, params, router]);
  const assets =
    detail.data?.spec.sources.flatMap((s) =>
      s.startsWith("synthetic") ? ["SYN_A", "SYN_B"] : [s],
    ) ?? [];
  const asset = assets.includes(params.get("asset") ?? "")
    ? params.get("asset")!
    : (assets[0] ?? "");
  const evidence = useQuery({
    queryKey: ["lab-evaluation", id, part],
    enabled: !!id,
    queryFn: () => laboratoryApi<Evaluation>(`/${id}/evaluation/${part}`),
    retry: false,
    refetchInterval:
      detail.data?.status.evaluation_state === "running" ? 1800 : false,
  });
  const result = chooseEntry(evidence.data?.results ?? [], entry, evidenceRun);
  function selectEntry(value: EvaluationEntry) {
    const match = runs.data?.find(
      (r) =>
        r.state === "complete" &&
        r.feature_profile === value.feature_profile &&
        r.preset === value.preset &&
        r.seed !== undefined &&
        value.seeds.includes(r.seed),
    );
    const same =
      runRecord?.feature_profile === value.feature_profile &&
      runRecord?.preset === value.preset &&
      runRecord.seed !== undefined &&
      value.seeds.includes(runRecord.seed);
    patch({ run: same ? run : (match?.key ?? null), entry: entryKey(value) });
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
  if (params.get("mode") === "present")
    return (
      <Presentation
        id={id}
        run={runRecord}
        detail={detail.data}
        result={result}
        part={part}
        asset={asset}
        catalogue={catalogue.data}
        step={Number(params.get("step") ?? 0)}
        onStep={(step) => patch({ step })}
        onExit={(tab) => patch({ mode: null, step: null, tab })}
      />
    );
  return (
    <div className="research-stack laboratory">
      <section className="bench-panel lab-context">
        <div className="section-heading context-heading">
          <div>
            <p className="eyebrow">Current workspace</p>
            <h2>{detail.data?.spec.name ?? "Choose a research campaign"}</h2>
          </div>
          <span className="bench-badge">Protected test locked</span>
        </div>
        <div className="context-fields">
          <label>
            Selected campaign
            <select
              value={id}
              disabled={!ready}
              onChange={(e) => navigate(active, e.target.value, "")}
            >
              <option value="">No campaign selected</option>
              {inventory.data?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} · {humanLabel(c.state)} ({c.done}/{c.total})
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
                  {runLabel(r)} · {humanLabel(r.state)}
                </option>
              ))}
            </select>
          </label>
          {(active === "Evaluation" ||
            active === "Explainability" ||
            active === "Replay") && (
            <label>
              Evaluation partition
              <select
                value={part}
                onChange={(e) => patch({ part: e.target.value })}
              >
                <option value="inner_validation">Development validation</option>
                <option value="validation">Historical validation</option>
              </select>
            </label>
          )}
          {active === "Replay" && (
            <label>
              Replay asset
              <select
                value={asset}
                onChange={(e) => patch({ asset: e.target.value })}
              >
                {assets.map((a) => (
                  <option key={a}>{a}</option>
                ))}
              </select>
            </label>
          )}
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
            <span>{humanLabel(detail.data.spec.profile)}</span>
            <span>{detail.data.spec.sources.map(humanLabel).join(", ")}</span>
            <span>
              {humanLabel(detail.data.status.state)} · {detail.data.status.done}
              /{detail.data.status.total} runs
            </span>
            <details>
              <summary>Experiment identity</summary>
              <code>{detail.data.fingerprint}</code>
            </details>
          </div>
        )}
      </section>
      {active === "Overview" && (
        <>
          <section className="bench-panel">
            <p className="eyebrow">Research question</p>
            <h2>Does financial AI add measurable predictive value?</h2>
            <p className="lead-copy">
              Compare causal model inputs and architectures against simple
              statistical baselines, validate over chronological folds, and
              replay the exact registered model through the deterministic
              broker.
            </p>
            {result && (
              <FindingSummary
                entry={result}
                part={part}
                onOpen={() => navigate("Evaluation")}
              />
            )}
            {!result && (
              <p className="muted">
                No matching evaluation evidence yet. Train a campaign and
                generate its development report to see the first finding here.
              </p>
            )}
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
              <button
                className="bench-button"
                onClick={() => patch({ mode: "present", step: 0 })}
              >
                Start presentation
              </button>
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
                        <td>{humanLabel(c.state)}</td>
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
          {id && (
            <details className="bench-panel">
              <summary>Protected final evaluation protocol</summary>
              <Holdout key={id} id={id} run={selectedRun} />
            </details>
          )}
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
        <DataView
          catalogue={catalogue.data}
          sources={sources.data}
          id={id}
          source={
            params.get("source") ?? detail.data?.spec.sources[0] ?? "synthetic"
          }
          feature={params.get("feature") ?? "rsi_14"}
          onSource={(source) => patch({ source })}
          onFeature={(feature) => patch({ feature })}
        />
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
              queue={inventory.data ?? []}
              onCampaign={(id) => navigate("Training", id, "")}
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
            key={id + active + part}
            id={id}
            part={part}
            entry={entry}
            changeEntry={selectEntry}
            run={evidenceRun}
            autoEvaluating={detail.data?.status.evaluation_state === "running"}
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
          <ReplayView
            key={id + selectedRun + part + asset}
            id={id}
            run={selectedRun}
            part={part}
            asset={asset}
            initialCursor={Number(params.get("cursor") ?? 0)}
            onCursor={(cursor) => patch({ cursor })}
            runs={runs.data ?? []}
            opponent={params.get("opponent") ?? ""}
            onOpponent={(opponent) => patch({ opponent: opponent || null })}
          />
        </>
      )}
    </div>
  );
}

function PartitionTimeline({
  parts,
}: {
  parts: Record<
    string,
    { first: string | null; last: string | null; rows: number }
  >;
}) {
  const rows = Object.entries(parts).filter(
    ([key, p]) => key !== "test" && p.first && p.last,
  );
  if (!rows.length) return null;
  const min = Math.min(...rows.map(([, p]) => Date.parse(p.first!))),
    max = Math.max(...rows.map(([, p]) => Date.parse(p.last!)));
  const x = (v: string) =>
    155 + (510 * (Date.parse(v) - min)) / Math.max(1, max - min);
  const height = 82 + rows.length * 32;
  return (
    <PlotFrame title="Chronological research partitions" height={height}>
      {rows.map(([name, p], i) => (
        <g key={name}>
          <text
            x={145}
            y={39 + i * 32}
            textAnchor="end"
            fontSize="11"
            fill="var(--fg)"
          >
            {humanLabel(name)}
          </text>
          <rect
            x={x(p.first!)}
            y={26 + i * 32}
            width={Math.max(3, x(p.last!) - x(p.first!))}
            height={20}
            rx={3}
            fill={name === "validation" ? "#527bd1" : "#148b82"}
          >
            <title>
              {p.first} – {p.last} · {p.rows} rows
            </title>
          </rect>
          <text
            x={665}
            y={39 + i * 32}
            textAnchor="end"
            fontSize="10"
            fill="var(--fg)"
          >
            {p.rows.toLocaleString()} rows
          </text>
        </g>
      ))}
      <text x={155} y={height - 9} fontSize="11" fill="var(--muted)">
        {new Date(min).toISOString().slice(0, 10)}
      </text>
      <text
        x={665}
        y={height - 9}
        textAnchor="end"
        fontSize="11"
        fill="var(--muted)"
      >
        {new Date(max).toISOString().slice(0, 10)}
      </text>
    </PlotFrame>
  );
}

function DataView({
  catalogue,
  sources,
  id,
  source,
  feature,
  onSource,
  onFeature,
}: {
  catalogue: Catalogue;
  sources: Source[];
  id: string;
  source: string;
  feature: string;
  onSource: (s: string) => void;
  onFeature: (f: string) => void;
}) {
  const description = catalogue.features.find((f) => f.id === feature);
  const preview = useQuery({
    queryKey: ["lab-preview", source, feature],
    queryFn: () =>
      laboratoryApi<{
        dates: string[];
        columns: string[];
        values: (number | null)[][];
        adjusted_close?: (number | null)[];
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
              up_share?: number | null;
            }
          >;
        }[]
      >(`/${id}/datasets`),
  });
  const data = preview.data;
  const [column, setColumn] = useState(0);
  const columnIndex = Math.min(
    column,
    Math.max(0, (data?.columns.length ?? 1) - 1),
  );
  const values =
    data?.values.flatMap((v, i) =>
      v[columnIndex] == null ? [] : [{ x: i, y: v[columnIndex]! }],
    ) ?? [];
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
            <select value={source} onChange={(e) => onSource(e.target.value)}>
              {sources
                .filter((s) => s.status === "ok")
                .map((s) => (
                  <option key={s.source} value={s.source}>
                    {humanLabel(s.source)}
                  </option>
                ))}
            </select>
          </label>
          <label>
            Feature
            <select
              value={feature}
              onChange={(e) => {
                setColumn(0);
                onFeature(e.target.value);
              }}
            >
              {catalogue.features
                .filter((f) => f.blind_safe)
                .map((f) => (
                  <option key={f.id} value={f.id}>
                    {humanLabel(f.id)} · {humanLabel(f.group)}
                  </option>
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
          <>
            {data.columns.length > 1 && (
              <label className="lab-field">
                Feature output column
                <select
                  value={columnIndex}
                  onChange={(e) => setColumn(Number(e.target.value))}
                >
                  {data.columns.map((c, i) => (
                    <option key={c} value={i}>
                      {humanLabel(c)}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {data.adjusted_close && (
              <LinePlot
                title="Aligned underlying market history"
                xLabel="Session"
                yLabel="Adjusted close"
                formatX={(v) => data.dates[Math.round(v)] ?? ""}
                series={[
                  {
                    name: "Adjusted close",
                    points: data.adjusted_close.flatMap((y, x) =>
                      y === null ? [] : [{ x, y }],
                    ),
                  },
                ]}
              />
            )}
            <div className="lab-two">
              <LinePlot
                title={`${data.columns[columnIndex]} · development history`}
                xLabel="Session"
                yLabel={description?.units}
                formatX={(v) => data.dates[Math.round(v)]?.slice(5) ?? ""}
                series={[
                  {
                    name: humanLabel(data.columns[columnIndex]),
                    points: values,
                  },
                ]}
              />
              <LinePlot
                title="Observed feature distribution · 12 bins"
                xLabel="Feature value"
                yLabel="Count"
                series={[{ name: "Frequency", points: bins, dots: true }]}
              />
            </div>
          </>
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
            <PartitionTimeline parts={manifests.data[0].parts} />
            {manifests.data[0].spec.target.startsWith("direction_binary") && (
              <div className="metric-strip">
                {Object.entries(manifests.data[0].parts)
                  .filter(([, p]) => typeof p.up_share === "number")
                  .map(([name, p]) => (
                    <div key={name}>
                      <span>{humanLabel(name)} · positive labels</span>
                      <strong>{numberText(p.up_share! * 100, 1)}%</strong>
                      <small>{p.rows} observations</small>
                    </div>
                  ))}
              </div>
            )}
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
