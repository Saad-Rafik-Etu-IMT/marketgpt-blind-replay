"use client";

import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Catalogue,
  Epoch,
  Estimate,
  laboratoryApi,
  labelText,
  Run,
  Campaign,
  Source,
  Spec,
  Status,
  numberText,
} from "@/lib/research/laboratory";
import { LinePlot, PlotFrame, plotColors } from "./ScientificPlot";
import { humanLabel, runLabel } from "@/lib/research/product-context";

const initial: Spec = {
  name: "New research campaign",
  profile: "development",
  sources: ["synthetic"],
  feature_profiles: ["returns_only_v1"],
  presets: ["small"],
  targets: ["direction_binary_v1"],
  epochs: 2,
  auto_evaluate: true,
};

function MultiSelect({
  title,
  choices,
  selected,
  change,
}: {
  title: string;
  choices: { id: string; description?: string }[];
  selected: string[];
  change: (items: string[]) => void;
}) {
  return (
    <fieldset className="lab-choice">
      <legend>{title}</legend>
      {choices.map((item) => (
        <label key={item.id} title={item.description}>
          <input
            type="checkbox"
            checked={selected.includes(item.id)}
            onChange={(e) =>
              change(
                e.target.checked
                  ? [...selected, item.id]
                  : selected.filter((x) => x !== item.id),
              )
            }
          />
          {humanLabel(item.id)}
        </label>
      ))}
    </fieldset>
  );
}

export function CampaignForm({
  catalogue,
  sources,
  onCreated,
  existingSpec,
}: {
  catalogue: Catalogue;
  sources: Source[];
  onCreated: (id: string) => void;
  existingSpec?: Spec;
}) {
  const [spec, set] = useState<Spec>(initial);
  const [template, setTemplate] = useState("");
  const [estimate, setEstimate] = useState<Estimate>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const steps = ["Objective", "Data & inputs", "Models", "Budget", "Review"];
  const [step, setStep] = useState(0);
  const revision = useRef(0);
  const [estimateRevision, setEstimateRevision] = useState(-1);
  function patch(values: Partial<Spec>) {
    revision.current++;
    set((old) => ({ ...old, ...values }));
    setEstimate(undefined);
    setError("");
  }
  const profile = catalogue.experiment_profiles.find(
    (p) => p.id === spec.profile,
  );
  async function action(start: boolean) {
    const issue = configurationError(spec, catalogue, sources, 4);
    if (issue) {
      setError(issue);
      return;
    }
    if (start && estimateRevision !== revision.current) {
      setError("The configuration changed. Estimate the exact protocol again.");
      return;
    }
    setBusy(true);
    setError("");
    const currentRevision = revision.current;
    try {
      if (start) {
        const result = await laboratoryApi<{ id: string }>(
          "?enqueue=true",
          spec,
        );
        onCreated(result.id);
        setEstimate(undefined);
        setEstimateRevision(-1);
        setStep(0);
      } else {
        const result = await laboratoryApi<Estimate>("/estimate", spec);
        if (currentRevision === revision.current) {
          setEstimate(result);
          setEstimateRevision(currentRevision);
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="bench-panel lab-form">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Experiment builder</p>
          <h2>Configure a campaign</h2>
        </div>
        <span className="bench-badge">Development only</span>
      </div>
      {existingSpec && (
        <button
          className="bench-button"
          disabled={busy}
          onClick={() => {
            revision.current++;
            set({
              ...existingSpec,
              name: existingSpec.name.slice(0, 68) + " copy",
              report_validation: false,
            });
            setEstimate(undefined);
            setEstimateRevision(-1);
            setTemplate("");
            setStep(0);
          }}
        >
          Duplicate selected campaign configuration
        </button>
      )}
      <ol className="wizard-steps" aria-label="Campaign setup progress">
        {steps.map((name, i) => (
          <li key={name}>
            <button
              className="wizard-step"
              aria-current={i === step ? "step" : undefined}
              disabled={busy || i > step}
              onClick={() => {
                setStep(i);
                setError("");
              }}
            >
              <b>{i + 1}</b>
              <span>{name}</span>
            </button>
          </li>
        ))}
      </ol>
      <fieldset disabled={busy} className="wizard-body">
        {step === 0 && (
          <>
            <h3>01 · What do you want to investigate?</h3>
            <p className="muted">
              Start from a predeclared research recipe or create a custom
              comparison.
            </p>
            <div className="lab-two">
              <label>
                Research recipe
                <select
                  value={template}
                  onChange={(e) => {
                    setTemplate(e.target.value);
                    const selected = catalogue.campaigns.find(
                      (c) => c.id === e.target.value,
                    );
                    if (selected) {
                      revision.current++;
                      set(selected.spec);
                      setEstimate(undefined);
                      setError("");
                    }
                  }}
                >
                  <option value="">Custom campaign</option>
                  {catalogue.campaigns.map((c) => (
                    <option value={c.id} key={c.id}>
                      {c.title}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Campaign name
                <input
                  value={spec.name}
                  onChange={(e) => patch({ name: e.target.value })}
                />
              </label>
            </div>
            {template && (
              <p className="bench-notice">
                {catalogue.campaigns.find((c) => c.id === template)?.hypothesis}
              </p>
            )}
          </>
        )}
        {step === 3 && (
          <>
            <h3>04 · Control the research budget</h3>
            <div className="lab-two">
              <label>
                Training profile
                <select
                  value={spec.profile}
                  onChange={(e) =>
                    patch({ profile: e.target.value, epochs: null })
                  }
                >
                  {catalogue.experiment_profiles.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.id} · {p.seeds} seed(s)
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Maximum epochs
                <input
                  type="number"
                  min={1}
                  max={profile?.max_epochs ?? 60}
                  value={spec.epochs ?? profile?.max_epochs ?? 10}
                  onChange={(e) => patch({ epochs: Number(e.target.value) })}
                />
              </label>
              <label>
                Time budget (minutes)
                <input
                  type="number"
                  min={1}
                  max={1440}
                  placeholder="Worker default"
                  value={spec.max_minutes ?? ""}
                  onChange={(e) =>
                    patch({
                      max_minutes: e.target.value
                        ? Number(e.target.value)
                        : null,
                    })
                  }
                />
              </label>
              <label>
                Automatic evaluation
                <select
                  value={spec.auto_evaluate ? "yes" : "no"}
                  onChange={(e) =>
                    patch({ auto_evaluate: e.target.value === "yes" })
                  }
                >
                  <option value="yes">Yes · inner validation only</option>
                  <option value="no">Manual</option>
                </select>
              </label>
            </div>
          </>
        )}
        {step === 1 && (
          <>
            <h3>02 · What information is available?</h3>
            <MultiSelect
              title="Datasets"
              choices={sources
                .filter((s) => s.status === "ok")
                .map((s) => ({ id: s.source, description: s.kind }))}
              selected={spec.sources}
              change={(items) => {
                const added = items.find((x) => !spec.sources.includes(x));
                patch({
                  sources: added?.startsWith("synthetic")
                    ? [added]
                    : added
                      ? items.filter((x) => !x.startsWith("synthetic"))
                      : items,
                });
              }}
            />
            <MultiSelect
              title="Feature profiles"
              choices={catalogue.feature_profiles.filter((p) => p.available)}
              selected={spec.feature_profiles}
              change={(feature_profiles) => patch({ feature_profiles })}
            />
            <MultiSelect
              title="Prediction targets"
              choices={catalogue.targets.map((t) => ({
                id: t.id,
                description: t.interpretation,
              }))}
              selected={spec.targets}
              change={(targets) => patch({ targets })}
            />
            <div className="wizard-selection">
              {catalogue.feature_profiles
                .filter((p) => spec.feature_profiles.includes(p.id))
                .map((p) => (
                  <p key={p.id}>
                    <strong>{humanLabel(p.id)}</strong> ·{" "}
                    {p.column_count ?? "—"} inputs · {p.description}
                  </p>
                ))}
              {catalogue.targets
                .filter((t) => spec.targets.includes(t.id))
                .map((t) => (
                  <p key={t.id}>
                    <strong>{humanLabel(t.id)}</strong> · {t.interpretation}
                  </p>
                ))}
            </div>
          </>
        )}
        {step === 2 && (
          <>
            <h3>03 · Choose comparable model families</h3>
            <MultiSelect
              title="Model architectures"
              choices={Object.keys(catalogue.presets).map((id) => ({ id }))}
              selected={spec.presets}
              change={(presets) => patch({ presets })}
            />
            {!catalogue.transformer_available && (
              <p className="muted">
                Optional PyTorch is not installed. Install it to enable the
                reference Transformer.
              </p>
            )}
            {spec.presets.some((p) => p.startsWith("transformer")) && (
              <label className="lab-field">
                Transformer device
                <select
                  value={spec.device ?? "cpu"}
                  onChange={(e) => patch({ device: e.target.value })}
                >
                  <option value="cpu">CPU</option>
                  <option value="auto">Auto · compatible GPU or CPU</option>
                  <option value="cuda">CUDA · require GPU</option>
                </select>
              </label>
            )}
            <div className="architecture-grid">
              {spec.presets.map((preset) => {
                const info = catalogue.presets[preset] as
                  | { architecture?: { hidden?: number[] } | string }
                  | undefined;
                const hidden =
                  typeof info?.architecture === "object"
                    ? info.architecture.hidden
                    : undefined;
                return (
                  <article key={preset}>
                    <h4>{humanLabel(preset)}</h4>
                    <ArchitectureDiagram
                      hidden={hidden ?? []}
                      heads={spec.targets.length}
                      activation={
                        typeof info?.architecture === "object" &&
                        "activation" in info.architecture
                          ? String(info.architecture.activation)
                          : undefined
                      }
                      transformer={typeof info?.architecture === "string"}
                    />
                    <p className="muted">
                      Exact input counts and trainable parameters are estimated
                      during review.
                    </p>
                  </article>
                );
              })}
            </div>
          </>
        )}
        {step === 3 && (
          <details className="supporting-details">
            <summary>Search space and optimization</summary>
            <div className="lab-two">
              <label>
                Search method
                <select
                  value={spec.search?.method ?? "none"}
                  onChange={(e) =>
                    patch({
                      search:
                        e.target.value === "none"
                          ? null
                          : {
                              method: e.target.value as "random" | "grid",
                              budget: 3,
                            },
                    })
                  }
                >
                  <option value="none">Preset comparison</option>
                  <option value="random">Bounded random search</option>
                  <option value="grid">Bounded grid</option>
                </select>
              </label>
              {spec.search && (
                <>
                  <label>
                    Candidate budget
                    <input
                      type="number"
                      min={1}
                      max={60}
                      value={spec.search.budget}
                      onChange={(e) =>
                        patch({
                          search: {
                            ...spec.search!,
                            budget: Number(e.target.value),
                          },
                        })
                      }
                    />
                  </label>
                  {(
                    [
                      "learning_rates",
                      "weight_decays",
                      "dropouts",
                      "batch_sizes",
                    ] as const
                  ).map((key) => (
                    <label key={key}>
                      {labelText(key)}
                      <input
                        placeholder="Use defaults"
                        defaultValue={spec.search?.[key]?.join(",") ?? ""}
                        onBlur={(e) =>
                          patch({
                            search: {
                              ...spec.search!,
                              [key]: e.target.value.trim()
                                ? e.target.value.split(",").map(Number)
                                : undefined,
                            },
                          })
                        }
                      />
                    </label>
                  ))}
                </>
              )}
            </div>
            <p className="muted">
              Each candidate uses the same selected feature profiles and seeds.
              Maximum {catalogue.max_runs} runs per campaign.
            </p>
          </details>
        )}
        {step === 4 && (
          <>
            <h3>05 · Review the exact experiment</h3>
            <dl className="review-grid">
              <div>
                <dt>Campaign</dt>
                <dd>{spec.name}</dd>
              </div>
              <div>
                <dt>Research profile</dt>
                <dd>
                  {humanLabel(spec.profile)} · {profile?.seeds ?? "—"} seeds
                </dd>
              </div>
              <div>
                <dt>Datasets</dt>
                <dd>{spec.sources.map(humanLabel).join(", ")}</dd>
              </div>
              <div>
                <dt>Feature profiles</dt>
                <dd>{spec.feature_profiles.map(humanLabel).join(", ")}</dd>
              </div>
              <div>
                <dt>Models</dt>
                <dd>{spec.presets.map(humanLabel).join(", ")}</dd>
              </div>
              <div>
                <dt>Targets</dt>
                <dd>{spec.targets.map(humanLabel).join(", ")}</dd>
              </div>
              <div>
                <dt>Maximum epochs</dt>
                <dd>{spec.epochs ?? profile?.max_epochs ?? "Default"}</dd>
              </div>
              <div>
                <dt>Search</dt>
                <dd>
                  {spec.search
                    ? `${humanLabel(spec.search.method)} · ${spec.search.budget} candidates`
                    : "Preset comparison"}
                </dd>
              </div>
            </dl>
            <p className="bench-notice">
              Selection and training use development data. Protected final test
              values will not be evaluated.
            </p>
            {estimate && (
              <div className="metric-strip">
                <div>
                  <span>Run count</span>
                  <strong>{estimate.runs}</strong>
                </div>
                <div>
                  <span>Estimated training time</span>
                  <strong>{numberText(estimate.estimated_seconds)} s</strong>
                </div>
                <div>
                  <span>Budget</span>
                  <strong>
                    {estimate.within_budget ? "Within limit" : "Exceeds limit"}
                  </strong>
                </div>
              </div>
            )}
            {estimate?.per_run?.length ? (
              <details className="supporting-details">
                <summary>Estimated run parameters</summary>
                <div className="lab-table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Run</th>
                        <th>Inputs</th>
                        <th>Parameters</th>
                        <th>Epochs</th>
                      </tr>
                    </thead>
                    <tbody>
                      {estimate.per_run.map((r) => (
                        <tr key={r.key}>
                          <td>{runLabel(r.key)}</td>
                          <td>{r.inputs}</td>
                          <td>{numberText(r.parameters, 0)}</td>
                          <td>{r.epochs}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            ) : null}
          </>
        )}
      </fieldset>
      {error && (
        <p role="alert" className="bench-notice">
          {error}
        </p>
      )}
      <div className="bench-actions">
        {step > 0 && (
          <button
            className="bench-button"
            disabled={busy}
            onClick={() => {
              setStep(step - 1);
              setError("");
            }}
          >
            Previous step
          </button>
        )}
        {step < 4 && (
          <button
            className="bench-button primary-action"
            disabled={busy}
            onClick={() => {
              const issue = configurationError(spec, catalogue, sources, step);
              if (issue) {
                setError(issue);
                return;
              }
              setError("");
              setStep(step + 1);
            }}
          >
            Continue
          </button>
        )}
        {step === 4 && (
          <>
            <button
              className="bench-button"
              disabled={busy}
              onClick={() => void action(false)}
            >
              Estimate campaign
            </button>
            <button
              className="bench-button primary-action"
              disabled={
                busy ||
                !estimate?.within_budget ||
                estimateRevision !== revision.current
              }
              onClick={() => void action(true)}
            >
              Queue campaign
            </button>
          </>
        )}
      </div>
    </section>
  );
}

export function configurationError(
  spec: Spec,
  catalogue: Catalogue,
  sources: Source[],
  stage: number,
): string {
  if (!spec.name.trim()) return "Give this experiment a meaningful name.";
  if (stage < 1) return "";
  if (
    !spec.sources.length ||
    spec.sources.some(
      (s) => !sources.some((x) => x.source === s && x.status === "ok"),
    )
  )
    return "Choose at least one available dataset.";
  if (!spec.feature_profiles.length || !spec.targets.length)
    return "Select at least one feature profile and prediction target.";
  if (stage < 2) return "";
  if (!spec.presets.length) return "Select at least one model architecture.";
  const transformer = spec.presets.some((p) => p.startsWith("transformer"));
  if (
    transformer &&
    (spec.feature_profiles.join() !== "returns_only_v1" ||
      spec.targets.join() !== "direction_binary_v1")
  )
    return "The reference Transformer requires returns-only inputs and the executed-return direction target. Adjust Data & inputs.";
  if (transformer && !catalogue.transformer_available)
    return "Install optional PyTorch to enable the Transformer.";
  if (stage < 3) return "";
  const max =
    catalogue.experiment_profiles.find((p) => p.id === spec.profile)
      ?.max_epochs ?? 60;
  if (
    spec.epochs != null &&
    (!Number.isInteger(spec.epochs) || spec.epochs < 1 || spec.epochs > max)
  )
    return `Use between 1 and ${max} epochs for this profile.`;
  if (transformer && spec.search)
    return "Numeric hyperparameter search is MLP-only. Disable search for this Transformer comparison.";
  return "";
}

export function LossPlot({ history }: { history: Epoch[] }) {
  if (!history.length)
    return (
      <p className="muted">Epoch metrics will appear as training progresses.</p>
    );
  return (
    <LinePlot
      title="Convergence · training vs selection loss"
      xLabel="Epoch"
      yLabel="Loss"
      series={[
        {
          name: "Training",
          color: plotColors[0],
          points: history.map((x) => ({ x: x.epoch, y: x.train_loss })),
        },
        {
          name: "Inner validation",
          color: plotColors[1],
          points: history.map((x) => ({ x: x.epoch, y: x.select_loss })),
        },
        {
          name: "Retained checkpoint",
          color: plotColors[2],
          scatter: true,
          dots: true,
          points: history
            .filter((x) => x.selected)
            .map((x) => ({ x: x.epoch, y: x.select_loss })),
        },
      ]}
    />
  );
}

function ArchitectureDiagram({
  hidden,
  heads,
  inputs,
  activation,
  transformer = false,
}: {
  hidden: number[];
  heads: number;
  inputs?: number;
  activation?: string;
  transformer?: boolean;
}) {
  const [chosen, setSelected] = useState(0);
  const layers = [
    { label: "Inputs", width: inputs ?? null },
    ...hidden.map((width, i) => ({ label: `Hidden ${i + 1}`, width })),
    ...(transformer ? [{ label: "Attention encoder", width: null }] : []),
    { label: "Output heads", width: heads },
  ];
  const selected = Math.min(chosen, layers.length - 1);
  const layout = layers.map((l, i) => ({
    x: 50 + (600 * i) / (layers.length - 1),
    nodes: Array.from(
      { length: Math.min(6, l.width ?? 4) },
      (_, j) => 57 + (j * 84) / Math.max(1, Math.min(6, l.width ?? 4) - 1),
    ),
  }));
  return (
    <div>
      <PlotFrame title="Model architecture · schematic" height={200}>
        {layout
          .slice(1)
          .flatMap((l, i) =>
            l.nodes.flatMap((y, j) =>
              layout[i].nodes.map((from, k) => (
                <line
                  key={`${i}-${j}-${k}`}
                  x1={layout[i].x}
                  x2={l.x}
                  y1={from}
                  y2={y}
                  stroke="var(--line)"
                  opacity=".7"
                />
              )),
            ),
          )}
        {layout.map((l, i) => (
          <g key={i}>
            {l.nodes.map((y, j) => (
              <circle
                key={j}
                cx={l.x}
                cy={y}
                r="6"
                fill={i === selected ? plotColors[0] : plotColors[2]}
              />
            ))}
            <text
              x={l.x}
              y="35"
              fontSize="10"
              textAnchor="middle"
              fill="var(--fg)"
            >
              {layers[i].label}
            </text>
            <text
              x={l.x}
              y="166"
              fontSize="11"
              textAnchor="middle"
              fill="var(--muted)"
            >
              {layers[i].width ?? "—"}
            </text>
          </g>
        ))}
        <text
          x="350"
          y="190"
          textAnchor="middle"
          fontSize="10"
          fill="var(--muted)"
        >
          Representative nodes · exact widths in labels
        </text>
      </PlotFrame>
      <div className="bench-actions">
        {layers.map((l, i) => (
          <button
            key={i}
            className="bench-button"
            aria-pressed={selected === i}
            onClick={() => setSelected(i)}
          >
            {l.label}
          </button>
        ))}
      </div>
      <p className="muted">
        {layers[selected].label}:{" "}
        {layers[selected].width ?? "depends on registered feature contract"} ·{" "}
        {selected > 0 && selected < layers.length - 1
          ? (activation ??
            (transformer ? "attention" : "activation defined by preset"))
          : "input/output representation"}
      </p>
    </div>
  );
}

export function TrainingMonitor({
  id,
  status,
  runs,
  onRefresh,
  run,
  onRun,
  queue = [],
  onCampaign,
}: {
  id: string;
  status: Status;
  runs: Run[];
  onRefresh: () => void;
  run: string;
  onRun: (key: string) => void;
  queue?: Campaign[];
  onCampaign?: (id: string) => void;
}) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const key = status.state === "running" ? status.current?.run : run;
  const progress = useQuery({
    queryKey: ["lab-progress", id, key],
    enabled: !!key,
    queryFn: () =>
      laboratoryApi<{ history: Epoch[] }>(`/${id}/runs/${key}/progress`),
    refetchInterval: status.state === "running" ? 1500 : false,
  });
  const selected = useQuery({
    queryKey: ["lab-run", id, run],
    enabled: runs.some((r) => r.key === run && r.state === "complete"),
    queryFn: () => laboratoryApi<Run>(`/${id}/runs/${run}`),
  });
  async function action(method: string) {
    setBusy(true);
    setError("");
    try {
      await laboratoryApi(`/${id}/${method}`, {});
      onRefresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="bench-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Execution</p>
          <h2>Campaign monitor</h2>
        </div>
        <span className="bench-badge">{humanLabel(status.state)}</span>
      </div>
      <p>{status.message}</p>
      <progress
        className="lab-progress"
        aria-label="Campaign completion"
        value={status.done}
        max={Math.max(1, status.total)}
      />
      <div className="metric-strip">
        <div>
          <span>Runs</span>
          <strong>
            {status.done}/{status.total}
          </strong>
        </div>
        <div>
          <span>Current epoch</span>
          <strong>
            {status.current?.epoch ?? "—"}/{status.current?.epochs ?? "—"}
          </strong>
        </div>
        <div>
          <span>Training engine</span>
          <strong>
            {status.current?.run.includes("transformer")
              ? "PyTorch"
              : "NumPy / CPU"}
          </strong>
        </div>
      </div>
      {status.current?.run && (
        <p className="muted">{runLabel(status.current.run)}</p>
      )}
      {status.current && (
        <div className="metric-strip">
          <div>
            <span>Measured training loss</span>
            <strong>{numberText(status.current.train_loss)}</strong>
          </div>
          <div>
            <span>Measured selection loss</span>
            <strong>{numberText(status.current.select_loss)}</strong>
          </div>
        </div>
      )}
      <p className="muted">
        Actual device: {selected.data?.device ?? "not yet recorded"} · GPU
        utilization is not instrumented.
      </p>
      {status.current?.elapsed_seconds !== undefined ||
      selected.data?.telemetry?.length ? (
        <div className="metric-strip">
          <div>
            <span>Elapsed (s)</span>
            <strong>
              {numberText(
                status.current?.elapsed_seconds ?? selected.data?.seconds,
                1,
              )}
            </strong>
          </div>
          <div>
            <span>Process CPU (% of one core)</span>
            <strong>
              {numberText(
                status.current?.cpu_core_percent ??
                  selected.data?.telemetry?.at(-1)?.cpu_core_percent,
                1,
              )}
            </strong>
          </div>
          <div>
            <span>Peak RSS MiB (when supported)</span>
            <strong>
              {numberText(
                status.current?.peak_rss_mib ??
                  selected.data?.telemetry?.at(-1)?.peak_rss_mib,
                1,
              )}
            </strong>
          </div>
          <div>
            <span>Estimated remaining run time (s)</span>
            <strong>
              {status.state === "running"
                ? numberText(status.current?.estimated_remaining_seconds, 1)
                : "—"}
            </strong>
          </div>
        </div>
      ) : null}
      {queue.filter((q) => ["queued", "running"].includes(q.state)).length >
        0 && (
        <div className="queue-strip">
          <strong>Training queue</strong>
          {queue
            .filter((q) => ["queued", "running"].includes(q.state))
            .map((q) => (
              <button
                key={q.id}
                className="bench-button"
                aria-pressed={q.id === id}
                onClick={() => onCampaign?.(q.id)}
              >
                {q.name} · {humanLabel(q.state)} · {q.done}/{q.total}
              </button>
            ))}
        </div>
      )}
      {status.evaluation_state && (
        <p className="muted">Automatic evaluation: {status.evaluation_state}</p>
      )}
      <div className="bench-actions">
        {["running", "queued"].includes(status.state) && (
          <button
            className="bench-button"
            disabled={busy}
            onClick={() => void action("cancel")}
          >
            Cancel campaign
          </button>
        )}
        {[
          "interrupted",
          "cancelled",
          "failed",
          "stopped",
          "complete_with_failures",
        ].includes(status.state) && (
          <button
            className="bench-button primary-action"
            disabled={busy}
            onClick={() => void action("resume?enqueue=true")}
          >
            Resume remaining runs
          </button>
        )}
      </div>
      {error && <p role="alert">{error}</p>}
      <LossPlot history={progress.data?.history ?? []} />
      {progress.data?.history.length
        ? (() => {
            const history = progress.data.history,
              last = history.at(-1)!,
              best = history.reduce((a, b) =>
                b.select_loss < a.select_loss ? b : a,
              );
            return (
              <div className="finding-card">
                <strong>Selection diagnostics</strong>
                <p>
                  Best measured selection epoch {best.epoch} · loss{" "}
                  {numberText(best.select_loss)}. Current selection–training gap{" "}
                  {numberText(last.select_loss - last.train_loss)}.
                </p>
                {history.length > 2 &&
                  last.select_loss > best.select_loss &&
                  last.train_loss < best.train_loss && (
                    <p className="bench-notice">
                      Selection loss has risen while training loss fell after
                      its minimum. Possible overfitting; not a statistical
                      diagnosis.
                    </p>
                  )}
                {selected.data?.stopped_early && (
                  <p>
                    Early stopping retained epoch {selected.data.best_epoch}.
                  </p>
                )}
              </div>
            );
          })()
        : null}
      {selected.data?.architecture && (
        <details>
          <summary>Explore retained architecture</summary>
          <ArchitectureDiagram
            hidden={
              Array.isArray(selected.data.architecture.hidden)
                ? (selected.data.architecture.hidden as number[])
                : []
            }
            inputs={selected.data.inputs}
            heads={
              Array.isArray(selected.data.architecture.heads)
                ? selected.data.architecture.heads.length
                : 1
            }
            activation={
              typeof selected.data.architecture.activation === "string"
                ? selected.data.architecture.activation
                : undefined
            }
            transformer={!!selected.data.architecture.encoder}
          />
        </details>
      )}
      {selected.data?.telemetry?.length ? (
        <details>
          <summary>Recorded worker telemetry</summary>
          <LinePlot
            title="Training elapsed time"
            xLabel="Epoch"
            yLabel="Seconds"
            series={[
              {
                name: "Elapsed",
                points: selected.data.telemetry.map((t) => ({
                  x: t.epoch,
                  y: t.elapsed_seconds,
                })),
              },
            ]}
          />
        </details>
      ) : null}
      {progress.data?.history.some(
        (h) => typeof h.learning_rate === "number",
      ) && (
        <details className="supporting-details">
          <summary>Optimization schedule</summary>
          <LinePlot
            title="Recorded learning rate"
            xLabel="Epoch"
            yLabel="Learning rate"
            series={[
              {
                name: "Learning rate",
                points: progress.data.history.flatMap((h) =>
                  h.learning_rate === undefined
                    ? []
                    : [{ x: h.epoch, y: h.learning_rate }],
                ),
              },
            ]}
          />
        </details>
      )}
      {selected.data && (
        <details className="supporting-details">
          <summary>Retained checkpoint and exact hyperparameters</summary>
          <p>
            Selected epoch {selected.data.best_epoch ?? "—"} ·{" "}
            {selected.data.epochs_run ?? "—"} epochs executed ·{" "}
            {numberText(selected.data.parameter_count, 0)} parameters.
          </p>
          {selected.data.training && (
            <pre className="technical-json">
              {JSON.stringify(selected.data.training, null, 2)}
            </pre>
          )}
          <code>{selected.data.key}</code>
        </details>
      )}
      {runs.length > 0 && (
        <div className="lab-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Best epoch</th>
                <th>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.key}>
                  <td>
                    <button
                      className="bench-button"
                      aria-pressed={r.key === run}
                      onClick={() => onRun(r.key)}
                    >
                      {runLabel(r)}
                    </button>
                  </td>
                  <td>
                    {humanLabel(r.state)}
                    {r.error && <small>{r.error}</small>}
                  </td>
                  <td>{r.best_epoch ?? "—"}</td>
                  <td>{numberText(r.parameter_count, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {selected.data?.walk_forward?.length ? (
        <>
          <h3>Chronological folds</h3>
          <div className="lab-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Start</th>
                  <th>Fit rows</th>
                  <th>Selection rows</th>
                  <th>Evaluation rows</th>
                  <th>Best epoch</th>
                </tr>
              </thead>
              <tbody>
                {selected.data.walk_forward.map((f) => (
                  <tr key={f.fold}>
                    <td>{f.start}</td>
                    <td>{f.train_rows}</td>
                    <td>{f.selection_rows ?? "—"}</td>
                    <td>{f.evaluated_rows}</td>
                    <td>{f.best_epoch ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </section>
  );
}
