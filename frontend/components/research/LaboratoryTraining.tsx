"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Catalogue,
  Epoch,
  Estimate,
  laboratoryApi,
  labelText,
  Run,
  Source,
  Spec,
  Status,
  numberText,
} from "@/lib/research/laboratory";
import { LinePlot, plotColors } from "./ScientificPlot";

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
          {labelText(item.id)}
        </label>
      ))}
    </fieldset>
  );
}

export function CampaignForm({
  catalogue,
  sources,
  onCreated,
}: {
  catalogue: Catalogue;
  sources: Source[];
  onCreated: (id: string) => void;
}) {
  const [spec, set] = useState<Spec>(initial);
  const [template, setTemplate] = useState("");
  const [estimate, setEstimate] = useState<Estimate>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  function patch(values: Partial<Spec>) {
    set((old) => ({ ...old, ...values }));
    setEstimate(undefined);
  }
  const profile = catalogue.experiment_profiles.find(
    (p) => p.id === spec.profile,
  );
  async function action(start: boolean) {
    setBusy(true);
    setError("");
    try {
      if (start) {
        const result = await laboratoryApi<{ id: string }>(
          "?enqueue=true",
          spec,
        );
        onCreated(result.id);
      } else setEstimate(await laboratoryApi<Estimate>("/estimate", spec));
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
          <p className="eyebrow">Design</p>
          <h2>Configure a campaign</h2>
        </div>
        <span className="bench-badge">Development only</span>
      </div>
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
                set(selected.spec);
                setEstimate(undefined);
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
        <label>
          Training profile
          <select
            value={spec.profile}
            onChange={(e) => patch({ profile: e.target.value, epochs: null })}
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
                max_minutes: e.target.value ? Number(e.target.value) : null,
              })
            }
          />
        </label>
        <label>
          Automatic evaluation
          <select
            value={spec.auto_evaluate ? "yes" : "no"}
            onChange={(e) => patch({ auto_evaluate: e.target.value === "yes" })}
          >
            <option value="yes">Yes · inner validation only</option>
            <option value="no">Manual</option>
          </select>
        </label>
      </div>
      {template && (
        <p className="bench-notice">
          {catalogue.campaigns.find((c) => c.id === template)?.hypothesis}
        </p>
      )}
      <MultiSelect
        title="Datasets"
        choices={sources
          .filter((s) => s.status === "ok")
          .map((s) => ({ id: s.source, description: s.kind }))}
        selected={spec.sources}
        change={(items) => {
          const last = items.at(-1);
          patch({
            sources: last?.startsWith("synthetic")
              ? [last]
              : items.filter((x) => !x.startsWith("synthetic")),
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
        title="Model architectures"
        choices={Object.keys(catalogue.presets).map((id) => ({ id }))}
        selected={spec.presets}
        change={(presets) => patch({ presets })}
      />
      {!catalogue.transformer_available && (
        <p className="muted">
          Optional PyTorch is not installed. Install it to enable the reference
          Transformer.
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
      <MultiSelect
        title="Prediction targets"
        choices={catalogue.targets.map((t) => ({
          id: t.id,
          description: t.interpretation,
        }))}
        selected={spec.targets}
        change={(targets) => patch({ targets })}
      />
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
      {error && (
        <p role="alert" className="bench-notice">
          {error}
        </p>
      )}
      <div className="bench-actions">
        <button
          className="bench-button"
          disabled={busy}
          onClick={() => void action(false)}
        >
          Estimate campaign
        </button>
        <button
          className="bench-button primary-action"
          disabled={busy || !estimate?.within_budget}
          onClick={() => void action(true)}
        >
          Queue campaign
        </button>
      </div>
    </section>
  );
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

export function TrainingMonitor({
  id,
  status,
  runs,
  onRefresh,
  run,
  onRun,
}: {
  id: string;
  status: Status;
  runs: Run[];
  onRefresh: () => void;
  run: string;
  onRun: (key: string) => void;
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
        <span className="bench-badge">{labelText(status.state)}</span>
      </div>
      <p>{status.message}</p>
      <progress
        className="lab-progress"
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
      {status.current?.run && <p className="muted">{status.current.run}</p>}
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
                      {r.key}
                    </button>
                  </td>
                  <td>
                    {r.state}
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
