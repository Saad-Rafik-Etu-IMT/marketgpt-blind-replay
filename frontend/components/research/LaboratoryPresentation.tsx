"use client";

import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  laboratoryApi,
  type Catalogue,
  type Detail,
  type EvaluationEntry,
  type Run,
  numberText,
} from "@/lib/research/laboratory";
import {
  humanLabel,
  runLabel,
  presentationStages,
  presentationTabs,
  type Laboratory,
} from "@/lib/research/product-context";
import { FindingSummary } from "./ProductEvidence";
import { LossPlot } from "./LaboratoryTraining";
import { ConfidenceIntervals, ReplayView } from "./LaboratoryResults";
import { LinePlot } from "./ScientificPlot";

export function Presentation({
  id,
  run,
  detail,
  result,
  part,
  asset,
  catalogue,
  step,
  onStep,
  onExit,
}: {
  id: string;
  run?: Run;
  detail?: Detail;
  result?: EvaluationEntry;
  part: string;
  asset: string;
  catalogue?: Catalogue;
  step: number;
  onStep: (step: number) => void;
  onExit: (tab: Laboratory) => void;
}) {
  const index = Math.max(
    0,
    Math.min(
      presentationStages.length - 1,
      Number.isFinite(step) ? Math.floor(step) : 0,
    ),
  );
  const root = useRef<HTMLDivElement>(null);
  const record = useQuery({
    queryKey: ["lab-run", id, run?.key],
    enabled: !!id && !!run,
    queryFn: () => laboratoryApi<Run>(`/${id}/runs/${run?.key}`),
  });
  const source = detail?.spec.sources[0];
  const preview = useQuery({
    queryKey: ["presentation-preview", source],
    enabled: index === 1 && !!source,
    queryFn: () =>
      laboratoryApi<{ dates: string[]; adjusted_close: (number | null)[] }>(
        `/sources/${source}/preview?feature=rsi_14&last=150`,
      ),
  });
  useEffect(() => {
    function key(e: KeyboardEvent) {
      if (
        (e.target as HTMLElement)?.closest(
          "input,select,textarea,button,a,dialog",
        )
      )
        return;
      if (e.key === "ArrowRight" && index < 7) {
        e.preventDefault();
        onStep(index + 1);
      }
      if (e.key === "ArrowLeft" && index > 0) {
        e.preventDefault();
        onStep(index - 1);
      }
      if (e.key === "Escape" && !document.fullscreenElement)
        onExit(presentationTabs[index]);
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [index, onStep, onExit]);
  const missing = (
    <p className="bench-notice">
      No matching recorded evidence. Open the detailed laboratory to create or
      select it.
    </p>
  );
  return (
    <div className="laboratory presentation-stage" ref={root}>
      <header className="presentation-header">
        <div>
          <p className="eyebrow">Blind Replay Bench · guided demonstration</p>
          <strong>{detail?.spec.name ?? "No campaign selected"}</strong>
          <p className="muted">
            {run ? runLabel(run) : "No run selected"} · {humanLabel(part)}
          </p>
        </div>
        <div className="bench-actions">
          <button
            className="bench-button"
            onClick={() => {
              if (document.fullscreenElement)
                void document.exitFullscreen().catch(() => {});
              else void root.current?.requestFullscreen?.().catch(() => {});
            }}
          >
            Fullscreen
          </button>
          <button
            className="bench-button"
            onClick={() => onExit(presentationTabs[index])}
          >
            Exit presentation
          </button>
        </div>
      </header>
      <progress
        className="lab-progress"
        value={index + 1}
        max={8}
        aria-label="Presentation progress"
      />
      <main className="bench-panel presentation-content">
        <p className="eyebrow">Stage {index + 1} of 8</p>
        <h1>{presentationStages[index]}</h1>
        {index === 0 && (
          <>
            <h2>Can a model learn useful signals from historical markets?</h2>
            <p className="lead-copy">
              Predict an outcome using only information available at the
              decision time. Compare against simple models, then inspect how a
              deterministic virtual broker applies the decisions.
            </p>
            <p>
              No real money is at risk. Prediction quality and simulated
              financial performance are separate questions.
            </p>
          </>
        )}
        {index === 1 &&
          (detail ? (
            <>
              <h2>Chronological data, not random slices</h2>
              <p>
                We train on older data, select models using development
                validation, and review a later historical validation partition.
                The protected final test is not exposed here.
              </p>
              <ul>
                {detail.spec.sources.map((s) => (
                  <li key={s}>{humanLabel(s)}</li>
                ))}
              </ul>
              <p>
                Feature groups:{" "}
                {detail.spec.feature_profiles.map(humanLabel).join(", ")}.
              </p>
              {preview.data && (
                <LinePlot
                  title="Development market observations"
                  xLabel="Historical session"
                  yLabel="Adjusted close"
                  formatX={(v) => preview.data?.dates[Math.round(v)] ?? ""}
                  series={[
                    {
                      name: humanLabel(source ?? ""),
                      points: preview.data.adjusted_close.flatMap((y, x) =>
                        y === null ? [] : [{ x, y }],
                      ),
                    },
                  ]}
                />
              )}
            </>
          ) : (
            missing
          ))}
        {index === 2 &&
          (detail ? (
            <>
              <h2>What the model sees and predicts</h2>
              <div className="architecture-flow">
                <b>Past observations</b>
                <span>→</span>
                <b>{detail.spec.presets.map(humanLabel).join(", ")}</b>
                <span>→</span>
                <b>Prediction heads</b>
              </div>
              <ul>
                {detail.spec.targets.map((t) => (
                  <li key={t}>
                    <strong>{humanLabel(t)}</strong> —{" "}
                    {catalogue?.targets.find((x) => x.id === t)
                      ?.interpretation ?? "Registered prediction target"}
                  </li>
                ))}
              </ul>
              <p>
                Simple statistical baselines provide the reference, not a
                guarantee of investment returns.
              </p>
            </>
          ) : (
            missing
          ))}
        {index === 3 &&
          (record.data?.history?.length ? (
            <>
              <h2>Measured convergence</h2>
              <p>
                Training adjusts the model on historical data. Selection loss is
                measured on later development observations.
              </p>
              <LossPlot history={record.data.history} />
              <p>
                Selected epoch: {numberText(record.data.best_epoch, 0)} ·
                parameters: {numberText(record.data.parameter_count, 0)}
              </p>
            </>
          ) : (
            missing
          ))}
        {index === 4 &&
          (result ? (
            <>
              <FindingSummary entry={result} part={part} />
              <ConfidenceIntervals entry={result} />
            </>
          ) : (
            missing
          ))}
        {index === 5 &&
          (run?.state === "complete" ? (
            <ReplayView
              key={id + run.key + part + asset}
              id={id}
              run={run.key}
              part={part}
              asset={asset}
            />
          ) : (
            missing
          ))}
        {index === 6 &&
          (result?.importance?.length ? (
            <>
              <h2>Which inputs affected measured predictions?</h2>
              <p>
                Feature perturbation measures sensitivity, not causal importance
                in the real world.
              </p>
              <div className="lab-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Loss increase</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...result.importance]
                      .sort((a, b) => b.loss_increase - a.loss_increase)
                      .slice(0, 8)
                      .map((v) => (
                        <tr key={v.feature}>
                          <td>{humanLabel(v.feature)}</td>
                          <td>{numberText(v.loss_increase)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            missing
          ))}
        {index === 7 && (
          <>
            {result ? <FindingSummary entry={result} part={part} /> : missing}
            <h2>What this does not establish</h2>
            <ul>
              <li>
                Development evidence is not protected final test evidence.
              </li>
              <li>Historical results do not establish future profitability.</li>
              <li>
                Virtual execution depends on recorded costs, policy and risk
                assumptions.
              </li>
            </ul>
          </>
        )}
      </main>
      <footer className="presentation-footer">
        <button
          className="bench-button"
          disabled={index === 0}
          onClick={() => onStep(index - 1)}
        >
          ← Previous
        </button>
        <div className="presentation-dots">
          {presentationStages.map((s, i) => (
            <button
              key={s}
              className="bench-button"
              aria-label={s}
              aria-current={i === index ? "step" : undefined}
              onClick={() => onStep(i)}
            >
              {i + 1}
            </button>
          ))}
        </div>
        <button
          className="bench-button primary-action"
          onClick={() => (index === 7 ? onExit("Overview") : onStep(index + 1))}
        >
          {index === 7 ? "Finish" : "Next →"}
        </button>
      </footer>
      <button
        className="bench-button"
        onClick={() => onExit(presentationTabs[index])}
      >
        Inspect this stage in the laboratory
      </button>
    </div>
  );
}
