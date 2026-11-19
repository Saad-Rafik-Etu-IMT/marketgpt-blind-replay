"use client";

import { useRef, useState } from "react";
import fixture from "@/lib/research/comparison-demo.json";
import { parseComparison, type Comparison } from "@/lib/research/comparison";
import { SeriesPlot } from "@/components/research/SeriesPlot";
import { TrainingCurve } from "@/components/research/TrainingCurve";

const pct = (x: number) => (100 * x).toFixed(1) + "%";
const demo = parseComparison(JSON.stringify(fixture));

function Review({ run }: { run: Comparison }) {
  const [index, select] = useState(0);
  const [opponent, compare] = useState(1);
  const [revealed, reveal] = useState(false);
  const [guess, vote] = useState<boolean | null>(null);
  const c = run.cases[index];
  const players = [0, opponent];
  const errors = run.cases
    .map((x, i) => i)
    .filter((i) => run.cases[i].p >= 0.5 !== run.cases[i].next > 0)
    .sort(
      (a, b) => Math.abs(run.cases[b].p - 0.5) - Math.abs(run.cases[a].p - 0.5),
    );
  const disagreements = run.cases
    .map((x, i) => i)
    .filter(
      (i) =>
        run.cases[i].actions[0].target !==
        run.cases[i].actions[opponent].target,
    );
  function inspect(i: number) {
    select(i);
    reveal(false);
    vote(null);
  }
  const window = run.returns.slice(index, index + 64);
  const prices = [100];
  for (const r of window) prices.push(prices[prices.length - 1] * Math.exp(r));
  const n = run.cases.length;
  const alwaysUp = run.cases.filter((x) => x.next > 0).length / n;
  return (
    <>
      <section className="bench-panel">
        <h2>Experiment</h2>
        <p>
          <strong>
            {run.origin === "synthetic"
              ? "Synthetic market · trained MLP"
              : "Imported experimental run"}
          </strong>{" "}
          · Seed {run.seed} · {run.profile}
        </p>
        <p>
          Train through {run.trainEnd} ({run.trainCount} windows) → embargo →
          validation {run.validationStart} to {run.validationEnd} ({n}{" "}
          decisions).
        </p>
        <details>
          <summary>Training and provenance</summary>
          <p>
            65 adjusted closes → 64 log returns → train-only normalization →
            NumPy MLP (32 hidden units). Lowest validation loss selects the
            checkpoint. Target long when P(up) ≥ 0.5.
          </p>
          <p style={{ overflowWrap: "anywhere" }}>
            Dataset {run.dataset}
            <br />
            Checkpoint {run.checkpoint}
            <br />
            Code {run.commit}
          </p>
          <p>
            Contribution history:{" "}
            <a href="https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/commits/main/backend/src/data">
              Data
            </a>{" "}
            ·{" "}
            <a href="https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/commits/main/backend/learning/mlp">
              MLP and training
            </a>{" "}
            ·{" "}
            <a href="https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/commits/main/backend/src/eval">
              Execution
            </a>{" "}
            ·{" "}
            <a href="https://github.com/Saad-Rafik-Etu-IMT/marketgpt-blind-replay/commits/main/frontend">
              Cockpit
            </a>
          </p>
          <TrainingCurve
            run={{
              id: "comparison",
              model: "MLP",
              dataset: run.dataset,
              split: run.split,
              checkpoint: run.checkpoint,
              seed: run.seed,
              lossName: "Cross-entropy",
              curves: run.curves,
            }}
          />
        </details>
      </section>
      <section className="bench-panel">
        <h2>
          Blind decision · {index + 1} / {n}
        </h2>
        <div className="bench-actions">
          <label>
            Compare MLP with{" "}
            <select
              value={opponent}
              onChange={(e) => {
                compare(Number(e.target.value));
                inspect(index);
              }}
            >
              {run.strategies.slice(1).map((x, i) => (
                <option value={i + 1} key={i}>
                  {x.name}
                </option>
              ))}
            </select>
          </label>
          <button
            className="bench-button"
            disabled={!disagreements.length}
            onClick={() => inspect(disagreements[0])}
          >
            First disagreement ({disagreements.length})
          </button>
          <button
            className="bench-button"
            disabled={!errors.length}
            onClick={() => inspect(errors[0])}
          >
            Most confident error ({errors.length})
          </button>
          <button
            className="bench-button"
            onClick={() =>
              inspect(
                ((Math.imul(run.seed + 1, 1664525) + 1013904223) >>> 0) % n,
              )
            }
          >
            Seeded case
          </button>
        </div>
        <SeriesPlot values={prices} label="Relative adjusted closes through the decision, starting at 100" startLabel="t − 64" endLabel="Decision · t" />
        <details>
          <summary>Inspect the 64 inputs and train-fitted tokens</summary>
          <p>
            The MLP consumes normalized log returns. The tokenizer uses{" "}
            {run.bins.length + 1} quantile bins fitted on the training windows.
          </p>
          <div className="table-scroll" style={{ maxHeight: 240 }}>
            <table>
              <thead>
                <tr>
                  <th>Offset</th>
                  <th>Relative close</th>
                  <th>Log return</th>
                  <th>Token</th>
                </tr>
              </thead>
              <tbody>
                {window.map((v, i) => (
                  <tr key={i}>
                    <td>{i - 63}</td>
                    <td>{prices[i + 1].toFixed(3)}</td>
                    <td>{v.toFixed(6)}</td>
                    <td>{run.bins.filter((edge) => v >= edge).length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
        <p>
          Relative adjusted close · first bar 100 · latest{" "}
          {prices[64].toFixed(2)}. MLP P(up) <strong>{pct(c.p)}</strong> →{" "}
          {c.actions[0].target}. {run.strategies[opponent].name} →{" "}
          <strong>{c.actions[opponent].target}</strong>.
        </p>
        {opponent === 1 && (
          <p>
            Short / long moving average (20 / 50 bars):{" "}
            {[20, 50]
              .map((k) =>
                (prices.slice(-k).reduce((a, b) => a + b, 0) / k).toFixed(2),
              )
              .join(" / ")}
            .
          </p>
        )}
        {!revealed ? (
          <div className="bench-actions">
            Your prediction{" "}
            {([true, false] as const).map((v) => (
              <button
                className="bench-button"
                key={String(v)}
                aria-pressed={guess === v}
                onClick={() => vote(v)}
              >
                {v ? "Up" : "Not up"}
                {guess === v ? " ✓" : ""}
              </button>
            ))}
            <button className="bench-button" onClick={() => reveal(true)}>
              Reveal next session
            </button>
          </div>
        ) : (
          <div aria-live="polite">
            <h3>Next session</h3>
            <p>
              Adjusted close return <strong>{pct(Math.expm1(c.next))}</strong>.{" "}
              {guess !== null &&
                (guess === c.next > 0
                  ? "Your prediction was correct."
                  : "Your prediction was incorrect.")}
            </p>
            {players.map((j) => (
              <p key={j}>
                {run.strategies[j].name} · {c.actions[j].action} ·{" "}
                {c.actions[j].status}
                {c.actions[j].price !== null &&
                  " @ " + c.actions[j].price?.toFixed(2)}{" "}
                {c.actions[j].reason}
              </p>
            ))}
          </div>
        )}
        <div className="bench-actions">
          <button
            className="bench-button"
            disabled={!index}
            onClick={() => inspect(index - 1)}
          >
            Previous case
          </button>
          <button
            className="bench-button"
            disabled={index === n - 1}
            onClick={() => inspect(index + 1)}
          >
            Next case
          </button>
        </div>
      </section>
      <details className="bench-panel">
        <summary>Review full validation results</summary>
        <p>
          MLP accuracy {pct(1 - errors.length / n)} · always-up accuracy{" "}
          {pct(alwaysUp)} · MLP macro-F1 {run.macroF1.toFixed(3)}.
        </p>
        <p>
          Accuracy difference vs always up, paired 95% block-bootstrap interval
          [{run.interval.map(pct).join(", ")}].{" "}
          {run.interval[0] <= 0 &&
            run.interval[1] >= 0 &&
            "No measurable difference on this validation run."}
        </p>
        <h3>Calibration</h3>
        <table>
          <thead>
            <tr>
              <th>Mean P(up)</th>
              <th>Observed up</th>
              <th>Cases</th>
            </tr>
          </thead>
          <tbody>
            {run.calibration.map((b, i) => (
              <tr key={i}>
                <td>{pct(b.predicted)}</td>
                <td>{pct(b.observed)}</td>
                <td>{b.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Simulated execution</h3>
        {run.origin === "experimental" ? (
          <p>
            Historical portfolio comparisons are withheld until dividend and
            split accounting is versioned. Recorded individual orders remain
            available in the reveal.
          </p>
        ) : (
          <>
            <p>
              Raw fills and final-close valuation. Cash distributions and splits
              are not accounted for.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Return</th>
                  <th>Max drawdown</th>
                </tr>
              </thead>
              <tbody>
                {run.strategies.map((s, i) => (
                  <tr key={i}>
                    <td>{s.name}</td>
                    <td>{pct(s.gain)}</td>
                    <td>{pct(s.dd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {errors.length > 0 && (
          <label>
            Inspect an error{" "}
            <select
              value=""
              onChange={(e) => {
                inspect(Number(e.target.value));
                document.getElementById("main-content")?.scrollIntoView();
              }}
            >
              <option value="" disabled>
                Select case
              </option>
              {errors.map((i) => (
                <option value={i} key={i}>
                  Case {i + 1} · P(up) {pct(run.cases[i].p)}
                </option>
              ))}
            </select>
          </label>
        )}
      </details>
    </>
  );
}

export default function ComparisonPage() {
  const [run, setRun] = useState<Comparison | null>(null);
  const [error, setError] = useState("");
  const version = useRef(0);
  return (
    <>
      <section className="bench-panel">
        <div className="bench-actions">
          <button
            className="bench-button primary-action"
            onClick={() => {
              version.current++;
              setError("");
              setRun(demo);
            }}
          >
            Open trained synthetic demo
          </button>
          <label className="import-control">
            Open validation JSON
            <input
              type="file"
              accept=".json"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (!file) return;
                const id = ++version.current;
                setRun(null);
                setError("");
                try {
                  if (file.size > 2097152)
                    throw Error("Maximum file size is 2 MiB.");
                  const parsed = parseComparison(await file.text());
                  if (id === version.current) setRun(parsed);
                } catch (err) {
                  if (id === version.current)
                    setError(
                      err instanceof Error ? err.message : "Invalid file",
                    );
                }
              }}
            />
          </label>
        </div>
        {error && (
          <p role="alert" aria-label="Comparison import error">
            {error}
          </p>
        )}
        {!run && (
          <p>
            Open the offline example or import a comparison exported from your
            local market data.
          </p>
        )}
      </section>
      {run && <Review key={version.current} run={run} />}
    </>
  );
}
