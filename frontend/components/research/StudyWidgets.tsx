"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { SeriesPlot } from "./SeriesPlot";
import { SensitivityPlot } from "./ResearchFigures";
import { api, pct } from "@/lib/research/study-api";
export { api, pct } from "@/lib/research/study-api";
type Observation = {
  inputs: number[];
  targets: string[];
  occlusion: number[];
  p: number;
  token?: string;
  outcome?: Pick<Outcome, "next" | "actions">;
};
type Outcome = {
  index: number;
  next: number;
  vote: boolean;
  actions: {
    status: string;
    target: string;
    price: number | null;
    reason: string | null;
  }[];
};
export function Decision({
  base,
  index,
  blind,
  jury = false,
}: {
  base: string;
  index: number;
  blind: boolean;
  jury?: boolean;
}) {
  const [challenge, c] = useState<Observation | null>(null),
    [out, o] = useState<Outcome | null>(null),
    [text, t] = useState(""),
    [err, x] = useState(""),
    [busy, b] = useState(false);
  const [mode, setMode] = useState("scientific");
  const q = useQuery({
    queryKey: ["case", base, index],
    queryFn: () => api<Observation>(base + "/case/" + index),
    enabled: !blind,
  });
  const obs = blind ? challenge : q.data;
  async function act(kind: "new" | "explain" | "vote", up = false) {
    b(true);
    x("");
    try {
      if (kind === "new") {
        c(await api<Observation>((jury ? "/jury" : base) + "/challenge", {}));
        o(null);
      }
      if (kind === "vote" && challenge)
        o(
          await api<Outcome>(
            (jury ? "/jury" : base) + "/vote/" + challenge.token,
            { up },
          ),
        );
      if (kind === "explain") {
        const v = await api<{ facts: string[]; source: string; check: string }>(
          base + "/explain/" + index + "?mode=" + mode,
          {},
        );
        t(v.source + " · " + v.check + ". " + v.facts.join(" "));
      }
    } catch (ex) {
      x(String(ex));
    } finally {
      b(false);
    }
  }
  return (
    <section className="bench-panel">
      <h2>{blind ? "Blind replay" : "Decision " + (index + 1)}</h2>
      {blind && (
        <button
          className="bench-button"
          disabled={busy}
          onClick={() => act("new")}
        >
          Draw blind case
        </button>
      )}
      {obs && (
        <>
          <SeriesPlot
            values={[
              100,
              ...obs.inputs.map(
                (_, i) =>
                  100 *
                  Math.exp(
                    obs.inputs
                      .slice(0, i + 1)
                      .reduce((sum, value) => sum + value, 0),
                  ),
              ),
            ]}
            label="Relative adjusted closes (starting base 100)"
            startLabel="t − 64"
            endLabel="Decision · t"
          />
          <p>
            P(up) {pct(obs.p)} · Model {obs.targets[0]} · MA 20/50{" "}
            {obs.targets[1]} · Buy-and-hold {obs.targets[2]}
          </p>
          <div className="probability-grid">
            <div>
              Price rises<strong>{pct(obs.p)}</strong>
              <meter
                value={obs.p}
                min={0}
                max={1}
                aria-label="Probability of rising"
              />
            </div>
            <div>
              Flat or falls<strong>{pct(1 - obs.p)}</strong>
              <meter
                value={1 - obs.p}
                min={0}
                max={1}
                aria-label="Probability of not rising"
              />
            </div>
          </div>
          {blind ? (
            <div className="bench-actions">
              {[true, false].map((v) => (
                <button
                  key={String(v)}
                  className="bench-button"
                  disabled={busy || !!out}
                  onClick={() => act("vote", v)}
                >
                  Vote {v ? "up" : "not up"}
                </button>
              ))}
            </div>
          ) : (
            <>
              {obs.outcome && (
                <p>
                  Observed next return {pct(Math.expm1(obs.outcome.next))} ·
                  model prediction{" "}
                  {obs.p >= 0.5 === obs.outcome.next > 0
                    ? "correct"
                    : "incorrect"}
                </p>
              )}
              <h3>Occlusion sensitivity</h3>
              <p>
                Change in P(up) when eight consecutive returns are replaced by
                the training mean. Positive values indicate that masking
                increased P(up).
              </p>
              <SensitivityPlot values={obs.occlusion} />
              <table>
                <tbody>
                  {obs.occlusion.map((v, i) => (
                    <tr key={i}>
                      <th>
                        Bars {8 * i - 63} to {8 * i - 56}
                      </th>
                      <td>{(-100 * v).toFixed(1)} pp</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <label>
                Explanation language{" "}
                <select
                  value={mode}
                  disabled={busy}
                  onChange={(e) => {
                    setMode(e.target.value);
                    t("");
                  }}
                >
                  <option value="standard">Accessible</option>
                  <option value="scientific">Scientific</option>
                </select>
              </label>{" "}
              <button
                className="bench-button"
                disabled={busy}
                onClick={() => act("explain")}
              >
                Explain from measured facts
              </button>
              {text && <p>{text}</p>}
            </>
          )}
        </>
      )}
      {out && (
        <>
          <h3>Revealed · case {out.index + 1}</h3>
          <p>
            Next return {pct(Math.expm1(out.next))} · Vote{" "}
            {out.vote === out.next > 0 ? "correct" : "incorrect"}
          </p>
          {out.actions.map((r, i) => (
            <p key={i}>
              {["Model", "MA 20/50", "Buy-and-hold"][i]} · {r.status}{" "}
              {r.price?.toFixed(2)} {r.reason}
            </p>
          ))}
        </>
      )}
      {(err || q.error) && <p role="alert">{err || String(q.error)}</p>}
    </section>
  );
}
