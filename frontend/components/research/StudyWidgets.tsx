"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { API_BASE_URL } from "@/lib/constants";
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
export const pct = (x: number) => (100 * x).toFixed(1) + "%";
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(
    (path.startsWith("/jury/")
      ? (process.env.NEXT_PUBLIC_JURY_API_BASE_URL || "http://localhost:8001") + "/api"
      : API_BASE_URL + "/api/studies") + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!res.ok) {
    const msg = await res.json().catch(() => ({}));
    throw Error(
      typeof msg.detail === "string"
        ? msg.detail
        : "Request failed (" + res.status + ")",
    );
  }
  return res.json();
}
export function Plot({ values, label }: { values: number[]; label: string }) {
  if (!values.length) return null;
  const low = Math.min(...values),
    high = Math.max(...values);
  return (
    <figure>
      <figcaption>
        {label} · {values[0].toFixed(3)} → {values.at(-1)?.toFixed(3)}
      </figcaption>
      <svg
        role="img"
        aria-label={label}
        viewBox="0 0 600 100"
        style={{ width: "100%", maxHeight: 140 }}
      >
        <polyline
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          points={values
            .map((v, i) =>
              [
                4 + (592 * i) / Math.max(1, values.length - 1),
                94 - (88 * (v - low)) / (high - low || 1),
              ].join(","),
            )
            .join(" ")}
        />
      </svg>
    </figure>
  );
}
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
          base + "/explain/" + index,
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
          <Plot
            values={obs.inputs.map(
              (_, i) =>
                100 *
                Math.exp(obs.inputs.slice(0, i + 1).reduce((a, b) => a + b, 0)),
            )}
            label="Relative adjusted closes (starting base 100)"
          />
          <p>
            P(up) {pct(obs.p)} · Model {obs.targets[0]} · MA 20/50{" "}
            {obs.targets[1]} · Buy-and-hold {obs.targets[2]}
          </p>
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
                  Observed next return {pct(Math.expm1(obs.outcome.next))} · model
                  prediction {(obs.p >= 0.5) === (obs.outcome.next > 0) ? "correct" : "incorrect"}
                </p>
              )}
              <h3>Occlusion sensitivity</h3>
              <p>
                Change in P(up) when eight consecutive returns are replaced by
                the training mean. Positive values indicate that masking
                increased P(up).
              </p>
              <table>
                <tbody>
                  {obs.occlusion.map((v, i) => (
                    <tr key={i}>
                      <th>
                        Bars {8 * i - 63} to {8 * i - 56}
                      </th>
                      <td>{pct(-v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
            {out.vote === (out.next > 0) ? "correct" : "incorrect"}
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
