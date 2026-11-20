"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, pct } from "@/lib/research/study-api";
import { LinePlot } from "./ScientificPlot";

type Mode = "facts" | "llm";
type Evaluation = {
  dataset: string;
  checkpoint: string;
  summary: {
    participants: number;
    incomplete: number;
    pairs: { participant: string; facts: number; llm: number }[];
    counts: Record<Mode, number>;
    correct: Record<Mode, number>;
    fallback: number;
    preference: { facts: number; llm: number; equal: number };
  };
};

export function HumanEvaluation({ base }: { base: string }) {
  const path = base + "/human-evaluation";
  const q = useQuery({
    queryKey: ["human-evaluation", base],
    queryFn: () => api<Evaluation | null>(path),
  });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function load(file: File) {
    setBusy(true);
    setError("");
    try {
      if (file.size > 2097152) throw Error("Maximum file size is 2 MiB");
      await api(path, JSON.parse(await file.text()));
      await q.refetch();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const summary = q.data?.summary;
  const average = summary?.pairs.length
    ? summary.pairs.reduce((s, p) => s + p.llm - p.facts, 0) /
      summary.pairs.length
    : null;
  const example = {
    responses: [
      {
        participant: "P01",
        case: 1,
        format: "facts",
        source: "Fact sheet",
        fallback: false,
        probability_percent: 72.1,
        block: 8,
        preference: "equal",
      },
    ],
  };
  return (
    <section className="bench-panel">
      <h2>Explanation comprehension study</h2>
      <p>
        Give participants the same measured facts in different formats on
        different, counterbalanced cases. Ask for the model’s estimated rising
        probability (percent, one decimal) and the number (1–8, oldest to
        newest) of the block with the largest absolute measured sensitivity.
        Each answer is worth one point. Capture preference after both formats.
      </p>
      <details>
        <summary>Response file format and collection instructions</summary>
        <p>
          Use anonymous codes, one row per participant and case, case numbers as
          shown in Inspector. Mark the actual explanation source and whether LLM
          generation fell back to the template. Keep the assignment balanced
          across cases and formats. The illustrative row below is a schema
          example only; collect real answers and replace its values.
        </p>
        <pre className="engine-json">{JSON.stringify(example, null, 2)}</pre>
      </details>
      <label className="import-control">
        Import and replace responses (JSON)
        <input
          type="file"
          accept=".json,application/json"
          disabled={busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void load(file);
          }}
        />
      </label>
      {(error || q.error) && <p role="alert">{error || String(q.error)}</p>}
      {!summary ? (
        <p className="muted">
          {q.isPending
            ? "Loading response evidence…"
            : "No participant responses collected for this run."}
        </p>
      ) : (
        <>
          <p>
            {summary.participants} participants ·{" "}
            {summary.counts.facts + summary.counts.llm} case responses ·{" "}
            {summary.pairs.length} complete participant pairs ·{" "}
            {summary.incomplete} incomplete · {summary.fallback} LLM fallbacks
          </p>
          {summary.pairs.length > 0 ? (
            <>
              <LinePlot
                title="Paired explanation comprehension scores"
                legend={false}
                series={summary.pairs.map((p) => ({
                  name: p.participant,
                  points: [
                    { x: 0, y: p.facts },
                    { x: 1, y: p.llm },
                  ],
                  dots: true,
                  color: "#148b82",
                }))}
                xDomain={[0, 1]}
                yDomain={[0, 1]}
                formatX={(v) => (v === 0 ? "Fact table" : "LLM assigned")}
                xLabel="Same participant · different cases (includes fallback)"
                formatY={pct}
              />
              <p>
                Mean paired LLM-assigned minus fact table:{" "}
                {average === null ? "—" : (100 * average).toFixed(1) + " pp"}.
              </p>
            </>
          ) : (
            <p>No complete paired participants yet.</p>
          )}
          <table>
            <thead>
              <tr>
                <th>Format</th>
                <th>Responses</th>
                <th>Correct answers</th>
                <th>Preferred</th>
              </tr>
            </thead>
            <tbody>
              {(["facts", "llm"] as const).map((m) => (
                <tr key={m}>
                  <td>{m}</td>
                  <td>{summary.counts[m]}</td>
                  <td>
                    {summary.correct[m]} / {2 * summary.counts[m]}
                  </td>
                  <td>{summary.preference[m]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>
            Equal preference: {summary.preference.equal}. Unbalanced case
            assignments and fallback usage may affect the comparison.
            Assigned-format scores include fallback responses.
          </p>
          <details>
            <summary>Exact participant scores</summary>
            <table>
              <thead>
                <tr>
                  <th>Participant</th>
                  <th>Facts</th>
                  <th>LLM assigned</th>
                </tr>
              </thead>
              <tbody>
                {summary.pairs.map((p) => (
                  <tr key={p.participant}>
                    <td>{p.participant}</td>
                    <td>{pct(p.facts)}</td>
                    <td>{pct(p.llm)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      )}
    </section>
  );
}
