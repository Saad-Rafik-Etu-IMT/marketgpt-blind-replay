import type { EvaluationEntry } from "@/lib/research/laboratory";
import { numberText } from "@/lib/research/laboratory";
import { humanLabel, scientificFinding } from "@/lib/research/product-context";

export function FindingSummary({
  entry,
  part,
  onOpen,
}: {
  entry: EvaluationEntry;
  part: string;
  onOpen?: () => void;
}) {
  const finding = scientificFinding(entry),
    interval = finding.comparison?.interval_95;
  return (
    <section
      className={`finding-card finding-${finding.tone}`}
      aria-label="Principal scientific finding"
    >
      <p className="eyebrow">{humanLabel(part)} · seed ensemble</p>
      <h3>{finding.headline}</h3>
      <p>
        {humanLabel(entry.preset)} with{" "}
        {humanLabel(entry.feature_profile).toLowerCase()} was compared with{" "}
        {finding.baseline} using {entry.loss} (lower is better).
      </p>
      <dl className="finding-values">
        <div>
          <dt>Model loss</dt>
          <dd>{numberText(entry.ensemble.loss)}</dd>
        </div>
        <div>
          <dt>95% paired interval</dt>
          <dd>
            {interval
              ? `[${numberText(interval[0])}, ${numberText(interval[1])}]`
              : "Unavailable"}
          </dd>
        </div>
        <div>
          <dt>Evidence</dt>
          <dd>
            {entry.rows.toLocaleString()} rows · {entry.seeds.length} seeds
          </dd>
        </div>
      </dl>
      <p className="muted">
        Difference is model minus baseline. An interval crossing zero is not
        proof of equivalence; this is not final test evidence.
      </p>
      {onOpen && (
        <button className="bench-button" onClick={onOpen}>
          Inspect supporting evidence →
        </button>
      )}
    </section>
  );
}
