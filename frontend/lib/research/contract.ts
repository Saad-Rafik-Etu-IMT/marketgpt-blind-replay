// Proposed browser integration contract, not a format emitted by the backend.
export const MAX_ARTIFACT_BYTES = 2 * 1024 * 1024;
export const PROBABILITY_TOLERANCE = 1e-6;
export type Direction = "long" | "flat" | "short";
export interface CurvePoint {
  step: number;
  trainingLoss: number | null;
  validationLoss: number | null;
}
export interface ResearchRun {
  id: string;
  model: string;
  dataset: string;
  split: string;
  checkpoint: string;
  seed: number;
  lossName: string;
  curves: CurvePoint[];
}
export interface ResearchSample {
  id: string;
  runId: string;
  decisionBar: number;
  predicted: Direction;
  probabilities: Record<Direction, number>;
  inputs: { bar: number; feature: string; value: number; unit: string }[];
  explanation: {
    text: string;
    template: string;
    modelVersion: string;
    promptVersion: string;
    facts: { id: string; bar: number; text: string }[];
    references: string[];
    checks: { name: string; passed: boolean; detail: string }[];
  } | null;
}
export interface ResearchReport {
  id: string;
  runId: string;
  split: string;
  cutoffBar: number;
  sampleCount: number;
  macroF1: number | null;
  failures: { sampleId: string; observed: Direction; outcomeBar: number; detail: string }[];
}
export interface ResearchBundle {
  schemaVersion: 1;
  origin: "synthetic" | "experimental";
  runs: ResearchRun[];
  samples: ResearchSample[];
  reports: ResearchReport[];
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${path}: expected an object.`);
  return value as Record<string, unknown>;
}
function text(value: unknown, path: string): string {
  if (typeof value !== "string" || !value.trim() || value.length > 4000) throw new Error(`${path}: expected 1–4000 characters.`);
  return value;
}
function id(value: unknown, path: string): string {
  const result = text(value, path);
  if (!/^[a-zA-Z0-9._-]{1,100}$/.test(result)) throw new Error(`${path}: use 1–100 letters, numbers, dots, underscores or hyphens.`);
  return result;
}
function number(value: unknown, path: string, min = -Infinity, max = Infinity): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) throw new Error(`${path}: expected a finite number in [${min}, ${max}].`);
  return value;
}
function integer(value: unknown, path: string, min = 0): number {
  const result = number(value, path, min, Number.MAX_SAFE_INTEGER);
  if (!Number.isSafeInteger(result)) throw new Error(`${path}: expected an integer.`);
  return result;
}
function metric(value: unknown, path: string, max = Infinity): number | null {
  return value == null ? null : number(value, path, 0, max);
}
function list<T>(value: unknown, path: string, max: number, parse: (v: unknown, p: string) => T): T[] {
  if (!Array.isArray(value) || value.length > max) throw new Error(`${path}: expected an array with at most ${max} entries.`);
  return value.map((v, i) => parse(v, `${path}[${i}]`));
}
function direction(value: unknown, path: string): Direction {
  if (value !== "long" && value !== "flat" && value !== "short") throw new Error(`${path}: expected long, flat or short.`);
  return value;
}
function unique(values: string[], path: string) {
  if (new Set(values).size !== values.length) throw new Error(`${path}: duplicate identifiers.`);
}
function parseRun(value: unknown, p: string): ResearchRun {
  const v = object(value, p);
  const curves = list(v.curves, `${p}.curves`, 2000, (item, q) => {
    const c = object(item, q);
    return { step: integer(c.step, `${q}.step`), trainingLoss: metric(c.trainingLoss, `${q}.trainingLoss`), validationLoss: metric(c.validationLoss, `${q}.validationLoss`) };
  });
  if (curves.some((c, i) => i > 0 && c.step <= curves[i - 1].step)) throw new Error(`${p}.curves: steps must be strictly increasing.`);
  return {
    id: id(v.id, `${p}.id`), model: text(v.model, `${p}.model`),
    dataset: text(v.dataset, `${p}.dataset`), split: text(v.split, `${p}.split`),
    checkpoint: text(v.checkpoint, `${p}.checkpoint`), seed: integer(v.seed, `${p}.seed`),
    lossName: text(v.lossName, `${p}.lossName`), curves,
  };
}
function parseSample(value: unknown, p: string): ResearchSample {
  const v = object(value, p);
  const probs = object(v.probabilities, `${p}.probabilities`);
  const probabilities = {
    long: number(probs.long, `${p}.probabilities.long`, 0, 1),
    flat: number(probs.flat, `${p}.probabilities.flat`, 0, 1),
    short: number(probs.short, `${p}.probabilities.short`, 0, 1),
  };
  if (Math.abs(probabilities.long + probabilities.flat + probabilities.short - 1) > PROBABILITY_TOLERANCE) throw new Error(`${p}.probabilities: sum must equal 1 within ${PROBABILITY_TOLERANCE}.`);
  const decisionBar = integer(v.decisionBar, `${p}.decisionBar`);
  const inputs = list(v.inputs, `${p}.inputs`, 512, (item, q) => {
    const f = object(item, q);
    return { bar: integer(f.bar, `${q}.bar`), feature: text(f.feature, `${q}.feature`), value: number(f.value, `${q}.value`), unit: text(f.unit, `${q}.unit`) };
  });
  if (inputs.some(f => f.bar > decisionBar)) throw new Error(`${p}.inputs: future bars are forbidden.`);
  let explanation: ResearchSample["explanation"] = null;
  if (v.explanation != null) {
    const e = object(v.explanation, `${p}.explanation`);
    const facts = list(e.facts, `${p}.explanation.facts`, 64, (item, q) => {
      const f = object(item, q);
      return { id: id(f.id, `${q}.id`), bar: integer(f.bar, `${q}.bar`), text: text(f.text, `${q}.text`) };
    });
    unique(facts.map(f => f.id), `${p}.explanation.facts`);
    if (facts.some(f => f.bar > decisionBar || !inputs.some(input => input.bar === f.bar))) throw new Error(`${p}.explanation.facts: each bar must reference a visible input.`);
    const references = list(e.references, `${p}.explanation.references`, 64, id);
    if (references.some(ref => !facts.some(f => f.id === ref))) throw new Error(`${p}.explanation.references: unknown fact ID.`);
    explanation = {
      text: text(e.text, `${p}.explanation.text`), template: text(e.template, `${p}.explanation.template`),
      modelVersion: text(e.modelVersion, `${p}.explanation.modelVersion`), promptVersion: text(e.promptVersion, `${p}.explanation.promptVersion`),
      facts, references,
      checks: list(e.checks, `${p}.explanation.checks`, 64, (item, q) => {
        const c = object(item, q);
        if (typeof c.passed !== "boolean") throw new Error(`${q}.passed: expected a boolean.`);
        return { name: text(c.name, `${q}.name`), passed: c.passed, detail: text(c.detail, `${q}.detail`) };
      }),
    };
  }
  return { id: id(v.id, `${p}.id`), runId: id(v.runId, `${p}.runId`), decisionBar, predicted: direction(v.predicted, `${p}.predicted`), probabilities, inputs, explanation };
}
function parseReport(value: unknown, p: string): ResearchReport {
  const v = object(value, p);
  return {
    id: id(v.id, `${p}.id`), runId: id(v.runId, `${p}.runId`), split: text(v.split, `${p}.split`),
    cutoffBar: integer(v.cutoffBar, `${p}.cutoffBar`), sampleCount: integer(v.sampleCount, `${p}.sampleCount`, 1),
    macroF1: metric(v.macroF1, `${p}.macroF1`, 1),
    failures: list(v.failures, `${p}.failures`, 500, (item, q) => {
      const f = object(item, q);
      return { sampleId: id(f.sampleId, `${q}.sampleId`), observed: direction(f.observed, `${q}.observed`), outcomeBar: integer(f.outcomeBar, `${q}.outcomeBar`), detail: text(f.detail, `${q}.detail`) };
    }),
  };
}
export function parseResearchBundle(value: unknown): ResearchBundle {
  const v = object(value, "artifact");
  if (v.schemaVersion !== 1) throw new Error("Unsupported schemaVersion. Export a research bundle with schemaVersion: 1; see docs/research-interface.md.");
  if (v.origin !== "synthetic" && v.origin !== "experimental") throw new Error("origin: declare synthetic or experimental.");
  const runs = list(v.runs, "runs", 20, parseRun);
  const samples = list(v.samples, "samples", 500, parseSample);
  const reports = list(v.reports, "reports", 20, parseReport);
  unique(runs.map(r => r.id), "runs");
  unique(samples.map(s => s.id), "samples");
  unique(reports.map(r => r.id), "reports");
  for (const sample of samples) {
    if (!runs.some(r => r.id === sample.runId)) throw new Error(`${sample.id}: unknown runId.`);
  }
  for (const report of reports) {
    if (!runs.some(r => r.id === report.runId)) throw new Error(`${report.id}: unknown runId.`);
    unique(report.failures.map(f => f.sampleId), `${report.id}.failures`);
    if (report.failures.length > report.sampleCount) throw new Error(`${report.id}: failures exceed sampleCount.`);
    for (const failure of report.failures) {
      const sample = samples.find(s => s.id === failure.sampleId && s.runId === report.runId);
      if (!sample) throw new Error(`${report.id}: failure references an unknown sample or a different run.`);
      if (failure.outcomeBar <= sample.decisionBar || failure.outcomeBar > report.cutoffBar) throw new Error(`${report.id}: outcome must follow the decision and not exceed evaluation cutoff.`);
      if (failure.observed === sample.predicted) throw new Error(`${report.id}: misclassification must differ from the predicted class.`);
    }
  }
  return { schemaVersion: 1, origin: v.origin, runs, samples, reports };
}
export async function importResearchFile(file: File): Promise<ResearchBundle> {
  if (file.size > MAX_ARTIFACT_BYTES) throw new Error("Artifact exceeds 2 MiB. Export a smaller research bundle.");
  const content = await file.text();
  let value: unknown;
  try { value = JSON.parse(content); } catch { throw new Error("Invalid JSON. Select a version-1 research bundle."); }
  return parseResearchBundle(value);
}
