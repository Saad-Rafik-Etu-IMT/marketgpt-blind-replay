type Order = {
  target: "long" | "flat";
  action: string;
  status: string;
  price: number | null;
  reason: string | null;
};
export type Comparison = {
  bins: number[];
  version: 2;
  split: "validation";
  origin: "synthetic" | "experimental";
  seed: number;
  trainCount: number;
  trainEnd: string;
  validationStart: string;
  validationEnd: string;
  commit: string;
  dataset: string;
  checkpoint: string;
  profile: string;
  curves: { step: number; trainingLoss: number; validationLoss: number }[];
  strategies: { name: string; gain: number; dd: number }[];
  cases: { p: number; next: number; actions: Order[] }[];
  returns: number[];
  macroF1: number;
  interval: number[];
  calibration: { count: number; predicted: number; observed: number }[];
};
const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);
const prob = (v: unknown) => finite(v) && v >= 0 && v <= 1;
const label = (v: unknown) =>
  typeof v === "string" && v.length > 0 && v.length <= 256;
export function parseComparison(raw: string): Comparison {
  if (raw.length > 2097152) throw Error("Maximum file size is 2 MiB.");
  const v = JSON.parse(raw) as Comparison;
  const fail = () => {
    throw Error(
      "Expected a complete validation comparison from export_comparison.py.",
    );
  };
  if (
    !v ||
    v.version !== 2 ||
    v.split !== "validation" ||
    !["synthetic", "experimental"].includes(v.origin) ||
    !Number.isSafeInteger(v.seed) ||
    !Number.isSafeInteger(v.trainCount) ||
    v.trainCount < 1 ||
    ![
      v.trainEnd,
      v.validationStart,
      v.validationEnd,
      v.commit,
      v.dataset,
      v.checkpoint,
      v.profile,
    ].every(label) ||
    !Array.isArray(v.bins) ||
    v.bins.length > 31 ||
    !v.bins.every((x, i) => finite(x) && (!i || x > v.bins[i - 1])) ||
    !Array.isArray(v.cases) ||
    !v.cases.length ||
    v.cases.length > 4000 ||
    !Array.isArray(v.returns) ||
    v.returns.length !== v.cases.length + 63 ||
    !v.returns.every((x) => finite(x) && Math.abs(x) < 5) ||
    !Array.isArray(v.curves) ||
    !v.curves.length ||
    v.curves.length > 2000 ||
    !v.curves.every(
      (x) =>
        x &&
        Number.isSafeInteger(x.step) &&
        finite(x.trainingLoss) &&
        finite(x.validationLoss),
    ) ||
    !Array.isArray(v.strategies) ||
    v.strategies.length !== 3 ||
    !v.strategies.every(
      (x) => x && label(x.name) && finite(x.gain) && prob(x.dd),
    ) ||
    !prob(v.macroF1) ||
    !Array.isArray(v.interval) ||
    v.interval.length !== 2 ||
    !v.interval.every((x) => finite(x) && Math.abs(x) <= 1) ||
    v.interval[0] > v.interval[1] ||
    !Array.isArray(v.calibration) ||
    !v.calibration.length ||
    v.calibration.length > 5 ||
    !v.calibration.every(
      (x) =>
        x &&
        Number.isSafeInteger(x.count) &&
        x.count > 0 &&
        prob(x.predicted) &&
        prob(x.observed),
    ) ||
    !v.cases.every(
      (c) =>
        c &&
        prob(c.p) &&
        finite(c.next) &&
        Math.abs(c.next) < 5 &&
        Array.isArray(c.actions) &&
        c.actions.length === 3 &&
        c.actions.every(
          (a) =>
            a &&
            ["long", "flat"].includes(a.target) &&
            label(a.action) &&
            label(a.status) &&
            (a.price === null || finite(a.price)) &&
            (a.reason === null || label(a.reason)),
        ),
    )
  )
    fail();
  return v;
}
