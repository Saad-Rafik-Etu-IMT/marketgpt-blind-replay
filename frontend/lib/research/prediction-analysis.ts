export type PredictionCase = {
  index: number;
  asset: string;
  date: string;
  bar: number;
  probability: number;
  actual: number;
  volatility: string;
  trend: string;
};
export function confusion(cases: PredictionCase[], threshold: number) {
  const cells = [0, 0, 0, 0];
  for (const row of cases)
    cells[row.actual * 2 + Number(row.probability >= threshold)]++;
  const [tn, fp, fn, tp] = cells;
  return {
    cells,
    accuracy: cases.length ? (tp + tn) / cases.length : null,
    precision: tp + fp ? tp / (tp + fp) : null,
    recall: tp + fn ? tp / (tp + fn) : null,
  };
}
export function matchCases(a: PredictionCase[], b: PredictionCase[]) {
  const other = new Map(b.map((r) => [`${r.asset}/${r.date}`, r]));
  return a.flatMap((r) => {
    const o = other.get(`${r.asset}/${r.date}`);
    return o && o.actual === r.actual ? [{ first: r, second: o }] : [];
  });
}
