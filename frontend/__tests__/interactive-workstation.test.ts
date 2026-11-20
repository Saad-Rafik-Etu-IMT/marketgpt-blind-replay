import { expect, it } from "vitest";
import {
  bounds,
  clampRange,
  decimate,
  nearest,
  panRange,
  zoomRange,
} from "@/lib/research/chart-navigation";
import {
  confusion,
  matchCases,
  type PredictionCase,
} from "@/lib/research/prediction-analysis";

it("anchors zoom and clamps horizontal navigation to the disclosed market", () => {
  expect(zoomRange([0, 100], 25, 0.5, [0, 100])).toEqual([12.5, 62.5]);
  expect(panRange([20, 50], 100, [0, 80])).toEqual([50, 80]);
  expect(clampRange([-100, 1000], [0, 80])).toEqual([0, 80]);
  expect(bounds([3, 3])).toEqual([2.5, 3.5]);
});
it("keeps recorded extrema and endpoints when plotting large series", () => {
  const data = Array.from({ length: 15000 }, (_, x) => ({
    x,
    y: x === 500 ? 10000 : x === 1500 ? -10000 : 0,
  }));
  const reduced = decimate(data);
  expect(reduced.length).toBeLessThanOrEqual(1800);
  expect(reduced).toContain(data[500]);
  expect(reduced).toContain(data[1500]);
  expect(reduced[0]).toBe(data[0]);
  expect(reduced.at(-1)).toBe(data.at(-1));
  expect(nearest(data, 500.2)).toBe(data[500]);
});
const sample = (
  index: number,
  probability: number,
  actual: number,
): PredictionCase => ({
  index,
  probability,
  actual,
  asset: "A",
  date: `2020-01-0${index + 1}`,
  bar: index,
  volatility: "calm",
  trend: "up_60d",
});
it("threshold statistics use real binary labels and reject mismatched comparison rows", () => {
  const a = [
    sample(0, 0.1, 0),
    sample(1, 0.6, 0),
    sample(2, 0.3, 1),
    sample(3, 0.9, 1),
  ];
  expect(confusion(a, 0.5)).toEqual({
    cells: [1, 1, 1, 1],
    accuracy: 0.5,
    precision: 0.5,
    recall: 0.5,
  });
  expect(confusion(a, 1).precision).toBeNull();
  expect(matchCases(a, [sample(0, 0.3, 0), sample(1, 0.7, 1)])).toHaveLength(1);
});
