import { expect, it } from "vitest";
import {
  labHref,
  chooseEntry,
  entryKey,
  scientificFinding,
  playbackFrame,
  replayCompatibility,
} from "@/lib/research/product-context";
import type { EvaluationEntry, Replay, Run } from "@/lib/research/laboratory";

const entry = {
  feature_profile: "returns_only_v1",
  preset: "small",
  target: "direction_binary_v1",
  seeds: [0],
  strongest_baseline: "logistic_regression",
  baselines: {
    logistic_regression: { comparison: { interval_95: [-0.2, -0.1] } },
  },
} as unknown as EvaluationEntry;
const run: Run = {
  key: "returns_only_v1__small__s0",
  state: "complete",
  feature_profile: "returns_only_v1",
  preset: "small",
  seed: 0,
};
it("preserves context across screens and clears incompatible downstream evidence", () => {
  const q = new URLSearchParams(
    "campaign=a&run=r&part=validation&entry=e&asset=SYN_A&cursor=3&opponent=s",
  );
  expect(labHref(q, { tab: "Replay" })).toContain("cursor=3");
  const changed = new URL(
    labHref(q, { campaign: "b", run: "new" }),
    "http://local",
  ).searchParams;
  expect(changed.get("run")).toBe("new");
  expect(changed.has("entry")).toBe(false);
  expect(changed.has("asset")).toBe(false);
  expect(
    new URL(
      labHref(q, { part: "inner_validation" }),
      "http://local",
    ).searchParams.has("cursor"),
  ).toBe(false);
});
it("does not substitute findings from an unrelated run or explicit configuration", () => {
  expect(chooseEntry([entry], entryKey(entry), run)).toBe(entry);
  expect(chooseEntry([entry], null, { ...run, seed: 1 })).toBeUndefined();
  expect(chooseEntry([entry], "missing", run)).toBeUndefined();
});
it("describes uncertainty without treating inconclusive evidence as equivalence", () => {
  expect(scientificFinding(entry).tone).toBe("positive");
  const uncertain = {
    ...entry,
    baselines: {
      logistic_regression: { comparison: { interval_95: [-0.1, 0.1] } },
    },
  } as unknown as EvaluationEntry;
  expect(scientificFinding(uncertain).headline).toBe(
    "No measurable difference",
  );
});
const replay = {
  asset: "SYN_A",
  part: "inner_validation",
  profile_hash: "abc",
  market: [
    { date: "a", close: 10 },
    { date: "b", close: 11 },
    { date: "c", close: 12 },
  ],
  first_decision: 1,
  players: [
    {
      events: [
        { decision: { decision_bar_index: 1 }, portfolio: { bar_index: 2 } },
      ],
    },
  ],
} as unknown as Replay;
it("never reveals the next bar before advancing and rejects different comparison clocks", () => {
  expect(playbackFrame(replay, 0).market.map((b) => b.close)).toEqual([10, 11]);
  expect(playbackFrame(replay, 1).market.map((b) => b.close)).toEqual([
    10, 11, 12,
  ]);
  expect(playbackFrame(replay, 999).cursor).toBe(1);
  expect(replayCompatibility(replay, replay)).toBeNull();
  expect(
    replayCompatibility(replay, { ...replay, asset: "SYN_B" }),
  ).not.toBeNull();
  expect(
    replayCompatibility(replay, {
      ...replay,
      market: [...replay.market.slice(0, 2), { date: "c", close: 50 }],
    }),
  ).not.toBeNull();
});
