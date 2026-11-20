import { describe, expect, it } from "vitest";
import { competitionRunSignature, isTerminalStudy, replayClock } from "@/lib/research/replay-clock";

describe("recorded replay visibility", () => {
  it("exposes a prediction but no outcome at the starting session", () => {
    expect(replayClock(4, 0, false)).toEqual({
      decisionIndex: 0, revealed: false, decisionCount: 1,
      outcomeCount: 0, marketCount: 1,
    });
  });

  it("distinguishes the previous revealed case from the next blind case", () => {
    expect(replayClock(4, 2, true)).toEqual({
      decisionIndex: 1, revealed: true, decisionCount: 3,
      outcomeCount: 2, marketCount: 3,
    });
    expect(replayClock(4, 2, false)).toMatchObject({
      decisionIndex: 2, revealed: false, outcomeCount: 2,
    });
  });

  it("does not access a nonexistent next prediction at completion", () => {
    expect(replayClock(4, 4, false)).toEqual({
      decisionIndex: 3, revealed: true, decisionCount: 4,
      outcomeCount: 4, marketCount: 5,
    });
  });

  it.each(["complete", "failed", "interrupted"])("stops polling %s studies", state => {
    expect(isTerminalStudy(state)).toBe(true);
  });
  it.each(["queued", "running", undefined])("keeps polling %s studies", state => {
    expect(isTerminalStudy(state)).toBe(false);
  });

  it("refreshes a partial competition when its selected run finishes", () => {
    const mlp = { key: "MLP-reference-0", model: "MLP", variant: "reference", seed: 0 };
    const transformer = { key: "Transformer-reference-0", model: "Transformer", variant: "reference", seed: 0 };
    const other = { ...transformer, key: "Transformer-reference-1", seed: 1 };
    expect(competitionRunSignature([mlp, other], 0, "reference")).toBe(mlp.key);
    expect(competitionRunSignature([transformer, mlp, other], 0, "reference"))
      .toBe(mlp.key + "," + transformer.key);
  });
});
