/** The recorded replay's cursor counts revealed outcomes, not predictions. */
export function replayClock(length: number, cursor: number, previous: boolean) {
  const decisionCount = Math.min(cursor + 1, length);
  const decisionIndex = Math.min(
    previous && cursor > 0 ? cursor - 1 : cursor,
    length - 1,
  );
  return {
    decisionIndex,
    revealed: decisionIndex < cursor,
    decisionCount,
    outcomeCount: cursor,
    marketCount: cursor + 1,
  };
}

export function isTerminalStudy(state: string | undefined) {
  return state === "complete" || state === "failed" || state === "interrupted";
}

export function competitionRunSignature(
  runs: { key: string; model: string; variant: string; seed: number }[],
  seed: number | undefined,
  variant: string,
) {
  return runs
    .filter((run) => run.seed === seed &&
      (run.model === "MLP" || run.variant === variant))
    .map((run) => run.key)
    .sort()
    .join(",");
}
