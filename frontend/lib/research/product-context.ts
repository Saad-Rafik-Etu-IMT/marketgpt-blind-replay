import type { EvaluationEntry, Replay, Run } from "./laboratory";

export const laboratories = [
  "Overview",
  "Data",
  "Training",
  "Evaluation",
  "Replay",
  "Explainability",
] as const;
export type Laboratory = (typeof laboratories)[number];
export const presentationStages = [
  "Research question",
  "Data",
  "Models",
  "Training",
  "Results",
  "Replay",
  "Explainability",
  "Conclusion",
] as const;
export const presentationTabs: Laboratory[] = [
  "Overview",
  "Data",
  "Training",
  "Training",
  "Evaluation",
  "Replay",
  "Explainability",
  "Overview",
];

type Params = { toString(): string };
export function labHref(
  params: Params,
  changes: Record<string, string | number | null | undefined>,
) {
  const next = new URLSearchParams(params.toString());
  // Different objects cannot inherit a result selection or a playback clock.
  if (
    changes.campaign !== undefined &&
    String(changes.campaign ?? "") !== (next.get("campaign") ?? "")
  ) {
    for (const key of [
      "run",
      "entry",
      "asset",
      "source",
      "cursor",
      "opponent",
      "feature",
    ])
      next.delete(key);
  }
  if (
    changes.run !== undefined &&
    String(changes.run ?? "") !== (next.get("run") ?? "")
  ) {
    for (const key of ["entry", "cursor", "opponent"]) next.delete(key);
  }
  if (changes.part !== undefined && changes.part !== next.get("part")) {
    for (const key of ["entry", "cursor", "opponent"]) next.delete(key);
  }
  if (changes.asset !== undefined && changes.asset !== next.get("asset"))
    next.delete("cursor");
  for (const [key, value] of Object.entries(changes)) {
    if (value === undefined) continue;
    if (value === null || value === "") next.delete(key);
    else next.set(key, String(value));
  }
  const query = next.toString();
  return "/laboratory" + (query ? "?" + query : "");
}

const labels: Record<string, string> = {
  returns_only_v1: "Returns only",
  returns_plus_trend: "Returns + trend",
  returns_plus_volatility: "Returns + volatility",
  returns_plus_volume: "Returns + volume",
  returns_plus_price_structure: "Returns + price structure",
  returns_plus_statistics: "Returns + statistics",
  core_features: "Core features",
  extended_features: "Extended features",
  w1_baseline: "Shallow MLP",
  small: "Small MLP",
  medium: "Medium MLP",
  large: "Large MLP",
  deep_v1: "Deep MLP",
  transformer_reference: "Reference Transformer",
  transformer_no_position: "Transformer · no position encoding",
  transformer_no_dropout: "Transformer · no dropout",
  transformer_no_decay: "Transformer · no weight decay",
  transformer_fixed_lr: "Transformer · fixed learning rate",
  direction_binary_v1: "Direction · executed return",
  direction_binary_close_v1: "Direction · closing price",
  direction_above_1bp_v1: "Return above 1 bp",
  synthetic: "Synthetic noise control",
  synthetic_signal: "Synthetic learnable signal",
  inner_validation: "Development validation",
  validation: "Historical validation",
  complete_with_failures: "Completed with failed runs",
  complete: "Completed",
  running: "Training",
  queued: "Waiting",
  interrupted: "Interrupted",
  class_frequency: "Constant class frequency",
  logistic_regression: "Logistic regression",
  decision_stump: "Decision stump",
  ma_crossover_20_50: "Moving-average crossover",
  buy_and_hold: "Buy & hold",
  none_already_long: "Existing invested position maintained",
  none_already_flat: "Remains in cash",
  none_pending_order: "Existing order pending",
  none_zero_quantity: "No affordable order",
  pre_trade_rejected: "Risk check refused the order",
  submit_buy: "Buy order submitted",
  submit_sell: "Sell order submitted",
  long: "Invested",
  flat: "Cash",
};
export function humanLabel(value: string) {
  if (labels[value]) return labels[value];
  const trial = value.match(/^(.+)_h(\d+)$/);
  if (trial && labels[trial[1]])
    return labels[trial[1]] + " · Trial " + trial[2];
  return value.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
}
export function runLabel(run: Run | string) {
  const key = typeof run === "string" ? run : run.key;
  const [profile, preset, seed] = key.split("__");
  if (!preset || !seed) return humanLabel(key);
  return [
    humanLabel(profile),
    humanLabel(preset),
    "Seed " + seed.replace(/^s/, ""),
  ].join(" · ");
}
export function entryKey(entry: EvaluationEntry) {
  return [entry.feature_profile, entry.preset, entry.target].join("__");
}
export function chooseEntry(
  entries: EvaluationEntry[],
  key: string | null,
  run?: Run,
) {
  if (!run) return key ? entries.find((e) => entryKey(e) === key) : entries[0];
  if (run.state !== "complete") return undefined;
  const profile = run.feature_profile ?? run.key.split("__")[0];
  const preset = run.preset ?? run.key.split("__")[1];
  const matched = entries.filter(
    (e) =>
      e.feature_profile === profile &&
      e.preset === preset &&
      e.seeds.includes(run.seed ?? Number(run.key.split("__")[2]?.slice(1))),
  );
  return key ? matched.find((e) => entryKey(e) === key) : matched[0];
}
export function scientificFinding(entry: EvaluationEntry) {
  const baseline = entry.baselines[entry.strongest_baseline];
  const comparison = baseline?.comparison;
  const [lo, hi] = comparison?.interval_95 ?? [null, null];
  const valid =
    typeof lo === "number" &&
    Number.isFinite(lo) &&
    typeof hi === "number" &&
    Number.isFinite(hi);
  const headline = !valid
    ? "Insufficient statistical evidence"
    : hi < 0
      ? "Model has lower measured loss"
      : lo > 0
        ? "Baseline has lower measured loss"
        : "No measurable difference";
  return {
    headline,
    comparison,
    baseline: humanLabel(entry.strongest_baseline),
    tone: !valid
      ? "neutral"
      : hi < 0
        ? "positive"
        : lo > 0
          ? "negative"
          : "neutral",
  };
}

export function playbackFrame(replay: Replay, requested: number) {
  const events = replay.players[0]?.events ?? [];
  const cursor = Math.min(
    events.length,
    Math.max(0, Math.floor(Number.isFinite(requested) ? requested : 0)),
  );
  const index = Math.min(cursor, Math.max(0, events.length - 1));
  const event = events[index];
  const end =
    cursor < events.length
      ? event.decision.decision_bar_index
      : (events.at(-1)?.portfolio.bar_index ?? replay.first_decision);
  return {
    cursor,
    index,
    event,
    end,
    length: events.length,
    market: replay.market.slice(0, end + 1),
  };
}

export function replayCompatibility(a: Replay, b: Replay): string | null {
  if (!a.profile_hash || a.profile_hash !== b.profile_hash)
    return "Different or unrecorded execution profiles.";
  if (
    a.asset !== b.asset ||
    a.part !== b.part ||
    a.accounting !== b.accounting ||
    a.policy !== b.policy ||
    a.label !== b.label
  )
    return "Different asset, partition, label, or accounting/policy assumptions.";
  if (
    a.first_decision !== b.first_decision ||
    a.market.length !== b.market.length
  )
    return "Different market clock or warm-up.";
  if (
    a.market.some((m, i) =>
      ["date", "open", "high", "low", "close", "volume"].some(
        (k) => m[k as keyof typeof m] !== b.market[i][k as keyof typeof m],
      ),
    )
  )
    return "Market observations differ.";
  const ae = a.players[0]?.events ?? [],
    be = b.players[0]?.events ?? [];
  if (
    !ae.length ||
    !be.length ||
    ae
      .slice(0, Math.min(ae.length, be.length))
      .some(
        (e, i) =>
          e.decision.decision_bar_index !== be[i].decision.decision_bar_index,
      )
  )
    return "Different decision clocks.";
  return null;
}
