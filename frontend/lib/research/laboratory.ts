import { API_BASE_URL } from "@/lib/constants";

export const laboratoryUrl = API_BASE_URL + "/api/experiments";

export async function laboratoryApi<T>(path = "", body?: unknown): Promise<T> {
  const response = await fetch(
    laboratoryUrl + path,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    const value = await response.json().catch(() => ({}));
    throw new Error(
      typeof value.detail === "string"
        ? value.detail
        : JSON.stringify(value.detail ?? "Request failed"),
    );
  }
  return response.json() as Promise<T>;
}

export type Search = {
  method: "random" | "grid";
  budget: number;
  seed?: number;
  learning_rates?: number[];
  weight_decays?: number[];
  dropouts?: number[];
  batch_sizes?: number[];
};
export type Spec = {
  name: string;
  profile: string;
  sources: string[];
  feature_profiles: string[];
  presets: string[];
  targets: string[];
  epochs?: number | null;
  search?: Search | null;
  max_minutes?: number | null;
  auto_evaluate?: boolean;
  report_validation?: boolean;
  device?: string;
};
export type Feature = {
  id: string;
  group: string;
  description: string;
  definition: string;
  units: string;
  columns: string[];
  lookback: number;
  blind_safe: boolean;
};
export type FeatureProfile = {
  id: string;
  description: string;
  available: boolean;
  column_count?: number;
  lookback?: number;
  features: string[];
  unavailable_reason: string;
};
export type Catalogue = {
  features: Feature[];
  feature_profiles: FeatureProfile[];
  targets: {
    id: string;
    interpretation: string;
    output_names: string[];
    definition: string;
  }[];
  presets: Record<string, unknown>;
  experiment_profiles: {
    id: string;
    seeds: number;
    max_epochs: number;
    description: string;
  }[];
  campaigns: { id: string; title: string; hypothesis: string; spec: Spec }[];
  max_runs: number;
  transformer_available: boolean;
};
export type Source = {
  source: string;
  status: string;
  kind: string;
  rows?: number;
  first?: string;
  last?: string;
  provenance_match?: boolean | null;
};
export type Status = {
  state: string;
  done: number;
  total: number;
  message: string;
  evaluation_state?: string;
  current?: {
    run: string;
    epoch?: number;
    epochs?: number;
    train_loss?: number;
    select_loss?: number;
  } | null;
};
export type Campaign = Status & { id: string; name: string; profile: string };
export type Detail = {
  id: string;
  spec: Spec;
  fingerprint: string;
  status: Status;
};
export type Epoch = {
  epoch: number;
  train_loss: number;
  select_loss: number;
  selected?: boolean;
};
export type Run = {
  key: string;
  state: string;
  feature_profile?: string;
  preset?: string;
  seed?: number;
  parameter_count?: number;
  best_epoch?: number;
  epochs_run?: number;
  error?: string;
  inner_validation?: Record<string, Record<string, unknown>>;
  history?: Epoch[];
  walk_forward?: {
    fold: number;
    start: string;
    train_rows: number;
    selection_rows?: number;
    evaluated_rows: number;
    best_epoch?: number;
  }[];
};
export type Estimate = {
  runs: number;
  estimated_seconds: number;
  budget_seconds: number;
  within_budget: boolean;
};
export type Comparison = {
  mean_difference: number;
  interval_95: [number | null, number | null];
  verdict: string;
  dates: number;
};
export type EvaluationEntry = {
  feature_profile: string;
  preset: string;
  target: string;
  seeds: number[];
  rows: number;
  loss: string;
  ensemble: { loss: number; metrics: Record<string, unknown> };
  baselines: Record<
    string,
    {
      loss: number;
      metrics: Record<string, unknown>;
      comparison: Comparison;
      seeds_better: number;
    }
  >;
  strongest_baseline: string;
  per_seed_loss: Record<string, number>;
  calibration?: {
    ece: number;
    reliability_up: { confidence: number; frequency: number; count: number }[];
  };
  curves?: { roc: [number, number][]; precision_recall: [number, number][] };
  cost_sensitivity?: {
    cost_bps_per_change: number;
    model_ensemble: { total_log_return_equal_weight: number };
    logistic_regression: { total_log_return_equal_weight: number };
    always_long: { total_log_return_equal_weight: number };
  }[];
  importance?: { feature: string; columns: number; loss_increase: number }[];
  regimes: Record<
    string,
    Record<
      string,
      { rows: number; model_loss: number; strongest_baseline_loss: number }
    >
  >;
};
export type Evaluation = {
  part: string;
  results: EvaluationEntry[];
  notes: string[];
  ablation: {
    summary: string;
    findings: {
      feature_profile: string;
      preset: string;
      target: string;
      category: string;
      comparison: Comparison;
    }[];
  };
};
export type Event = {
  decision: {
    decision_bar_index: number;
    target: string;
    action: string;
    probabilities: { long: number; flat: number } | null;
  };
  execution: {
    status: string;
    fill_bar: number | null;
    fill_price: number | null;
  } | null;
  portfolio: { equity: number; bar_index: number; cash: number };
};
export type Replay = {
  run: string;
  asset: string;
  part: string;
  accounting: string;
  policy: string;
  label: string;
  market: { date: string; close: number }[];
  first_decision: number;
  players: {
    name: string;
    initial: { equity: number; bar_index: number };
    events: Event[];
    gain: number;
    max_drawdown: number;
    decisions: number;
    state: string;
  }[];
};
export const numberText = (value: unknown, digits = 4) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", { maximumFractionDigits: digits })
    : "—";
export const labelText = (value: string) => value.replaceAll("_", " ");
