"""Compute-cost estimate shown before an experiment starts.

Cost of one run ~ 6 x training rows x parameters x epochs floating-point
operations (forward + backward). The rate is calibrated on the T5 run of
2026-11-20 (about 4e9 operations per second for NumPy on one CPU core); the
estimate is an order of magnitude, not a promise. Real runs usually stop
earlier (early stopping).
"""

from __future__ import annotations

from typing import Any

from src.ai.mlp import MLP
from src.experiments.spec import TRANSFORMERS, ExperimentSpec
from src.features import DEFAULT_REGISTRY, get_profile

OPERATIONS_PER_SECOND = 3e9
MAX_ESTIMATED_SECONDS = 4 * 3600
TRAIN_SHARE = (
    0.75  # inner_train rows / all bars before validation (measured on the six ETFs)
)


def estimate(spec: ExperimentSpec, rows_per_source: dict[str, int]) -> dict[str, Any]:
    rows = int(sum(rows_per_source.get(s, 0) for s in spec.sources) * TRAIN_SHARE)
    runs, total_ops = [], 0.0
    folds = spec.experiment_profile.walk_forward_folds
    for run in spec.runs():
        columns = len(
            DEFAULT_REGISTRY.columns(get_profile(run.feature_profile).features)
        )
        params = (
            4000
            if run.preset in TRANSFORMERS
            else MLP(columns, spec.architecture(run.preset)).parameter_count
        )
        epochs = spec.train_config(run.preset, run.seed).epochs
        # Attention also evaluates pairwise token interactions.
        ops = (
            6.0
            * rows
            * params
            * epochs
            * (1 + folds)
            * (16 if run.preset in TRANSFORMERS else 1)
        )
        total_ops += ops
        runs.append(
            {"key": run.key, "inputs": columns, "parameters": params, "epochs": epochs}
        )
    seconds = total_ops / OPERATIONS_PER_SECOND
    return {
        "runs": len(runs),
        "training_rows_estimate": rows,
        "estimated_seconds": round(seconds, 1),
        "within_budget": seconds <= MAX_ESTIMATED_SECONDS,
        "budget_seconds": MAX_ESTIMATED_SECONDS,
        "per_run": runs,
        "note": "order of magnitude; early stopping usually ends runs sooner",
    }
