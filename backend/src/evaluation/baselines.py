"""Non-neural baselines, fitted on inner_train and tuned on inner_validation only.

Every baseline returns raw outputs ``z`` in the target's own parameterization,
so the T4 ``evaluate`` and the per-row losses apply unchanged.

| target kind    | baselines |
| classification | class frequencies of the training rows; logistic regression |
| regression     | training mean; zero return; ridge regression |
| volatility     | training mean; naive realized_vol_20 at t; ridge on log-volatility |
| quantiles      | training empirical quantiles; normal quantiles scaled by realized_vol_20 at t; linear quantile regression |

Logistic and linear quantile regression are the T5 network without hidden
layers (a linear model) trained with AdamW; the decoupled decay is chosen on
inner_validation from a fixed grid. Ridge uses the closed form with alpha from
a fixed grid. Gradient-boosted trees are reported as unavailable: no tree
library is a project dependency, and none is added for this.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.ai.mlp import (
    MLP,
    ArchitectureConfig,
    Head,
    TrainConfig,
    TrainingData,
    evaluate_loss,
    train,
)
from src.targets import Target, TargetKind
from src.targets.targets import QUANTILES

DECAY_GRID: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1)
RIDGE_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
NORMAL_Z = {0.1: -1.2815515655446004, 0.5: 0.0, 0.9: 1.2815515655446004}
UNAVAILABLE = {
    "gradient_boosted_trees": "no tree library in the project dependencies (not added)"
}


def _inverse_softplus(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, float), 1e-6)
    return np.where(v > 30, v, np.log(np.expm1(v)))


def _quantile_outputs(
    q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, state: dict[str, Any]
) -> np.ndarray:
    mean, std = state["mean"][0], state["std"][0]
    return np.column_stack(
        [
            (q50 - mean) / std,
            _inverse_softplus((q50 - q10) / std),
            _inverse_softplus((q90 - q50) / std),
        ]
    )


def _linear(
    target: Target, train_part, select_part, state
) -> tuple[MLP, dict[str, Any]]:
    """Linear model with the target's head; decay chosen on the selection rows."""
    data = TrainingData(
        train_part.x,
        {target.id: train_part.y},
        select_part.x,
        {target.id: select_part.y},
        {target.id: state},
    )
    best, best_loss, chosen = None, np.inf, None
    for decay in DECAY_GRID:
        model = MLP(
            train_part.x.shape[1],
            ArchitectureConfig(hidden=(), heads=(Head(target=target.id),)),
            seed=0,
        )
        config = TrainConfig(
            optimizer="adamw",
            learning_rate=1e-2,
            weight_decay=decay,
            epochs=30,
            patience=5,
            batch_size=256,
            seed=0,
        )
        train(model, data, config)
        loss = evaluate_loss(
            model, select_part.x, {target.id: select_part.y}, {target.id: state}
        )["total"]
        if loss < best_loss:
            best, best_loss, chosen = model, loss, decay
    assert best is not None
    return best, {"weight_decay": chosen, "selection_loss": float(best_loss)}


def _ridge(train_part, select_part) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Closed-form ridge on standardized labels; alpha by selection mean squared error."""
    x, y = train_part.x, np.asarray(train_part.y, float).reshape(len(train_part.x), -1)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    gram, cross = xc.T @ xc, xc.T @ yc
    best = (None, None, np.inf, None)
    ys = np.asarray(select_part.y, float).reshape(len(select_part.x), -1)
    for alpha in RIDGE_GRID:
        w = np.linalg.solve(gram + alpha * np.eye(len(gram)), cross)
        b = y_mean - x_mean @ w
        mse = float(np.mean((select_part.x @ w + b - ys) ** 2))
        if mse < best[2]:
            best = (w, b, mse, alpha)
    return best[0], best[1], {"alpha": best[3], "selection_mse": best[2]}


def decision_stump(train_part, eval_part, classes: int):
    """One deterministic split with Laplace-smoothed leaf frequencies.

    Thresholds are training quantiles; no evaluation data chooses a split.
    """
    x, y = train_part.x, np.asarray(train_part.y, int)
    counts = np.bincount(y, minlength=classes)
    prior = (counts + 1) / (len(y) + classes)
    best = (-float(counts @ np.log(prior)), None, None, prior, prior)
    for col in range(x.shape[1]):
        for threshold in np.unique(np.quantile(x[:, col], np.arange(0.1, 1, 0.1))):
            left = x[:, col] <= threshold
            if min(int(left.sum()), int((~left).sum())) < 10:
                continue
            a = np.bincount(y[left], minlength=classes)
            b = counts - a
            pa, pb = (a + 1) / (a.sum() + classes), (b + 1) / (b.sum() + classes)
            loss = -float(a @ np.log(pa) + b @ np.log(pb))
            if loss < best[0]:
                best = (loss, col, float(threshold), pa, pb)
    _, column, threshold, pa, pb = best
    if column is None:
        probabilities = np.tile(prior, (len(eval_part), 1))
    else:
        probabilities = np.where((eval_part.x[:, column] <= threshold)[:, None], pa, pb)
    return np.log(probabilities), {"column": column, "threshold": threshold, "depth": 1}


def baseline_outputs(
    target: Target, state: dict[str, Any], train_part, select_part, eval_part
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Raw outputs of every baseline on ``eval_part`` and their fitted settings."""
    n = len(eval_part)
    outputs: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {"unavailable": dict(UNAVAILABLE)}
    if target.kind is TargetKind.CLASSIFICATION:
        shares = np.asarray(state["train_class_shares"], float)
        outputs["class_frequency"] = np.tile(
            np.log(np.clip(shares, 1e-9, None)), (n, 1)
        )
        model, info["logistic_regression"] = _linear(
            target, train_part, select_part, state
        )
        outputs["logistic_regression"] = model.predict(eval_part.x)[target.id]
        outputs["decision_stump"], info["decision_stump"] = decision_stump(
            train_part, eval_part, target.output_dim
        )
    elif target.kind is TargetKind.REGRESSION:
        k = target.output_dim
        outputs["train_mean"] = np.zeros((n, k))
        outputs["zero_return"] = np.tile(
            -np.asarray(state["mean"]) / np.asarray(state["std"]), (n, 1)
        )
        w, b, info["ridge"] = _ridge(train_part, select_part)
        outputs["ridge"] = eval_part.x @ w + b
    elif target.kind is TargetKind.VOLATILITY:
        outputs["train_mean"] = np.zeros((n, 1))
        sigma = np.clip(np.asarray(eval_part.sigma, float), 1e-12, None)
        outputs["naive_realized_vol_20"] = (
            (np.log(sigma) - state["mean"][0]) / state["std"][0]
        )[:, None]
        w, b, info["ridge"] = _ridge(train_part, select_part)
        outputs["ridge"] = eval_part.x @ w + b
    else:
        raw = np.asarray(train_part.targets, float)[:, 0]
        q10, q50, q90 = np.quantile(raw, QUANTILES)
        outputs["train_quantiles"] = _quantile_outputs(
            np.full(n, q10), np.full(n, q50), np.full(n, q90), state
        )
        mu = float(raw.mean())
        scale = np.asarray(eval_part.sigma, float) * np.sqrt(target.horizons[0])
        outputs["normal_scaled_by_vol_20"] = _quantile_outputs(
            mu + NORMAL_Z[0.1] * scale,
            np.full(n, mu),
            mu + NORMAL_Z[0.9] * scale,
            state,
        )
        model, info["linear_quantile_regression"] = _linear(
            target, train_part, select_part, state
        )
        outputs["linear_quantile_regression"] = model.predict(eval_part.x)[target.id]
    return outputs, info
