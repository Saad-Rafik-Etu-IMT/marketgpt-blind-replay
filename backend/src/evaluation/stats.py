"""Per-row losses, paired date-block bootstrap, verdicts and drift diagnostics.

Paired comparisons use the SAME rows for both predictors and resample blocks
of consecutive calendar dates: every asset's rows of a date stay together, so
cross-asset correlation on a day is not counted as independent evidence, and
blocks of at least the label horizon keep overlapping labels together.

A difference is "better" or "worse" only when its 95% interval excludes 0;
otherwise the verdict is "no measurable difference", which is not proof of
equivalence. Seed spread is reported separately and never used as a
significance test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.targets import Target, TargetKind, softmax
from src.targets.targets import QUANTILES, softplus

EPS = 1e-12


def row_losses(
    target: Target, z: np.ndarray, y: np.ndarray, raw: np.ndarray, state: dict[str, Any]
) -> np.ndarray:
    """Primary loss of every row, in the target's evaluation units (lower is better).

    classification: negative log-likelihood; regression: absolute error of the
    first horizon in return units; volatility: QLIKE; quantiles: mean pinball
    loss in return units.
    """
    z = target.check_output(z)
    if target.kind is TargetKind.CLASSIFICATION:
        p = softmax(z)
        return -np.log(np.clip(p[np.arange(len(y)), np.asarray(y, int)], EPS, 1.0))
    if target.kind is TargetKind.REGRESSION:
        prediction = np.asarray(state["mean"]) + np.asarray(state["std"]) * z
        return np.abs(np.asarray(raw, float) - prediction).mean(axis=1)
    if target.kind is TargetKind.VOLATILITY:
        sigma_hat = np.exp(state["mean"][0] + state["std"][0] * z[:, 0])
        ratio = (np.asarray(raw, float)[:, 0] / sigma_hat) ** 2
        return ratio - np.log(ratio) - 1
    center = z[:, 0]
    q = np.column_stack(
        [center - softplus(z[:, 1]), center, center + softplus(z[:, 2])]
    )
    q = state["mean"][0] + state["std"][0] * q
    diff = np.asarray(raw, float)[:, :1] - q
    taus = np.asarray(QUANTILES)
    return np.maximum(taus * diff, (taus - 1) * diff).mean(axis=1)


def loss_name(target: Target) -> str:
    return {
        TargetKind.CLASSIFICATION: "log_loss",
        TargetKind.REGRESSION: "mean_absolute_error",
        TargetKind.VOLATILITY: "qlike",
        TargetKind.QUANTILE: "mean_pinball",
    }[target.kind]


def block_bootstrap_mean(
    values: np.ndarray,
    dates: np.ndarray,
    *,
    block: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval of the mean of ``values`` by circular blocks of consecutive dates."""
    values = np.asarray(values, float)
    unique, inverse = np.unique(np.asarray(dates), return_inverse=True)
    per_date_sum = np.bincount(inverse, weights=values, minlength=len(unique))
    per_date_count = np.bincount(inverse, minlength=len(unique)).astype(float)
    n = len(unique)
    if n == 0:
        return float("nan"), float("nan")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = (n + block - 1) // block
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    days = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(
        n_boot, -1
    )[:, :n]
    means = per_date_sum[days].sum(axis=1) / per_date_count[days].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def verdict(low: float, high: float) -> str:
    """For a difference of LOSSES (model - reference): negative = model better."""
    if not np.isfinite(low) or not np.isfinite(high):
        return "undefined"
    if high < 0:
        return "better"
    if low > 0:
        return "worse"
    return "no measurable difference"


def paired_difference(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    dates: np.ndarray,
    *,
    block: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean of (a - b) per row with its date-block interval and verdict for ``a``."""
    diff = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    low, high = block_bootstrap_mean(diff, dates, block=block, n_boot=n_boot, seed=seed)
    return {
        "mean_difference": float(diff.mean()),
        "interval_95": [low, high],
        "verdict": verdict(low, high),
        "rows": int(len(diff)),
        "dates": int(len(np.unique(dates))),
        "block": block,
    }


def interval_of(
    values: np.ndarray, dates: np.ndarray, *, block: int, n_boot: int = 1000
) -> dict[str, Any]:
    low, high = block_bootstrap_mean(values, dates, block=block, n_boot=n_boot)
    return {"mean": float(np.mean(values)), "interval_95": [low, high]}


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index with decile bins fitted on ``reference``."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / max(1, len(current))
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift(
    x_ref: np.ndarray, x_cur: np.ndarray, columns: tuple[str, ...], top: int = 15
) -> list[dict[str, Any]]:
    """Standardized mean difference and PSI per column, largest PSI first."""
    out = []
    for j, name in enumerate(columns):
        ref, cur = x_ref[:, j], x_cur[:, j]
        sd = ref.std()
        out.append(
            {
                "column": name,
                "standardized_mean_difference": (
                    float((cur.mean() - ref.mean()) / sd) if sd > 0 else None
                ),
                "psi": psi(ref, cur),
            }
        )
    out.sort(
        key=lambda d: -(
            d["psi"] if d["psi"] is not None and np.isfinite(d["psi"]) else -1
        )
    )
    return out[:top]
