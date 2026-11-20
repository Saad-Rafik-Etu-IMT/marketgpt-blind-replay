"""Small dependency-free statistics helpers for the training audit (NumPy only).

Everything here operates on validation-period artifacts already saved by a
study. Nothing reads the sealed test period.
"""

from __future__ import annotations

import math

import numpy as np


def macro_f1(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred).astype(int)
    y = np.asarray(y).astype(int)
    scores = []
    for cls in (0, 1):
        denom = int((pred == cls).sum() + (y == cls).sum())
        scores.append(2 * int(((pred == cls) & (y == cls)).sum()) / denom if denom else 0.0)
    return float(np.mean(scores))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact via log-gamma (no SciPy)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(p)
        + (n - i) * math.log1p(-p)
        for i in range(k, n + 1)
    ]
    m = max(logs)
    return float(min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs)))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value (doubling the smaller tail)."""
    upper = binom_sf(k, n, p)
    lower = 1.0 - binom_sf(k + 1, n, p)
    return float(min(1.0, 2.0 * min(upper, lower)))


def block_bootstrap_indices(
    n: int, n_boot: int = 2000, block: int = 10, seed: int = 0
) -> np.ndarray:
    """Circular moving-block bootstrap index matrix (n_boot, n)."""
    rng = np.random.default_rng(seed)
    n_blocks = (n + block - 1) // block
    starts = rng.integers(n, size=(n_boot, n_blocks))
    idx = ((starts[..., None] + np.arange(block)) % n).reshape(n_boot, -1)
    return idx[:, :n]


def bootstrap_ci(
    stat, *arrays: np.ndarray, n_boot: int = 2000, block: int = 10, seed: int = 0
) -> tuple[float, float]:
    n = len(arrays[0])
    idx = block_bootstrap_indices(n, n_boot, block, seed)
    vals = np.array([stat(*(a[i] for a in arrays)) for i in idx])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def permutation_null_f1(
    pred: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Macro-F1 of the SAME predictions against shuffled labels.

    Returns (95th percentile of the null, one-sided p-value of the observed F1).
    This null keeps the model's long share, so it answers "is the macro-F1
    better than a random predictor with the same class balance?".
    """
    rng = np.random.default_rng(seed)
    obs = macro_f1(pred, y)
    null = np.array([macro_f1(pred, rng.permutation(y)) for _ in range(n_perm)])
    return float(np.quantile(null, 0.95)), float((np.sum(null >= obs) + 1) / (n_perm + 1))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    p = np.asarray(p, dtype=float)
    which = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = which == b
        if m.any():
            out.append(
                dict(
                    bin=f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
                    n=int(m.sum()),
                    predicted=float(p[m].mean()),
                    observed=float(y[m].mean()),
                )
            )
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    rel = reliability(p, y, bins)
    n = len(p)
    return float(sum(r["n"] / n * abs(r["predicted"] - r["observed"]) for r in rel))


def histogram(p: np.ndarray, edges=(0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0001)) -> dict:
    counts, _ = np.histogram(np.asarray(p, dtype=float), bins=np.asarray(edges))
    return {f"[{a:.2f},{min(b, 1):.2f})": int(c) for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)}


def max_drawdown_from_log(log_rets: np.ndarray) -> float:
    eq = np.exp(np.cumsum(np.r_[0.0, log_rets]))
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / peak))


def long_flat_pnl(
    positions: np.ndarray, next_returns: np.ndarray, cost_per_switch: float = 0.00005
) -> dict:
    """Vectorized close-to-close long/flat P&L proxy (log returns).

    ``positions[i]`` is the position held over ``next_returns[i]``.
    ``cost_per_switch`` defaults to half the benchmark-v2 1 bp spread.
    This is a proxy for fast ablations, not the broker replay: it ignores
    next-open execution, whole-unit sizing and the cash reserve.
    """
    positions = np.asarray(positions, dtype=float)
    switches = np.abs(np.diff(np.r_[0.0, positions]))
    log_r = positions * next_returns - cost_per_switch * switches
    return dict(
        gain=float(np.exp(log_r.sum()) - 1),
        dd=max_drawdown_from_log(log_r),
        exposure=float(positions.mean()),
        switches=int(switches.sum()),
        sharpe=float(log_r.mean() / (log_r.std() + 1e-12) * math.sqrt(252)),
    )
