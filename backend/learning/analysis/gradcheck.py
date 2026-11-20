"""Finite-difference gradient check of ``deep_mlp_backward``.

The loss checked is exactly the one ``deep_mlp_backward`` differentiates: the
sample-weighted soft-target cross-entropy, with BatchNorm in training mode
(batch statistics) and a FIXED dropout mask (dropout is replayed by reusing the
same RNG seed for every forward pass).

Usage:
    .venv/Scripts/python.exe backend/learning/analysis/gradcheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mlp"))
from deep_mlp import deep_mlp_backward, deep_mlp_forward, init_deep_mlp  # noqa: E402


def _loss(params, x, t, w, dropout, seed):
    rm = {k: params[k].copy() for k in params if k.startswith(("rm", "rv"))}
    probs, cache = deep_mlp_forward(
        x, params, training=True, dropout=dropout, rng=np.random.default_rng(seed)
    )
    for k, v in rm.items():  # forward(training=True) mutates running stats
        params[k][:] = v
    per = -np.sum(t * np.log(np.clip(probs, 1e-300, 1)), axis=1)
    return float(np.sum(per * w) / np.sum(w)), cache


def gradient_check(
    n: int = 16, n_in: int = 7, dropout: float = 0.25, seed: int = 3, eps: float = 1e-6
) -> dict[str, float]:
    """Return the max relative error per parameter tensor."""
    rng = np.random.default_rng(seed)
    params = init_deep_mlp(n_in, h1=6, h2=5, h3=4, seed=seed)
    # Non-trivial BN affine params so their gradients are exercised.
    for k in ("gamma1", "gamma2", "gamma3", "beta1", "beta2", "beta3"):
        params[k] = params[k] + 0.3 * rng.normal(size=params[k].shape)
    x = rng.normal(size=(n, n_in))
    y = rng.integers(0, 2, n)
    t = 0.72 * np.eye(2)[y] + 0.28 * np.eye(2)[rng.integers(0, 2, n)]
    w = rng.uniform(0.5, 1.5, n)

    _, cache = _loss(params, x, t, w, dropout, seed)
    grads = deep_mlp_backward(cache, t, w, params)
    errors = {}
    for k, g in grads.items():
        num = np.zeros_like(params[k])
        it = np.nditer(params[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = params[k][i]
            params[k][i] = old + eps
            lp, _ = _loss(params, x, t, w, dropout, seed)
            params[k][i] = old - eps
            lm, _ = _loss(params, x, t, w, dropout, seed)
            params[k][i] = old
            num[i] = (lp - lm) / (2 * eps)
        # Mixed tolerance: pre-BatchNorm biases (b1, b2, b3) have an exact
        # gradient of 0 (BN removes the batch mean), so a pure relative error
        # would compare two round-off noises.
        denom = np.maximum(np.abs(num) + np.abs(g), 1e-6)
        errors[k] = float(np.max(np.abs(num - g) / denom))
    return errors


if __name__ == "__main__":
    for dp in (0.0, 0.25):
        errs = gradient_check(dropout=dp)
        worst = max(errs.values())
        print(f"dropout={dp}: worst relative error {worst:.2e}")
        for k, v in errs.items():
            print(f"  {k:8s} {v:.2e}")
