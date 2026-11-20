"""Deep feature-engineered MLP + Hindsight Best-Decisions (Oracle) logger.

Combines the architecture and feature engineering of Projet_LLM.ipynb
(RobustScaler, 128 -> 64 -> 32 -> 2 MLP with BatchNorm, ReLU, Dropout,
AdamW) with the Blind Replay 64-return observation contract and an
execution-aware Hindsight Best-Decisions logger. Selection and temperature use
an inner-validation set only (see ``protocol.py`` and docs/training_audit.md).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_best_decisions(
    values: np.ndarray,
    indices: tuple[int, ...],
    dates: list[Any] | None = None,
    split_name: str = "train",
    horizon: int = 3,
    spread_bps: float = 1.0,
) -> dict[str, Any]:
    """Compute and log hindsight-optimal ('best') long/flat decisions.

    Uses dynamic programming (Viterbi path) over cost- and drawdown-adjusted
    forward returns so that noisy 1-day blips inside a drawdown or chop are
    filtered out, while sustained upward moves are held without excess churn.
    """
    n_total = len(values)
    n_idx = len(indices)
    if n_idx == 0:
        return {
            "split": split_name,
            "horizon": horizon,
            "count": 0,
            "longShare": 0.0,
            "naiveUpShare": 0.0,
            "agreementRate": 0.0,
            "oracleCumulativeReturn": 0.0,
            "oracleMaxDrawdown": 0.0,
            "naiveFilteredCount": 0,
            "trendHeldCount": 0,
            "labels": np.zeros(0, dtype=np.int64),
            "weights": np.ones(0, dtype=np.float64),
            "records": [],
        }

    switch_cost = max(spread_bps / 10000.0, 0.0005)
    # The split guarantees only bar ``t + 1`` (label_horizon = 1) lies inside the
    # set, so forward paths stop at the set's last label bar. Without this clamp
    # the last ``horizon - 1`` examples read gap bars beyond the set boundary.
    last_label_bar = int(indices[-1]) + 1
    next_1d = np.zeros(n_idx, dtype=np.float64)
    fwd_h = np.zeros(n_idx, dtype=np.float64)
    fwd_min = np.zeros(n_idx, dtype=np.float64)
    local_vol = np.zeros(n_idx, dtype=np.float64)

    for i, t in enumerate(indices):
        r1 = float(values[t + 1]) if t + 1 < n_total else 0.0
        end_h = min(n_total, t + 1 + horizon, last_label_bar + 1)
        path = values[t + 1 : end_h]
        cum_path = np.cumsum(path) if len(path) > 0 else np.array([0.0])
        past_window = values[max(0, t - 13) : t + 1]
        vol = float(np.std(past_window)) if len(past_window) > 1 else 0.01

        next_1d[i] = r1
        fwd_h[i] = float(cum_path[-1])
        fwd_min[i] = float(np.min(cum_path))
        local_vol[i] = max(vol, 1e-4)

    # Risk-adjusted step reward for holding long at decision i:
    # blends immediate t+1 return with 3-day horizon drift and penalizes adverse excursion.
    step_utility = (
        0.55 * next_1d
        + 0.45 * (fwd_h / max(1, horizon))
        - 0.25 * np.maximum(0.0, -fwd_min - 0.6 * local_vol)
    )

    # Dynamic programming (Viterbi) for optimal state sequence s_i in {0 (flat), 1 (long)}
    dp = np.zeros((n_idx, 2), dtype=np.float64)
    back = np.zeros((n_idx, 2), dtype=np.int64)
    dp[0, 0] = 0.0
    dp[0, 1] = step_utility[0] - 0.5 * switch_cost

    for i in range(1, n_idx):
        for s_curr in (0, 1):
            reward = step_utility[i] if s_curr == 1 else 0.0
            stay = dp[i - 1, s_curr]
            flip = dp[i - 1, 1 - s_curr] - switch_cost * (1.2 + 0.6 * (local_vol[i] / 0.01))
            if stay >= flip:
                dp[i, s_curr] = stay + reward
                back[i, s_curr] = s_curr
            else:
                dp[i, s_curr] = flip + reward
                back[i, s_curr] = 1 - s_curr

    best_path = np.zeros(n_idx, dtype=np.int64)
    best_path[-1] = int(np.argmax(dp[-1]))
    for i in range(n_idx - 2, -1, -1):
        best_path[i] = back[i + 1, best_path[i + 1]]

    naive_up = (next_1d > 0).astype(np.int64)
    agrees = best_path == naive_up

    # Sample importance weights: higher confidence when 1d and 3d oracle agree and move is strong
    snr = np.clip(np.abs(next_1d) / local_vol, 0.25, 2.5)
    weights = np.where(agrees, 1.0 + 0.35 * snr, 0.65).astype(np.float64)
    weights /= np.mean(weights)

    # Simulate oracle equity curve (with switch costs)
    oracle_log_rets = best_path * next_1d - switch_cost * np.abs(
        np.diff(np.r_[0, best_path])
    )
    oracle_equity = np.exp(np.cumsum(oracle_log_rets))
    running_peak = np.maximum.accumulate(np.r_[1.0, oracle_equity])
    dd_series = (running_peak[1:] - oracle_equity) / np.maximum(running_peak[1:], 1e-12)
    oracle_dd = float(np.max(dd_series)) if len(dd_series) else 0.0
    oracle_gain = float(oracle_equity[-1] - 1.0) if len(oracle_equity) else 0.0

    naive_filtered = int(np.sum((naive_up == 1) & (best_path == 0)))
    trend_held = int(np.sum((naive_up == 0) & (best_path == 1)))

    records = []
    for i, t in enumerate(indices):
        date_str = str(dates[t])[:10] if dates is not None and t < len(dates) else f"bar-{t}"
        period_str = date_str[:7] if len(date_str) >= 7 else "unknown"
        if best_path[i] == 1 and naive_up[i] == 1:
            reason = "Confirmed upward move (1d & 3d positive)"
        elif best_path[i] == 0 and naive_up[i] == 1:
            reason = "Filtered noisy 1d bounce ahead of 3d drawdown"
        elif best_path[i] == 1 and naive_up[i] == 0:
            reason = "Held trend through minor 1d dip"
        else:
            reason = "Avoided downward / negative drift regime"
        records.append(
            {
                "index": i,
                "barIndex": int(t),
                "period": period_str,
                "bestDecision": "long" if best_path[i] == 1 else "flat",
                "naiveUp": bool(naive_up[i]),
                "nextReturn": round(float(next_1d[i]), 6),
                "forward3dReturn": round(float(fwd_h[i]), 6),
                "localVolatility": round(float(local_vol[i]), 6),
                "reason": reason,
            }
        )

    return {
        "split": split_name,
        "horizon": horizon,
        "count": n_idx,
        "longShare": float(np.mean(best_path)),
        "naiveUpShare": float(np.mean(naive_up)),
        "agreementRate": float(np.mean(agrees)),
        "oracleCumulativeReturn": oracle_gain,
        "oracleMaxDrawdown": oracle_dd,
        "naiveFilteredCount": naive_filtered,
        "trendHeldCount": trend_held,
        "labels": best_path,
        "weights": weights,
        "records": records,
    }


def extract_blind_features(windows: np.ndarray) -> np.ndarray:
    """Extract Projet_LLM.ipynb stationary technical features from 64 log returns.

    Strictly obeys the BlindObservation boundary: uses ONLY the 64 log returns
    in each row of ``windows`` (shape: (N, 64)), with no external OHLCV or date inputs.
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n, w = x.shape
    eps = 1e-8

    # Relative price curve inside the 64-bar window (starting at 1.0)
    log_p = np.cumsum(x, axis=1)
    p = np.exp(log_p)

    feats = [x]  # Base 64 log returns

    # 1. Multi-Horizon Log Returns across the trailing 10-bar sub-window (as in Projet_LLM.ipynb)
    # For trailing steps t in [54..63], compute k-bar cumulative returns for k in [1, 3, 5, 10, 20]
    for k in (1, 3, 5, 10, 20):
        # Cumulative log return over k bars at each of the last 10 steps
        mhlr_steps = []
        for t in range(w - 10, w):
            start = max(0, t - k + 1)
            mhlr_steps.append(np.sum(x[:, start : t + 1], axis=1))
        feats.append(np.stack(mhlr_steps, axis=1))

    # Full 40-bar and 64-bar cumulative log returns
    feats.append(np.sum(x[:, -40:], axis=1, keepdims=True))
    feats.append(np.sum(x, axis=1, keepdims=True))

    # 2. Volatility Estimators (NATR proxy, Vol_Regime, Realized Vol, Downside Vol)
    abs_x = np.abs(x)
    vol_5 = np.mean(abs_x[:, -5:], axis=1, keepdims=True)
    vol_14 = np.mean(abs_x[:, -14:], axis=1, keepdims=True)
    vol_20 = np.mean(abs_x[:, -20:], axis=1, keepdims=True)
    vol_64 = np.mean(abs_x, axis=1, keepdims=True)
    vol_regime = vol_5 / (vol_20 + eps)

    std_5 = np.std(x[:, -5:], axis=1, keepdims=True)
    std_10 = np.std(x[:, -10:], axis=1, keepdims=True)
    std_20 = np.std(x[:, -20:], axis=1, keepdims=True)
    std_64 = np.std(x, axis=1, keepdims=True)
    std_regime = std_5 / (std_20 + eps)

    neg_x_20 = np.minimum(x[:, -20:], 0.0)
    downside_vol_20 = np.sqrt(np.mean(neg_x_20**2, axis=1, keepdims=True) + eps)
    upside_x_20 = np.maximum(x[:, -20:], 0.0)
    upside_vol_20 = np.sqrt(np.mean(upside_x_20**2, axis=1, keepdims=True) + eps)
    vol_asymmetry = (upside_vol_20 - downside_vol_20) / (std_20 + eps)

    feats.extend(
        [
            vol_5,
            vol_14,
            vol_20,
            vol_64,
            vol_regime,
            std_5,
            std_10,
            std_20,
            std_64,
            std_regime,
            downside_vol_20,
            vol_asymmetry,
        ]
    )

    # 3. Trend & Mean-Reversion Indicators (Dist_SMA, BB_Pct_B, RSI_14, Donchian_20, Drawdown)
    p_end = p[:, -1:]
    for span in (5, 10, 20, 50, 64):
        sma = np.mean(p[:, -span:], axis=1, keepdims=True)
        dist_sma = (p_end / (sma + eps)) - 1.0
        feats.append(dist_sma)

    # Trailing 5-step Dist_SMA_10 and BB_Pct_B
    sma_20 = np.mean(p[:, -20:], axis=1, keepdims=True)
    p_std_20 = np.std(p[:, -20:], axis=1, keepdims=True)
    bb_pct_b = (p_end - (sma_20 - 2.0 * p_std_20)) / (4.0 * p_std_20 + eps)
    feats.append(bb_pct_b)

    # RSI_14 (normalized [0, 1]) over trailing 14 and 28 bars
    for rsi_len in (7, 14, 28):
        sub = x[:, -rsi_len:]
        gains = np.mean(np.maximum(sub, 0.0), axis=1, keepdims=True)
        losses = np.mean(np.maximum(-sub, 0.0), axis=1, keepdims=True)
        rsi = 1.0 - (1.0 / (1.0 + gains / (losses + eps)))
        feats.append(rsi)

    # Donchian_20 & Donchian_64 on relative close path
    for d_len in (10, 20, 64):
        p_sub = p[:, -d_len:]
        low_d = np.min(p_sub, axis=1, keepdims=True)
        high_d = np.max(p_sub, axis=1, keepdims=True)
        donchian = (p_end - low_d) / (high_d - low_d + eps)
        feats.append(donchian)

    # Running drawdown from 64-bar peak and bounce from 64-bar trough
    peak_64 = np.max(p, axis=1, keepdims=True)
    trough_64 = np.min(p, axis=1, keepdims=True)
    dd_64 = (p_end - peak_64) / (peak_64 + eps)
    bounce_64 = (p_end - trough_64) / (trough_64 + eps)

    # Risk-adjusted momentum (Sharpe proxy over 5, 10, 20, 64 bars)
    sharpe_5 = np.mean(x[:, -5:], axis=1, keepdims=True) / (std_5 + eps)
    sharpe_10 = np.mean(x[:, -10:], axis=1, keepdims=True) / (std_10 + eps)
    sharpe_20 = np.mean(x[:, -20:], axis=1, keepdims=True) / (std_20 + eps)
    sharpe_64 = np.mean(x, axis=1, keepdims=True) / (std_64 + eps)

    # Return sign consistency & autocorrelation proxy
    pos_ratio_10 = np.mean((x[:, -10:] > 0).astype(np.float64), axis=1, keepdims=True)
    pos_ratio_20 = np.mean((x[:, -20:] > 0).astype(np.float64), axis=1, keepdims=True)
    autocorr_1 = np.mean(
        (x[:, -20:-1] - np.mean(x[:, -20:-1], axis=1, keepdims=True))
        * (x[:, -19:] - np.mean(x[:, -19:], axis=1, keepdims=True)),
        axis=1,
        keepdims=True,
    ) / (std_20**2 + eps)

    feats.extend(
        [
            dd_64,
            bounce_64,
            sharpe_5,
            sharpe_10,
            sharpe_20,
            sharpe_64,
            pos_ratio_10,
            pos_ratio_20,
            autocorr_1,
        ]
    )

    out = np.hstack(feats)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fit_robust_scaler(features_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-only RobustScaler (median and IQR with std fallback), as in Projet_LLM.ipynb."""
    median = np.median(features_train, axis=0)
    q75 = np.percentile(features_train, 75, axis=0)
    q25 = np.percentile(features_train, 25, axis=0)
    iqr = q75 - q25
    std = np.std(features_train, axis=0)
    scale = np.where(iqr > 1e-7, iqr, np.where(std > 1e-7, std, 1.0))
    return median.astype(np.float64), scale.astype(np.float64)


def transform_robust(
    features: np.ndarray, median: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply train-fitted RobustScaler and clip extreme outliers (> 5 IQR)."""
    scaled = (features - median) / scale
    return np.clip(scaled, -5.0, 5.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _weighted_cross_entropy(
    probs: np.ndarray,
    soft_targets: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    per_sample = -np.sum(soft_targets * np.log(p), axis=1)
    if sample_weights is not None:
        return float(np.sum(per_sample * sample_weights) / np.maximum(np.sum(sample_weights), 1e-12))
    return float(np.mean(per_sample))


def init_deep_mlp(
    n_in: int,
    h1: int = 128,
    h2: int = 64,
    h3: int = 32,
    n_out: int = 2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Initialize 3-hidden-layer MLP (128 -> 64 -> 32 -> 2) with BatchNorm1d & skip projection."""
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {
        "W1": rng.normal(0.0, math.sqrt(2.0 / n_in), size=(n_in, h1)),
        "b1": np.zeros(h1, dtype=np.float64),
        "gamma1": np.ones(h1, dtype=np.float64),
        "beta1": np.zeros(h1, dtype=np.float64),
        "rm1": np.zeros(h1, dtype=np.float64),
        "rv1": np.ones(h1, dtype=np.float64),
        "W2": rng.normal(0.0, math.sqrt(2.0 / h1), size=(h1, h2)),
        "b2": np.zeros(h2, dtype=np.float64),
        "gamma2": np.ones(h2, dtype=np.float64),
        "beta2": np.zeros(h2, dtype=np.float64),
        "rm2": np.zeros(h2, dtype=np.float64),
        "rv2": np.ones(h2, dtype=np.float64),
        "W3": rng.normal(0.0, math.sqrt(2.0 / h2), size=(h2, h3)),
        "b3": np.zeros(h3, dtype=np.float64),
        "gamma3": np.ones(h3, dtype=np.float64),
        "beta3": np.zeros(h3, dtype=np.float64),
        "rm3": np.zeros(h3, dtype=np.float64),
        "rv3": np.ones(h3, dtype=np.float64),
        # Residual linear shortcut from input to penultimate 32-dim representation
        "W_skip": rng.normal(0.0, math.sqrt(0.5 / n_in), size=(n_in, h3)),
        "W4": rng.normal(0.0, math.sqrt(1.0 / h3), size=(h3, n_out)),
        "b4": np.zeros(n_out, dtype=np.float64),
    }
    return params


def _bn_forward(
    z: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    training: bool,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    if training and len(z) > 1:
        mu = np.mean(z, axis=0)
        var = np.var(z, axis=0)
        running_mean[:] = (1.0 - momentum) * running_mean + momentum * mu
        running_var[:] = (1.0 - momentum) * running_var + momentum * var
    else:
        mu = running_mean
        var = running_var
    inv_std = 1.0 / np.sqrt(var + eps)
    z_hat = (z - mu) * inv_std
    out = gamma * z_hat + beta
    return out, {"z": z, "z_hat": z_hat, "mu": mu, "inv_std": inv_std, "gamma": gamma}


def _bn_backward(dout: np.ndarray, cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_hat = cache["z_hat"]
    inv_std = cache["inv_std"]
    gamma = cache["gamma"]
    m = max(1, len(dout))

    dgamma = np.sum(dout * z_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dz_hat = dout * gamma
    dz = (inv_std / m) * (
        m * dz_hat
        - np.sum(dz_hat, axis=0)
        - z_hat * np.sum(dz_hat * z_hat, axis=0)
    )
    return dz, dgamma, dbeta


def deep_mlp_forward(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    training: bool = False,
    dropout: float = 0.25,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Forward pass through the 128 -> 64 -> 32 -> 2 BatchNorm + Dropout Deep MLP."""
    # Layer 1: Linear -> BatchNorm1d -> ReLU -> Dropout
    z1 = x @ params["W1"] + params["b1"]
    bn1, c_bn1 = _bn_forward(
        z1, params["gamma1"], params["beta1"], params["rm1"], params["rv1"], training
    )
    a1 = np.maximum(bn1, 0.0)
    if training and dropout > 0.0 and rng is not None:
        m1 = (rng.random(a1.shape) >= dropout).astype(np.float64) / (1.0 - dropout)
        h1 = a1 * m1
    else:
        m1 = np.ones_like(a1)
        h1 = a1

    # Layer 2: Linear -> BatchNorm1d -> ReLU -> Dropout
    z2 = h1 @ params["W2"] + params["b2"]
    bn2, c_bn2 = _bn_forward(
        z2, params["gamma2"], params["beta2"], params["rm2"], params["rv2"], training
    )
    a2 = np.maximum(bn2, 0.0)
    if training and dropout > 0.0 and rng is not None:
        m2 = (rng.random(a2.shape) >= dropout).astype(np.float64) / (1.0 - dropout)
        h2 = a2 * m2
    else:
        m2 = np.ones_like(a2)
        h2 = a2

    # Layer 3: Linear -> BatchNorm1d -> ReLU + Residual Skip
    z3 = h2 @ params["W3"] + params["b3"]
    bn3, c_bn3 = _bn_forward(
        z3, params["gamma3"], params["beta3"], params["rm3"], params["rv3"], training
    )
    a3 = np.maximum(bn3, 0.0)
    skip = x @ params["W_skip"]
    h3 = a3 + 0.35 * skip

    # Output Layer
    logits = h3 @ params["W4"] + params["b4"]
    probs = _softmax(logits)

    cache = {
        "x": x,
        "bn1": bn1,
        "c_bn1": c_bn1,
        "m1": m1,
        "h1": h1,
        "bn2": bn2,
        "c_bn2": c_bn2,
        "m2": m2,
        "h2": h2,
        "bn3": bn3,
        "c_bn3": c_bn3,
        "h3": h3,
        "logits": logits,
        "probs": probs,
    }
    return probs, cache


def deep_mlp_backward(
    cache: dict[str, Any],
    soft_targets: np.ndarray,
    sample_weights: np.ndarray,
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Analytical gradients for the 3-layer BatchNorm + Dropout + Residual MLP."""
    probs = cache["probs"]
    w_norm = sample_weights[:, None] / np.maximum(np.sum(sample_weights), 1e-12)
    dlogits = (probs - soft_targets) * w_norm

    h3 = cache["h3"]
    dW4 = h3.T @ dlogits
    db4 = np.sum(dlogits, axis=0)
    dh3 = dlogits @ params["W4"].T

    x = cache["x"]
    dW_skip = 0.35 * (x.T @ dh3)

    da3 = dh3
    dbn3 = da3 * (cache["bn3"] > 0)
    dz3, dgamma3, dbeta3 = _bn_backward(dbn3, cache["c_bn3"])
    h2 = cache["h2"]
    dW3 = h2.T @ dz3
    db3 = np.sum(dz3, axis=0)
    dh2 = dz3 @ params["W3"].T

    da2 = dh2 * cache["m2"]
    dbn2 = da2 * (cache["bn2"] > 0)
    dz2, dgamma2, dbeta2 = _bn_backward(dbn2, cache["c_bn2"])
    h1 = cache["h1"]
    dW2 = h1.T @ dz2
    db2 = np.sum(dz2, axis=0)
    dh1 = dz2 @ params["W2"].T

    da1 = dh1 * cache["m1"]
    dbn1 = da1 * (cache["bn1"] > 0)
    dz1, dgamma1, dbeta1 = _bn_backward(dbn1, cache["c_bn1"])
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "gamma1": dgamma1,
        "beta1": dbeta1,
        "W2": dW2,
        "b2": db2,
        "gamma2": dgamma2,
        "beta2": dbeta2,
        "W3": dW3,
        "b3": db3,
        "gamma3": dgamma3,
        "beta3": dbeta3,
        "W_skip": dW_skip,
        "W4": dW4,
        "b4": db4,
    }


def _macro_f1(pred_binary: np.ndarray, y_true: np.ndarray) -> float:
    scores = []
    for cls in (0, 1):
        denom = int(np.sum(pred_binary == cls) + np.sum(y_true == cls))
        scores.append(
            float(2 * np.sum((pred_binary == cls) & (y_true == cls)) / denom)
            if denom > 0
            else 0.0
        )
    return float(np.mean(scores))


def fit_temperature(diff_logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the binary NLL of sigmoid(diff / T) on held-out data.

    Searched on a log grid; T > 1 softens, T < 1 sharpens. Only data the model
    was not trained on (inner-validation) may be passed here.
    """
    if len(y) == 0 or float(np.std(diff_logits)) < 1e-9:
        return 1.0
    y = np.asarray(y, dtype=np.float64)
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.25), np.log(20.0), 121)):
        z = diff_logits / t
        # log(1 + exp(-|z|)) formulation of the logistic NLL, numerically stable
        nll = float(np.mean(np.logaddexp(0.0, z) - y * z))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def train_deep_mlp(
    xt_raw: np.ndarray,
    yt: np.ndarray,
    xs_raw: np.ndarray,
    ys: np.ndarray,
    *,
    train_oracle: dict[str, Any] | None = None,
    seed: int = 0,
    epochs: int = 40,
    lr: float = 1.5e-3,
    weight_decay: float = 2e-3,
    dropout: float = 0.25,
    batch_size: int = 64,
    class_balance: bool = False,
    calibrate: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int]:
    """Train the deep MLP; ``xs``/``ys`` is the SELECTION set (inner-validation).

    Protocol (training audit 2026-11-20):
    - the retained epoch has the lowest selection-set cross-entropy;
    - an optional temperature is fitted by NLL on the same selection set;
    - no class re-weighting by default (it fights the market's drift);
    - the oracle soft targets are opt-in (``train_oracle``); they were neutral
      for discrimination in the audit.
    The validation set must not be passed here; it is evaluated once afterwards.
    """
    feat_train = extract_blind_features(xt_raw)
    feat_sel = extract_blind_features(xs_raw)
    median, scale = fit_robust_scaler(feat_train)
    x_train = transform_robust(feat_train, median, scale)
    x_sel = transform_robust(feat_sel, median, scale)

    n_train, n_in = x_train.shape
    params = init_deep_mlp(n_in=n_in, h1=128, h2=64, h3=32, n_out=2, seed=seed)
    rng = np.random.default_rng(seed)

    count_0 = max(1, int(np.sum(yt == 0)))
    count_1 = max(1, int(np.sum(yt == 1)))
    w_cls = (
        np.array([n_train / (2.0 * count_0), n_train / (2.0 * count_1)])
        if class_balance
        else np.ones(2)
    )
    y_onehot = np.eye(2, dtype=np.float64)[yt]
    if train_oracle is not None and len(train_oracle["labels"]) == n_train:
        soft_train = 0.72 * y_onehot + 0.28 * np.eye(2)[train_oracle["labels"]]
        base_weights = train_oracle["weights"] * w_cls[yt]
    else:
        soft_train = y_onehot
        base_weights = w_cls[yt]
    base_weights = base_weights / np.mean(base_weights)
    y_sel_onehot = np.eye(2, dtype=np.float64)[ys]

    trainable_keys = [
        k for k in params if not k.startswith("rm") and not k.startswith("rv")
    ]
    m_state = {k: np.zeros_like(params[k]) for k in trainable_keys}
    v_state = {k: np.zeros_like(params[k]) for k in trainable_keys}
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    step_num = 0

    history: list[dict[str, Any]] = []
    best_loss = np.inf
    best_params: dict[str, np.ndarray] | None = None
    selected_epoch = 0

    for epoch in range(1, epochs + 1):
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, epochs)))
        lr_t = lr * (0.15 + 0.85 * cosine)

        order = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            batch_idx = order[start : start + batch_size]
            if len(batch_idx) < 2:
                continue
            step_num += 1
            _, cache = deep_mlp_forward(
                x_train[batch_idx], params, training=True, dropout=dropout, rng=rng
            )
            grads = deep_mlp_backward(
                cache, soft_train[batch_idx], base_weights[batch_idx], params
            )
            bias_corr1 = 1.0 - beta1**step_num
            bias_corr2 = 1.0 - beta2**step_num
            for k in trainable_keys:
                g = np.clip(grads[k], -2.0, 2.0)
                m_state[k] = beta1 * m_state[k] + (1.0 - beta1) * g
                v_state[k] = beta2 * v_state[k] + (1.0 - beta2) * (g * g)
                m_hat = m_state[k] / bias_corr1
                v_hat = v_state[k] / bias_corr2
                if k.startswith("W"):
                    params[k] *= 1.0 - lr_t * weight_decay
                params[k] -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)

        train_probs, _ = deep_mlp_forward(x_train, params, training=False)
        sel_probs, _ = deep_mlp_forward(x_sel, params, training=False)
        train_loss = _weighted_cross_entropy(train_probs, y_onehot)
        sel_loss = _weighted_cross_entropy(sel_probs, y_sel_onehot)
        sel_preds = sel_probs.argmax(axis=1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(sel_loss),
                "val_accuracy": float(np.mean(sel_preds == ys)),
                "val_f1": _macro_f1(sel_preds, ys),
            }
        )
        if best_params is None or sel_loss < best_loss:
            best_loss = sel_loss
            best_params = {k: v.copy() for k, v in params.items()}
            selected_epoch = epoch

    assert best_params is not None
    for entry in history:
        entry["selected"] = entry["epoch"] == selected_epoch

    temperature = 1.0
    if calibrate:
        _, sel_cache = deep_mlp_forward(x_sel, best_params, training=False)
        diff = sel_cache["logits"][:, 1] - sel_cache["logits"][:, 0]
        temperature = fit_temperature(diff, ys)
        best_params["W4"] = best_params["W4"] / temperature
        best_params["b4"] = best_params["b4"] / temperature
    best_params["temperature"] = np.array(temperature)

    best_params["scaler_median"] = median
    best_params["scaler_scale"] = scale
    param_count = sum(best_params[k].size for k in trainable_keys)
    return best_params, history, param_count


def predict_deep_mlp(windows: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Predict [P(flat), P(long)] from raw 64-return windows."""
    feats = extract_blind_features(windows)
    x_scaled = transform_robust(feats, params["scaler_median"], params["scaler_scale"])
    probs, _ = deep_mlp_forward(x_scaled, params, training=False)
    return probs
