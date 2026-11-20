"""Synthetic OHLCV bars for feature tests (no market data in the repository)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_bars(
    n: int = 300, seed: int = 0, dividend_every: int = 63
) -> pd.DataFrame:
    """Random-walk bars with a step-wise adjustment factor, like a paying ETF.

    ``adjusted_close / close`` drops on periodic "ex-dates" so tests exercise the
    raw versus adjusted scale distinction.
    """
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    gap = rng.normal(0, 0.004, n)
    opens = np.r_[close[0], close[:-1] * np.exp(gap[1:])]
    spread = np.abs(rng.normal(0, 0.006, n))
    high = np.maximum(opens, close) * (1 + spread)
    low = np.minimum(opens, close) * (1 - spread)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    steps = np.arange(n) // dividend_every
    factor = 0.995 ** (steps.max() - steps)
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adjusted_close": close * factor,
        },
        index=pd.bdate_range("2010-01-04", periods=n, name="date"),
    )


def perturb_after(frame: pd.DataFrame, position: int, seed: int = 99) -> pd.DataFrame:
    """Copy of ``frame`` whose bars after ``position`` are replaced by other valid bars."""
    other = synthetic_bars(len(frame), seed=seed)
    other.index = frame.index
    scale = frame["close"].iloc[position] / other["close"].iloc[position]
    out = frame.copy()
    tail = other.iloc[position + 1 :].copy()
    for column in ("open", "high", "low", "close", "adjusted_close"):
        tail[column] = tail[column] * scale * 1.7
    tail["volume"] = tail["volume"] * 3
    out.iloc[position + 1 :] = tail.to_numpy()
    return out
