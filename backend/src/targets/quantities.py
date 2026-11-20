"""Raw future quantities measured after decision bar ``t`` (decided at its close).

Every function returns one value per bar and NaN where the needed future bars
do not exist. ``bars_needed`` gives how many bars after ``t`` each one reads,
which is the horizon the example selector checks against the partitions.

- executed cumulative return over h sessions: filled at the next open,
  marked h opens later, ``log(adj_open[t+1+h] / adj_open[t+1])``; needs h + 1
  bars. h = 1 is the T3 ``executed_return_v1`` label.
- close-to-close cumulative return: ``log(adj_close[t+h] / adj_close[t])``.
- future realized volatility: sample std (ddof=1) of the daily close-to-close
  log returns of bars ``t+1 .. t+h``, per session (not annualized).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ZERO_VOLATILITY = 1e-10


def executed_bars(h: int) -> int:
    return h + 1


def executed_cumulative(bars: pd.DataFrame, h: int) -> np.ndarray:
    opens = bars["adj_open"].to_numpy()
    out = np.full(len(opens), np.nan)
    if len(opens) > h + 1:
        out[: len(opens) - h - 1] = np.log(opens[h + 1 :] / opens[1 : len(opens) - h])
    return out


def close_cumulative(bars: pd.DataFrame, h: int) -> np.ndarray:
    close = bars["adj_close"].to_numpy()
    out = np.full(len(close), np.nan)
    if len(close) > h:
        out[: len(close) - h] = np.log(close[h:] / close[: len(close) - h])
    return out


def future_volatility(bars: pd.DataFrame, h: int) -> np.ndarray:
    """Std of returns t+1..t+h; NaN if undefined or numerically zero (constant prices)."""
    if h < 2:
        raise ValueError("future volatility needs h >= 2 returns")
    close = bars["adj_close"].to_numpy()
    returns = np.diff(np.log(close))  # returns[i] ends at bar i + 1
    out = np.full(len(close), np.nan)
    if len(returns) >= h:
        sd = sliding_window_view(returns, h).std(axis=1, ddof=1)
        scale = np.abs(sliding_window_view(returns, h)).max(axis=1)
        sd = np.where(sd <= ZERO_VOLATILITY * np.maximum(scale, 1e-300), np.nan, sd)
        # window k covers returns k .. k+h-1, i.e. bars k+1 .. k+h: decision t = k.
        out[: len(sd)] = sd
    return out
