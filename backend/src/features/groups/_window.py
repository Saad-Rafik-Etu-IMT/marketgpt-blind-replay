"""Window helpers: every rolling statistic is computed from its own window only.

``np.lib.stride_tricks.sliding_window_view`` gives each row exactly the bars
``t - size + 1 .. t``. Unlike running-sum implementations, the result does not
depend on where the series starts, so offline training and replay agree even
when their frames begin on different dates.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

WindowFn = Callable[[np.ndarray], np.ndarray]


def rolling(values: np.ndarray, size: int, fn: WindowFn) -> np.ndarray:
    """``fn`` applied to each window of ``size`` values (axis=1); NaN before the first."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan)
    if size < 1 or len(values) < size:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        out[size - 1 :] = fn(sliding_window_view(values, size))
    return out


def rolling_pair(x: np.ndarray, y: np.ndarray, size: int, fn) -> np.ndarray:
    """``fn(window_x, window_y)`` for aligned windows of two series."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if len(x) < size:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        out[size - 1 :] = fn(sliding_window_view(x, size), sliding_window_view(y, size))
    return out


def mean(w: np.ndarray) -> np.ndarray:
    return w.mean(axis=1)


# A spread below this fraction of the window's magnitude is floating-point noise
# (a constant series gives ~1e-17, not 0); it is snapped to exactly 0 so that
# ratios divided by it become NaN ("undefined") instead of huge random values.
RELATIVE_SPREAD_TOLERANCE = 1e-10


def snap_spread(spread: np.ndarray, windows: np.ndarray) -> np.ndarray:
    scale = np.abs(windows).max(axis=1)
    return np.where(spread <= RELATIVE_SPREAD_TOLERANCE * scale, 0.0, spread)


def std(ddof: int) -> WindowFn:
    return lambda w: snap_spread(w.std(axis=1, ddof=ddof), w)


def safe_ratio(
    num: np.ndarray, den: np.ndarray, *, when_zero: float = np.nan
) -> np.ndarray:
    """``num / den`` with an explicit value where ``den == 0`` (never inf)."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    zero = den == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(num, den, out=out, where=~zero & ~np.isnan(den))
    out[zero] = when_zero
    return out


def correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two window matrices; NaN if a side is constant."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sx = snap_spread(np.sqrt((xc**2).mean(axis=1)), x)
    sy = snap_spread(np.sqrt((yc**2).mean(axis=1)), y)
    return safe_ratio((xc * yc).mean(axis=1), sx * sy)


def ema_weights(span: int) -> np.ndarray:
    """Normalized EMA weights over a finite 4 x span window, newest weight last.

    alpha = 2 / (span + 1). Truncation leaves (1 - alpha)^(4 span) of the
    infinite EMA's weight out (0.03% for span 26): a documented approximation
    that buys an exact lookback and start-independence.
    """
    alpha = 2.0 / (span + 1.0)
    weights = (1.0 - alpha) ** np.arange(4 * span)[::-1]
    return weights / weights.sum()


def ema(values: np.ndarray, span: int) -> np.ndarray:
    weights = ema_weights(span)
    return rolling(values, len(weights), lambda w: w @ weights)


def frame(index: pd.Index, **columns: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(columns, index=index)


def shift(values: np.ndarray, k: int) -> np.ndarray:
    """``values[t - k]`` at row ``t`` (NaN for the first ``k`` rows)."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan)
    if k < len(values):
        out[k:] = values[: len(values) - k]
    return out
