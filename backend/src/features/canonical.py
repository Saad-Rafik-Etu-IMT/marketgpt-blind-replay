"""Canonical daily-bar frame used by the feature engine.

Input: one asset's bars with the F1 standard columns
(``open, high, low, close, volume, adjusted_close``), indexed by date or with a
``date`` column. Output: a validated copy indexed by a ``DatetimeIndex`` plus
the adjusted OHLC fields derived from Yahoo's adjustment factor.

Two price representations are never mixed (README, data policy): raw OHLC
drives replay fills; features use the adjusted scale. Opens, highs and lows
are moved onto the adjusted scale with the same-day factor
``adjusted_close / close``, so intraday ratios are unchanged and overnight
returns do not jump on dividend ex-dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
)
DERIVED_FIELDS: tuple[str, ...] = (
    "adj_factor",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
)
CANONICAL_FIELDS: frozenset[str] = frozenset(RAW_FIELDS + DERIVED_FIELDS)


class CanonicalFrameError(ValueError):
    """The bars cannot be used for feature computation."""


def to_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one asset's bars and add the adjusted OHLC fields.

    Refuses (never repairs) missing columns, duplicate or unordered dates,
    non-finite or non-positive prices, negative volume and broken OHLC order.
    """
    data = frame.copy()
    if "date" in data.columns:
        data = data.set_index("date")
    missing = [c for c in RAW_FIELDS if c not in data.columns]
    if missing:
        raise CanonicalFrameError(f"missing column(s) {missing}")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(data.index))
    except (TypeError, ValueError) as exc:
        raise CanonicalFrameError(f"index is not a date index: {exc}") from exc
    if index.tz is not None:
        raise CanonicalFrameError("dates must be timezone-naive trading days")
    if len(index) == 0:
        raise CanonicalFrameError("no bars")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise CanonicalFrameError("dates must be unique and strictly increasing")
    data.index = index.rename("date")
    data = data[list(RAW_FIELDS)].astype(np.float64)

    prices = data[["open", "high", "low", "close", "adjusted_close"]].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise CanonicalFrameError("prices must be finite and strictly positive")
    volume = data["volume"].to_numpy()
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise CanonicalFrameError("volume must be finite and non-negative")
    high, low = data["high"].to_numpy(), data["low"].to_numpy()
    body_max = np.maximum(data["open"].to_numpy(), data["close"].to_numpy())
    body_min = np.minimum(data["open"].to_numpy(), data["close"].to_numpy())
    if (high < body_max).any() or (low > body_min).any():
        raise CanonicalFrameError("OHLC order violated (low <= open, close <= high)")

    factor = data["adjusted_close"] / data["close"]
    data["adj_factor"] = factor
    data["adj_open"] = data["open"] * factor
    data["adj_high"] = data["high"] * factor
    data["adj_low"] = data["low"] * factor
    data["adj_close"] = data["adjusted_close"]
    return data
