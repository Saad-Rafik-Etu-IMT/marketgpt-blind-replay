"""Group B: trend and momentum, as scale-free ratios on the adjusted scale.

Finite-memory variants replace the classic infinite-memory recursions so every
feature has an exact lookback and does not depend on where the series starts:

- EMA: normalized EMA weights over 4 x span bars (see ``_window.ema_weights``).
- RSI: Cutler's variant (simple 14-bar averages of gains and losses).
- ADX: simple 14-bar sums instead of Wilder smoothing, scaled to [0, 1].
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.features.groups._window import (
    ema,
    ema_weights,
    frame,
    mean,
    rolling,
    safe_ratio,
    shift,
)
from src.features.spec import FeatureGroup, FeatureSpec, Visibility

SMA_SPANS: tuple[int, ...] = (10, 20, 50, 200)
EMA_SPANS: tuple[int, ...] = (12, 26)
ROC_HORIZONS: tuple[int, ...] = (5, 10, 20, 60)
RSI_LEN, STOCH_LEN, STOCH_SMOOTH, ADX_LEN = 14, 14, 3, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
EMA_LOOKBACK = {span: len(ema_weights(span)) - 1 for span in (*EMA_SPANS, MACD_SIGNAL)}
MACD_LOOKBACK = EMA_LOOKBACK[MACD_SLOW]
SIGNAL_LOOKBACK = MACD_LOOKBACK + EMA_LOOKBACK[MACD_SIGNAL]
ADX_LOOKBACK = 1 + (ADX_LEN - 1) + (ADX_LEN - 1)
# Empirical redundancy (docs/features.md, "Redundancy"): SPY, IEF, QQQ, XLE,
# GLD, EEM, bars up to 2016-12-31 (training period only), Spearman.
EMA_TWIN = {12: "dist_sma_10", 26: "dist_sma_20"}
EMA_TWIN_RHO = {12: 0.961, 26: 0.952}

G = FeatureGroup.TREND
CLOSE = ("adj_close",)
HLC = ("adj_high", "adj_low", "adj_close")


def _close(bars: pd.DataFrame) -> np.ndarray:
    return bars["adj_close"].to_numpy()


def _dist_sma(span: int):
    def compute(bars: pd.DataFrame, _: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close(bars)
        return frame(
            bars.index, **{f"dist_sma_{span}": close / rolling(close, span, mean) - 1}
        )

    return compute


def _dist_ema(span: int):
    def compute(bars: pd.DataFrame, _: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close(bars)
        return frame(bars.index, **{f"dist_ema_{span}": close / ema(close, span) - 1})

    return compute


def _macd_line_values(close: np.ndarray) -> np.ndarray:
    slow = ema(close, MACD_SLOW)
    return (ema(close, MACD_FAST) - slow) / slow


def _macd_line(bars, _):
    return frame(bars.index, macd_line=_macd_line_values(_close(bars)))


def _macd_signal(bars, deps):
    return frame(
        bars.index,
        macd_signal=ema(deps["macd_line"]["macd_line"].to_numpy(), MACD_SIGNAL),
    )


def _macd_hist(bars, deps):
    line = deps["macd_line"]["macd_line"].to_numpy()
    return frame(
        bars.index, macd_hist=line - deps["macd_signal"]["macd_signal"].to_numpy()
    )


def _rsi(bars, _):
    diff = np.diff(_close(bars), prepend=np.nan)
    gains = rolling(np.maximum(diff, 0.0), RSI_LEN, mean)
    losses = rolling(np.maximum(-diff, 0.0), RSI_LEN, mean)
    # No movement at all over 14 bars: neutral 50 by definition.
    return frame(
        bars.index, rsi_14=100.0 * safe_ratio(gains, gains + losses, when_zero=0.5)
    )


def _roc(horizon: int):
    def compute(bars, _):
        close = _close(bars)
        return frame(
            bars.index, **{f"roc_{horizon}": close / shift(close, horizon) - 1}
        )

    return compute


def _stoch_k_values(bars: pd.DataFrame) -> np.ndarray:
    high = rolling(bars["adj_high"].to_numpy(), STOCH_LEN, lambda w: w.max(axis=1))
    low = rolling(bars["adj_low"].to_numpy(), STOCH_LEN, lambda w: w.min(axis=1))
    # Zero range over the window: the close sits mid-range by definition.
    return safe_ratio(_close(bars) - low, high - low, when_zero=0.5)


def _stoch_k(bars, _):
    return frame(bars.index, stoch_k_14=_stoch_k_values(bars))


def _stoch_d(bars, deps):
    k = deps["stoch_k_14"]["stoch_k_14"].to_numpy()
    return frame(bars.index, stoch_d_3=rolling(k, STOCH_SMOOTH, mean))


def _adx(bars, _):
    high, low, close = (bars[c].to_numpy() for c in HLC)
    prev_close, prev_high, prev_low = shift(close, 1), shift(high, 1), shift(low, 1)
    true_range = np.fmax(high, prev_close) - np.fmin(low, prev_close)
    true_range[0] = np.nan
    up, down = high - prev_high, prev_low - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan
    total = lambda w: w.sum(axis=1)  # noqa: E731
    tr_sum = rolling(true_range, ADX_LEN, total)
    plus_di = safe_ratio(rolling(plus_dm, ADX_LEN, total), tr_sum, when_zero=0.0)
    minus_di = safe_ratio(rolling(minus_dm, ADX_LEN, total), tr_sum, when_zero=0.0)
    # No directional movement at all: DX = 0 by definition.
    dx = safe_ratio(np.abs(plus_di - minus_di), plus_di + minus_di, when_zero=0.0)
    return frame(bars.index, adx_14=rolling(dx, ADX_LEN, mean))


def _momentum_acceleration(bars, _):
    close = _close(bars)
    momentum = np.log(close / shift(close, 10))
    return frame(bars.index, momentum_accel_10=momentum - shift(momentum, 10))


def _spec(**fields) -> FeatureSpec:
    base = dict(
        version="1.0.0",
        group=G,
        required_fields=CLOSE,
        visibility=Visibility.RETURNS_ONLY,
        units="fraction (ratio - 1)",
    )
    base.update(fields)
    return FeatureSpec(**base)


def specs() -> list[FeatureSpec]:
    out: list[FeatureSpec] = []
    for span in SMA_SPANS:
        out.append(
            _spec(
                id=f"dist_sma_{span}",
                description=f"Relative distance of the close to its {span}-session simple average.",
                definition=f"adj_close[t] / mean(adj_close[t-{span - 1} .. t]) - 1",
                lookback=span - 1,
                compute=_dist_sma(span),
            )
        )
    for span in EMA_SPANS:
        out.append(
            _spec(
                id=f"dist_ema_{span}",
                description=f"Relative distance of the close to its {span}-session EMA (finite window).",
                definition=f"adj_close[t] / EMA_{span}[t] - 1, EMA over the last {4 * span} closes",
                lookback=EMA_LOOKBACK[span],
                compute=_dist_ema(span),
                redundant_with=(EMA_TWIN[span],),
                redundancy_reason=(
                    f"empirical: Spearman |rho| with {EMA_TWIN[span]} >= 0.95 on the "
                    f"training period of all six reference ETFs (min {EMA_TWIN_RHO[span]})"
                ),
            )
        )
    out += [
        _spec(
            id="macd_line",
            description="MACD line normalized by the slow EMA.",
            definition="(EMA_12 - EMA_26) / EMA_26",
            lookback=MACD_LOOKBACK,
            compute=_macd_line,
        ),
        _spec(
            id="macd_signal",
            description="9-session EMA of the normalized MACD line.",
            definition="EMA_9(macd_line)",
            required_fields=(),
            dependencies=("macd_line",),
            lookback=SIGNAL_LOOKBACK,
            compute=_macd_signal,
        ),
        _spec(
            id="macd_hist",
            description="MACD histogram: line minus signal.",
            definition="macd_line - macd_signal",
            required_fields=(),
            dependencies=("macd_line", "macd_signal"),
            lookback=SIGNAL_LOOKBACK,
            compute=_macd_hist,
            determined_by=("macd_line", "macd_signal"),
            redundancy_reason="macd_hist = macd_line - macd_signal exactly",
        ),
        _spec(
            id="rsi_14",
            description="Relative strength index, Cutler variant (simple 14-session averages).",
            definition="100 * G / (G + L), G and L = mean positive / negative close changes; 50 if no change",
            lookback=RSI_LEN,
            compute=_rsi,
            units="index 0-100",
            expected_range=(0.0, 100.0),
            hard_bounds=(0.0, 100.0),
        ),
    ]
    for horizon in ROC_HORIZONS:
        twin = f"cum_return_{horizon}"
        has_twin = horizon in (5, 20, 60)
        out.append(
            _spec(
                id=f"roc_{horizon}",
                description=f"Rate of change over {horizon} sessions.",
                definition=f"adj_close[t] / adj_close[t-{horizon}] - 1",
                lookback=horizon,
                compute=_roc(horizon),
                **(
                    {
                        "redundant_with": (twin,),
                        "redundancy_reason": f"roc_{horizon} = exp({twin}) - 1 (monotone transform)",
                    }
                    if has_twin
                    else {}
                ),
            )
        )
    out += [
        _spec(
            id="stoch_k_14",
            description="Stochastic %K: close position inside the 14-session high-low range.",
            definition="(C - min L14) / (max H14 - min L14); 0.5 if the range is zero",
            required_fields=HLC,
            visibility=Visibility.PRICE_RELATIVE,
            lookback=STOCH_LEN - 1,
            compute=_stoch_k,
            units="fraction 0-1",
            hard_bounds=(0.0, 1.0),
        ),
        _spec(
            id="stoch_d_3",
            description="Stochastic %D: 3-session average of %K.",
            definition="mean(stoch_k_14[t-2 .. t])",
            required_fields=(),
            dependencies=("stoch_k_14",),
            visibility=Visibility.PRICE_RELATIVE,
            lookback=STOCH_LEN - 1 + STOCH_SMOOTH - 1,
            compute=_stoch_d,
            units="fraction 0-1",
            hard_bounds=(0.0, 1.0),
        ),
        _spec(
            id="adx_14",
            description="Average directional index (simple 14-session sums), trend strength.",
            definition="mean_14(|+DI - -DI| / (+DI + -DI)), DI = sum_14(DM) / sum_14(TR)",
            required_fields=HLC,
            visibility=Visibility.PRICE_RELATIVE,
            lookback=ADX_LOOKBACK,
            compute=_adx,
            units="fraction 0-1",
            hard_bounds=(0.0, 1.0),
        ),
        _spec(
            id="momentum_accel_10",
            description="Change of the 10-session log momentum over the last 10 sessions.",
            definition="log(C[t]/C[t-10]) - log(C[t-10]/C[t-20])",
            lookback=20,
            compute=_momentum_acceleration,
            units="natural log return",
        ),
    ]
    return out
