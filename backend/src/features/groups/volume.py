"""Group E: volume relative to its own history (activity proxies).

These are OHLCV-derived proxies, NOT order-book liquidity: daily volume says
nothing about the executable bid-ask spread or market depth. Absolute volume
and dollar volume reveal the asset, so only ratios and z-scores are exposed.
Classic On-Balance Volume accumulates from the first bar of the file, which
makes its value depend on the download start; ``obv_flow_20`` is the
finite-window, volume-normalized equivalent of its 20-session slope.
"""

from __future__ import annotations

import numpy as np

from src.features.groups._window import (
    correlation,
    frame,
    mean,
    rolling,
    rolling_pair,
    safe_ratio,
    std,
)
from src.features.spec import FeatureGroup, FeatureSpec, MissingPolicy, Visibility

LOG_RETURN = "log_return_1d"


def _volume(bars) -> np.ndarray:
    return bars["volume"].to_numpy()


def _rel_volume(bars, _):
    v = _volume(bars)
    return frame(bars.index, rel_volume_20=safe_ratio(v, rolling(v, 20, mean)))


def _volume_z(bars, _):
    lv = np.log1p(_volume(bars))
    value = safe_ratio(lv - rolling(lv, 20, mean), rolling(lv, 20, std(1)))
    return frame(bars.index, volume_z_20=value)


def _volume_change(bars, _):
    v = _volume(bars)

    def change(w: np.ndarray) -> np.ndarray:
        return np.log1p(w[:, 5:].sum(axis=1)) - np.log1p(w[:, :5].sum(axis=1))

    return frame(bars.index, volume_change_5=rolling(v, 10, change))


def _obv_flow(bars, deps):
    signed = np.sign(deps[LOG_RETURN][LOG_RETURN].to_numpy()) * _volume(bars)
    total = lambda w: w.sum(axis=1)  # noqa: E731
    value = safe_ratio(rolling(signed, 20, total), rolling(_volume(bars), 20, total))
    return frame(bars.index, obv_flow_20=value)


def _price_volume_corr(bars, deps):
    r = deps[LOG_RETURN][LOG_RETURN].to_numpy()
    dlv = np.diff(np.log1p(_volume(bars)), prepend=np.nan)
    return frame(bars.index, price_volume_corr_20=rolling_pair(r, dlv, 20, correlation))


def _volume_trend(bars, _):
    x = np.arange(20.0) - 9.5
    weights = x / (x**2).sum()
    lv = np.log1p(_volume(bars))
    return frame(bars.index, volume_trend_20=rolling(lv, 20, lambda w: w @ weights))


def _dollar_volume_z(bars, _):
    ldv = np.log1p(bars["close"].to_numpy() * _volume(bars))
    value = safe_ratio(ldv - rolling(ldv, 60, mean), rolling(ldv, 60, std(1)))
    return frame(bars.index, dollar_volume_z_60=value)


def _spec(**fields) -> FeatureSpec:
    base = dict(
        version="1.0.0",
        group=FeatureGroup.VOLUME,
        required_fields=("volume",),
        visibility=Visibility.VOLUME_RELATIVE,
        missing_policy=MissingPolicy.ALLOW_NAN,
        adjustment="Yahoo volume (split-adjusted); used only relative to its own history",
    )
    base.update(fields)
    return FeatureSpec(**base)


def specs() -> list[FeatureSpec]:
    return [
        _spec(
            id="rel_volume_20",
            description="Today's volume over its 20-session average.",
            definition="V[t] / mean(V[t-19 .. t]); NaN if the average is 0",
            lookback=19,
            compute=_rel_volume,
            units="ratio",
            hard_bounds=(0.0, None),
        ),
        _spec(
            id="volume_z_20",
            description="Z-score of log volume against the last 20 sessions.",
            definition="(log1p V[t] - mean_20) / std_20 (ddof=1); NaN if constant",
            lookback=19,
            compute=_volume_z,
            units="standard deviations",
        ),
        _spec(
            id="volume_change_5",
            description="Log change of total volume: last 5 sessions versus the 5 before.",
            definition="log1p(sum V[t-4 .. t]) - log1p(sum V[t-9 .. t-5])",
            lookback=9,
            compute=_volume_change,
            missing_policy=MissingPolicy.WARMUP_NAN,
            units="natural log ratio",
        ),
        _spec(
            id="obv_flow_20",
            description="Signed volume share (finite-window On-Balance Volume slope).",
            definition="sum_20(sign(log_return_1d) * V) / sum_20(V)",
            required_fields=("volume",),
            dependencies=(LOG_RETURN,),
            lookback=20,
            compute=_obv_flow,
            units="fraction -1..1",
            hard_bounds=(-1.0, 1.0),
        ),
        _spec(
            id="price_volume_corr_20",
            description="Correlation of daily returns with daily log volume changes.",
            definition="corr_20(log_return_1d, diff(log1p V)); NaN if either is constant",
            dependencies=(LOG_RETURN,),
            lookback=20,
            compute=_price_volume_corr,
            units="correlation",
            hard_bounds=(-1.0, 1.0),
        ),
        _spec(
            id="volume_trend_20",
            description="Least-squares slope of log volume over the last 20 sessions.",
            definition="OLS slope of log1p V[t-19 .. t] on time",
            lookback=19,
            compute=_volume_trend,
            missing_policy=MissingPolicy.WARMUP_NAN,
            units="log volume per session",
        ),
        _spec(
            id="dollar_volume_z_60",
            description="Z-score of log traded value (raw close x volume) against 60 sessions.",
            definition="(log1p(close * V)[t] - mean_60) / std_60 (ddof=1)",
            required_fields=("close", "volume"),
            lookback=59,
            compute=_dollar_volume_z,
            adjustment="raw close x Yahoo volume: the traded value; the level is removed by the z-score",
            units="standard deviations",
        ),
    ]
