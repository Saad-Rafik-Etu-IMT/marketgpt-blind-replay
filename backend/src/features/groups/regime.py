"""Group G: interpretable market regimes with fixed, predeclared thresholds.

No threshold is fitted on any data, so no regime can leak validation or test
information. Each regime is a coarse version of a continuous feature and is
declared ``redundant_with`` it: profiles keep the continuous value for models;
the regime labels serve explanations and regime-sliced evaluation (T7).

Thresholds (version 1.0.0, chosen before any experiment):

- volatility: vol_percentile_252 < 1/3 -> -1 (calm), > 2/3 -> +1 (stressed)
- trend: sign of dist_sma_200
- momentum: sign of cum_return_60
- drawdown: drawdown_252 above -5% -> 0, -5% to -10% -> 1, -10% to -20% -> 2, below -20% -> 3
- volatility expansion: vol_ratio_5_20 > 1.25 -> +1, < 0.8 -> -1
"""

from __future__ import annotations

import numpy as np

from src.features.groups._window import frame, safe_ratio
from src.features.spec import FeatureGroup, FeatureSpec, MissingPolicy, Visibility


def _value(deps, name: str) -> np.ndarray:
    return deps[name][name].to_numpy()


def _three_state(values: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.where(values > high, 1.0, np.where(values < low, -1.0, 0.0))
    return np.where(np.isnan(values), np.nan, out)


def _sign(values: np.ndarray) -> np.ndarray:
    return np.sign(values)


def _vol_regime(bars, deps):
    value = _three_state(_value(deps, "vol_percentile_252"), 1 / 3, 2 / 3)
    return frame(bars.index, regime_volatility=value)


def _trend_regime(bars, deps):
    return frame(bars.index, regime_trend=_sign(_value(deps, "dist_sma_200")))


def _momentum_regime(bars, deps):
    return frame(bars.index, regime_momentum=_sign(_value(deps, "cum_return_60")))


def _drawdown_regime(bars, deps):
    dd = _value(deps, "drawdown_252")
    value = np.select(
        [dd > -0.05, dd > -0.10, dd > -0.20], [0.0, 1.0, 2.0], default=3.0
    )
    return frame(bars.index, regime_drawdown=np.where(np.isnan(dd), np.nan, value))


def _expansion(bars, deps):
    value = _three_state(_value(deps, "vol_ratio_5_20"), 0.8, 1.25)
    return frame(bars.index, regime_vol_expansion=value)


def _interaction(bars, deps):
    value = safe_ratio(_value(deps, "dist_sma_50"), _value(deps, "realized_vol_20"))
    return frame(bars.index, trend_vol_interaction=value)


def _spec(source: str, lookback: int, **fields) -> FeatureSpec:
    base = dict(
        version="1.0.0",
        group=FeatureGroup.REGIME,
        required_fields=(),
        dependencies=(source,),
        lookback=lookback,
        units="category",
        redundant_with=(source,),
        redundancy_reason=f"coarse version of {source} with fixed thresholds",
    )
    base.update(fields)
    return FeatureSpec(**base)


def specs() -> list[FeatureSpec]:
    return [
        _spec(
            "vol_percentile_252",
            271,
            id="regime_volatility",
            compute=_vol_regime,
            description="Volatility regime: -1 calm, 0 normal, +1 stressed.",
            definition="-1 if vol_percentile_252 < 1/3; +1 if > 2/3; else 0",
            hard_bounds=(-1.0, 1.0),
        ),
        _spec(
            "dist_sma_200",
            199,
            id="regime_trend",
            compute=_trend_regime,
            description="Trend regime: +1 above the 200-session average, -1 below.",
            definition="sign(dist_sma_200)",
            hard_bounds=(-1.0, 1.0),
        ),
        _spec(
            "cum_return_60",
            60,
            id="regime_momentum",
            compute=_momentum_regime,
            description="Momentum regime: sign of the 60-session return.",
            definition="sign(cum_return_60)",
            hard_bounds=(-1.0, 1.0),
        ),
        _spec(
            "drawdown_252",
            251,
            id="regime_drawdown",
            compute=_drawdown_regime,
            description="Drawdown regime: 0 near the high, 3 in a bear market (< -20%).",
            definition="0 if dd > -5%; 1 if > -10%; 2 if > -20%; else 3 (dd = drawdown_252)",
            hard_bounds=(0.0, 3.0),
        ),
        _spec(
            "vol_ratio_5_20",
            20,
            id="regime_vol_expansion",
            compute=_expansion,
            description="Volatility expansion (+1) or contraction (-1) of 5- versus 20-session volatility.",
            definition="+1 if vol_ratio_5_20 > 1.25; -1 if < 0.8; else 0",
            missing_policy=MissingPolicy.ALLOW_NAN,
            hard_bounds=(-1.0, 1.0),
        ),
        FeatureSpec(
            id="trend_vol_interaction",
            version="1.0.0",
            group=FeatureGroup.REGIME,
            description="Distance to the 50-session average in units of daily volatility.",
            definition="dist_sma_50 / realized_vol_20",
            required_fields=(),
            dependencies=("dist_sma_50", "realized_vol_20"),
            lookback=49,
            compute=_interaction,
            visibility=Visibility.RETURNS_ONLY,
            missing_policy=MissingPolicy.ALLOW_NAN,
            units="daily standard deviations",
        ),
    ]
