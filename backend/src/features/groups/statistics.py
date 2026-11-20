"""Group F: statistical behaviour of recent daily log returns.

Moments use population formulas over a full window (60 returns minimum).
Undefined values (constant returns) are NaN, never 0. Rolling beta and
correlation against a benchmark need a second aligned series: they belong to
Group H (external data, tranche T8).
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
    shift,
    snap_spread,
    std,
)
from src.features.spec import FeatureGroup, FeatureSpec, MissingPolicy

LOG_RETURN = "log_return_1d"
MOMENT_LEN = 60


def _r(deps) -> np.ndarray:
    return deps[LOG_RETURN][LOG_RETURN].to_numpy()


def _autocorr(bars, deps):
    r = _r(deps)
    return frame(
        bars.index, autocorr_1_60=rolling_pair(r, shift(r, 1), MOMENT_LEN, correlation)
    )


def _moments(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population std (snapped to 0 when degenerate), third and fourth moments."""
    centered = w - w.mean(axis=1, keepdims=True)
    sd = snap_spread(np.sqrt((centered**2).mean(axis=1)), w)
    return sd, (centered**3).mean(axis=1), (centered**4).mean(axis=1)


def _skew(bars, deps):
    def fn(w):
        sd, m3, _ = _moments(w)
        return safe_ratio(m3, sd**3)

    return frame(bars.index, skew_60=rolling(_r(deps), MOMENT_LEN, fn))


def _kurt(bars, deps):
    def fn(w):
        sd, _, m4 = _moments(w)
        return safe_ratio(m4, sd**4) - 3.0

    return frame(bars.index, kurtosis_60=rolling(_r(deps), MOMENT_LEN, fn))


def _return_to_vol(bars, deps):
    r = _r(deps)
    return frame(
        bars.index,
        return_to_vol_20=safe_ratio(rolling(r, 20, mean), rolling(r, 20, std(1))),
    )


def _up_fraction(bars, deps):
    return frame(
        bars.index, up_fraction_20=rolling((_r(deps) > 0).astype(float), 20, mean)
    )


def _efficiency(bars, deps):
    def fn(w):
        return safe_ratio(np.abs(w.sum(axis=1)), np.abs(w).sum(axis=1))

    return frame(bars.index, efficiency_ratio_20=rolling(_r(deps), 20, fn))


def _spec(**fields) -> FeatureSpec:
    base = dict(
        version="1.0.0",
        group=FeatureGroup.STATISTICS,
        required_fields=(),
        dependencies=(LOG_RETURN,),
        missing_policy=MissingPolicy.ALLOW_NAN,
    )
    base.update(fields)
    return FeatureSpec(**base)


def specs() -> list[FeatureSpec]:
    return [
        _spec(
            id="autocorr_1_60",
            description="Lag-1 autocorrelation of daily returns over 60 sessions.",
            definition="corr(r[t-59 .. t], r[t-60 .. t-1])",
            lookback=MOMENT_LEN + 1,
            compute=_autocorr,
            units="correlation",
            hard_bounds=(-1.0, 1.0),
        ),
        _spec(
            id="skew_60",
            description="Skewness of daily returns over 60 sessions (population formula).",
            definition="m3 / m2^1.5 over r[t-59 .. t]",
            lookback=MOMENT_LEN,
            compute=_skew,
            units="dimensionless",
        ),
        _spec(
            id="kurtosis_60",
            description="Excess kurtosis of daily returns over 60 sessions (population formula).",
            definition="m4 / m2^2 - 3 over r[t-59 .. t]",
            lookback=MOMENT_LEN,
            compute=_kurt,
            units="dimensionless",
            hard_bounds=(-3.0, None),
        ),
        _spec(
            id="return_to_vol_20",
            description="Mean daily return over its standard deviation, 20 sessions (not annualized).",
            definition="mean_20(r) / std_20(r) (ddof=1)",
            lookback=20,
            compute=_return_to_vol,
            units="ratio per session",
        ),
        _spec(
            id="up_fraction_20",
            description="Share of the last 20 sessions with a positive return.",
            definition="mean_20(r > 0)",
            lookback=20,
            compute=_up_fraction,
            missing_policy=MissingPolicy.WARMUP_NAN,
            units="fraction 0-1",
            hard_bounds=(0.0, 1.0),
        ),
        _spec(
            id="efficiency_ratio_20",
            description="Kaufman efficiency ratio: net move over total path, 20 sessions.",
            definition="|sum_20 r| / sum_20 |r|; NaN if no movement",
            lookback=20,
            compute=_efficiency,
            units="fraction 0-1",
            hard_bounds=(0.0, 1.0),
        ),
    ]
