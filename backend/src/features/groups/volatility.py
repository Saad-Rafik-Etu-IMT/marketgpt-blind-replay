"""Group C: realized (historical) volatility and risk, per-session units.

Every value here is measured from past bars: realized volatility, not implied
volatility (an options quote, Group J) and not a volatility forecast (a model
output, tranche T4). Volatilities are standard deviations of daily log returns,
NOT annualized; multiply by sqrt(252) for an annual figure.
"""

from __future__ import annotations

import numpy as np

from src.features.groups._window import frame, mean, rolling, safe_ratio, shift, std
from src.features.spec import FeatureGroup, FeatureSpec, MissingPolicy, Visibility

RV_SPANS: tuple[int, ...] = (5, 10, 20, 60)
PERCENTILE_HISTORY = 252
ATR_LEN, BB_LEN, SEMI_LEN, MDD_LEN, DD_LEN = 14, 20, 20, 60, 252
LOG_RETURN = "log_return_1d"
PER_DAY = "std of daily log returns (per session, not annualized)"


def _r(deps) -> np.ndarray:
    return deps[LOG_RETURN][LOG_RETURN].to_numpy()


def _realized(span: int):
    def compute(bars, deps):
        return frame(
            bars.index, **{f"realized_vol_{span}": rolling(_r(deps), span, std(1))}
        )

    return compute


def _ratio(short: int, long: int):
    def compute(bars, deps):
        a = deps[f"realized_vol_{short}"][f"realized_vol_{short}"].to_numpy()
        b = deps[f"realized_vol_{long}"][f"realized_vol_{long}"].to_numpy()
        return frame(bars.index, **{f"vol_ratio_{short}_{long}": safe_ratio(a, b)})

    return compute


def _percentile(bars, deps):
    vol = deps["realized_vol_20"]["realized_vol_20"].to_numpy()

    def rank(w: np.ndarray) -> np.ndarray:
        # Share of the previous 251 values strictly below today's value.
        return (w[:, :-1] < w[:, -1:]).mean(axis=1)

    return frame(bars.index, vol_percentile_252=rolling(vol, PERCENTILE_HISTORY, rank))


def _atr_pct(bars, _):
    high, low, close = (
        bars[c].to_numpy() for c in ("adj_high", "adj_low", "adj_close")
    )
    prev = shift(close, 1)
    true_range = np.fmax(high, prev) - np.fmin(low, prev)
    true_range[0] = np.nan
    return frame(bars.index, atr_14_pct=rolling(true_range, ATR_LEN, mean) / close)


def _bb_width(bars, _):
    close = bars["adj_close"].to_numpy()
    width = 4.0 * rolling(close, BB_LEN, std(0)) / rolling(close, BB_LEN, mean)
    return frame(bars.index, bb_width_20=width)


def _semivol(bars, deps):
    down = np.minimum(_r(deps), 0.0)
    return frame(
        bars.index,
        downside_semivol_20=np.sqrt(rolling(down**2, SEMI_LEN, mean)),
    )


def _max_drawdown(bars, _):
    def worst(w: np.ndarray) -> np.ndarray:
        return (w / np.maximum.accumulate(w, axis=1) - 1.0).min(axis=1)

    return frame(
        bars.index,
        max_drawdown_60=rolling(bars["adj_close"].to_numpy(), MDD_LEN, worst),
    )


def _drawdown(bars, _):
    close = bars["adj_close"].to_numpy()
    peak = rolling(close, DD_LEN, lambda w: w.max(axis=1))
    return frame(bars.index, drawdown_252=close / peak - 1.0)


def specs() -> list[FeatureSpec]:
    g = FeatureGroup.VOLATILITY
    out = [
        FeatureSpec(
            id=f"realized_vol_{span}",
            version="1.0.0",
            group=g,
            description=f"Realized volatility over {span} sessions.",
            definition=f"std(log_return_1d[t-{span - 1} .. t], ddof=1)",
            required_fields=(),
            dependencies=(LOG_RETURN,),
            lookback=span,
            compute=_realized(span),
            units=PER_DAY,
            hard_bounds=(0.0, None),
        )
        for span in RV_SPANS
    ]
    for short, long in ((5, 20), (20, 60)):
        out.append(
            FeatureSpec(
                id=f"vol_ratio_{short}_{long}",
                version="1.0.0",
                group=g,
                description=f"Short/long realized volatility ratio ({short}/{long}); > 1 = expansion.",
                definition=f"realized_vol_{short} / realized_vol_{long}",
                required_fields=(),
                dependencies=(f"realized_vol_{short}", f"realized_vol_{long}"),
                lookback=long,
                compute=_ratio(short, long),
                missing_policy=MissingPolicy.ALLOW_NAN,
                units="ratio",
                hard_bounds=(0.0, None),
            )
        )
    out += [
        FeatureSpec(
            id="vol_percentile_252",
            version="1.0.0",
            group=g,
            description="Rank of today's 20-session volatility within the past year (no fitted threshold).",
            definition="share of realized_vol_20[t-251 .. t-1] strictly below realized_vol_20[t]",
            required_fields=(),
            dependencies=("realized_vol_20",),
            lookback=20 + PERCENTILE_HISTORY - 1,
            compute=_percentile,
            units="fraction 0-1",
            hard_bounds=(0.0, 1.0),
        ),
        FeatureSpec(
            id="atr_14_pct",
            version="1.0.0",
            group=g,
            description="Average true range over 14 sessions, as a fraction of the close.",
            definition="mean_14(max(H, C[t-1]) - min(L, C[t-1])) / C[t]",
            required_fields=("adj_high", "adj_low", "adj_close"),
            lookback=ATR_LEN,
            compute=_atr_pct,
            visibility=Visibility.PRICE_RELATIVE,
            units="fraction of price per session",
            hard_bounds=(0.0, None),
        ),
        FeatureSpec(
            id="bb_width_20",
            version="1.0.0",
            group=g,
            description="Bollinger band width (2 standard deviations each side) over the middle band.",
            definition="4 * std(C[t-19 .. t], ddof=0) / mean(C[t-19 .. t])",
            required_fields=("adj_close",),
            lookback=BB_LEN - 1,
            compute=_bb_width,
            units="fraction of price",
            hard_bounds=(0.0, None),
        ),
        FeatureSpec(
            id="downside_semivol_20",
            version="1.0.0",
            group=g,
            description="Downside semivolatility (= downside deviation with target 0) over 20 sessions.",
            definition="sqrt(mean(min(log_return_1d, 0)^2)) over the last 20 sessions",
            required_fields=(),
            dependencies=(LOG_RETURN,),
            lookback=SEMI_LEN,
            compute=_semivol,
            units=PER_DAY,
            hard_bounds=(0.0, None),
        ),
        FeatureSpec(
            id="max_drawdown_60",
            version="1.0.0",
            group=g,
            description="Worst peak-to-trough decline inside the last 60 closes.",
            definition="min over the window of C[i] / max(C[start .. i]) - 1",
            required_fields=("adj_close",),
            lookback=MDD_LEN - 1,
            compute=_max_drawdown,
            units="fraction (<= 0)",
            hard_bounds=(-1.0, 0.0),
        ),
        FeatureSpec(
            id="drawdown_252",
            version="1.0.0",
            group=g,
            description="Current decline from the highest close of the past year.",
            definition="C[t] / max(C[t-251 .. t]) - 1",
            required_fields=("adj_close",),
            lookback=DD_LEN - 1,
            compute=_drawdown,
            units="fraction (<= 0)",
            hard_bounds=(-1.0, 0.0),
        ),
    ]
    return out
