"""Group A: historical returns on the adjusted-close scale.

``returns_window_64`` reproduces the baseline input exactly: the 64 most recent
daily log returns ending at the decision bar, oldest first (the F3
``BlindObservation`` order and the W1 ``make_xy`` window).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.features.groups._window import mean, rolling
from src.features.spec import FeatureGroup, FeatureSpec, Visibility

WINDOW_SIZES: tuple[int, ...] = (16, 32, 64, 128)
CUMULATIVE_HORIZONS: tuple[int, ...] = (5, 20, 60)
MEAN_HORIZONS: tuple[int, ...] = (5, 20)
LOG_RETURN = "log_return_1d"


def _log_return(bars: pd.DataFrame, _: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    close = bars["adj_close"].to_numpy()
    values = np.full(len(close), np.nan)
    values[1:] = np.log(close[1:] / close[:-1])
    return pd.DataFrame({LOG_RETURN: values}, index=bars.index)


def _window(size: int):
    def compute(bars: pd.DataFrame, deps: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        r = deps[LOG_RETURN][LOG_RETURN]
        # Column lag k holds r[t - k]; columns run from lag size-1 down to lag 0.
        return pd.DataFrame(
            {
                f"returns_window_{size}.lag{k}": r.shift(k)
                for k in range(size - 1, -1, -1)
            },
            index=bars.index,
        )

    return compute


def _cumulative(horizon: int):
    name = f"cum_return_{horizon}"

    def compute(bars: pd.DataFrame, _: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = bars["adj_close"].to_numpy()
        values = np.full(len(close), np.nan)
        values[horizon:] = np.log(close[horizon:] / close[:-horizon])
        return pd.DataFrame({name: values}, index=bars.index)

    return compute


def _mean(horizon: int):
    name = f"mean_return_{horizon}"

    def compute(bars: pd.DataFrame, deps: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        r = deps[LOG_RETURN][LOG_RETURN].to_numpy()
        return pd.DataFrame({name: rolling(r, horizon, mean)}, index=bars.index)

    return compute


def _overnight(bars: pd.DataFrame, _: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    opens, close = bars["adj_open"].to_numpy(), bars["adj_close"].to_numpy()
    values = np.full(len(close), np.nan)
    values[1:] = np.log(opens[1:] / close[:-1])
    return pd.DataFrame({"overnight_return": values}, index=bars.index)


def _intraday(bars: pd.DataFrame, _: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    values = np.log(bars["adj_close"].to_numpy() / bars["adj_open"].to_numpy())
    return pd.DataFrame({"intraday_return": values}, index=bars.index)


def _window_redundancy(size: int) -> dict[str, object]:
    larger = tuple(f"returns_window_{n}" for n in WINDOW_SIZES if n > size)
    if not larger:
        return {}
    return {
        "redundant_with": larger,
        "redundancy_reason": f"its {size} columns are the last {size} columns of each larger window",
    }


def specs() -> list[FeatureSpec]:
    group = FeatureGroup.RETURNS
    units = "natural log return"
    out = [
        FeatureSpec(
            id=LOG_RETURN,
            version="1.0.0",
            group=group,
            description="Daily close-to-close log return.",
            definition="log(adj_close[t] / adj_close[t-1])",
            required_fields=("adj_close",),
            lookback=1,
            compute=_log_return,
            units=units,
            expected_range=(-0.1, 0.1),
        )
    ]
    for size in WINDOW_SIZES:
        out.append(
            FeatureSpec(
                id=f"returns_window_{size}",
                version="1.0.0",
                group=group,
                description=(
                    f"The {size} most recent daily log returns, oldest first"
                    + (" (baseline MLP input)." if size == 64 else ".")
                ),
                definition=f"[log_return_1d[t-k] for k = {size - 1} .. 0]",
                required_fields=(),
                dependencies=(LOG_RETURN,),
                lookback=size,
                compute=_window(size),
                columns=tuple(
                    f"returns_window_{size}.lag{k}" for k in range(size - 1, -1, -1)
                ),
                units=units,
                expected_range=(-0.1, 0.1),
                **_window_redundancy(size),
            )
        )
    for horizon in CUMULATIVE_HORIZONS:
        out.append(
            FeatureSpec(
                id=f"cum_return_{horizon}",
                version="1.0.0",
                group=group,
                description=f"Cumulative log return over the last {horizon} sessions.",
                definition=f"log(adj_close[t] / adj_close[t-{horizon}])",
                required_fields=("adj_close",),
                lookback=horizon,
                compute=_cumulative(horizon),
                units=units,
            )
        )
    for horizon in MEAN_HORIZONS:
        out.append(
            FeatureSpec(
                id=f"mean_return_{horizon}",
                version="1.0.0",
                group=group,
                description=f"Average daily log return over the last {horizon} sessions.",
                definition=f"mean(log_return_1d[t-{horizon - 1} .. t])",
                required_fields=(),
                dependencies=(LOG_RETURN,),
                lookback=horizon,
                compute=_mean(horizon),
                units=units,
                redundant_with=(f"cum_return_{horizon}",),
                redundancy_reason=f"mean_return_{horizon} = cum_return_{horizon} / {horizon} exactly",
            )
        )
    out += [
        FeatureSpec(
            id="overnight_return",
            version="1.0.0",
            group=group,
            description="Gap between the previous close and today's open.",
            definition="log(adj_open[t] / adj_close[t-1])",
            required_fields=("adj_open", "adj_close"),
            lookback=1,
            compute=_overnight,
            visibility=Visibility.PRICE_RELATIVE,
            units=units,
        ),
        FeatureSpec(
            id="intraday_return",
            version="1.0.0",
            group=group,
            description="Open-to-close return of the current session.",
            definition="log(adj_close[t] / adj_open[t]) = log(close[t] / open[t])",
            required_fields=("adj_open", "adj_close"),
            lookback=0,
            compute=_intraday,
            visibility=Visibility.PRICE_RELATIVE,
            units=units,
            determined_by=(LOG_RETURN, "overnight_return"),
            redundancy_reason="intraday_return = log_return_1d - overnight_return exactly",
        ),
    ]
    return out
