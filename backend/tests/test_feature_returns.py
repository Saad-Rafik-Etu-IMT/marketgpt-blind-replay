"""Group A (returns) numerical tests against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from feature_fixtures import synthetic_bars

from src.features import DEFAULT_REGISTRY, compute
from src.features.canonical import CanonicalFrameError


def tiny_bars() -> pd.DataFrame:
    """Six bars with an ex-dividend factor change between bars 2 and 3."""
    close = np.array([100.0, 102.0, 101.0, 99.0, 103.0, 104.0])
    opens = np.array([100.0, 101.0, 102.5, 98.0, 100.0, 103.5])
    adjusted = close * np.array([0.98, 0.98, 0.98, 1.0, 1.0, 1.0])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close) + 1,
            "low": np.minimum(opens, close) - 1,
            "close": close,
            "volume": 1000.0,
            "adjusted_close": adjusted,
        },
        index=pd.bdate_range("2021-01-04", periods=6),
    )


def test_log_return_cumulative_and_mean_by_hand():
    bars = tiny_bars()
    adj = bars["adjusted_close"].to_numpy()
    m = compute(bars, ["log_return_1d", "cum_return_5", "mean_return_5"]).values
    assert math.isnan(m["log_return_1d"].iloc[0])
    assert m["log_return_1d"].iloc[3] == pytest.approx(math.log(adj[3] / adj[2]))
    assert m["cum_return_5"].iloc[:5].isna().all()
    assert m["cum_return_5"].iloc[5] == pytest.approx(math.log(adj[5] / adj[0]))
    assert m["mean_return_5"].iloc[5] == pytest.approx(math.log(adj[5] / adj[0]) / 5)


def test_overnight_uses_adjusted_scale_across_an_ex_date():
    bars = tiny_bars()
    m = compute(bars, ["overnight_return", "intraday_return"]).values
    # Raw prices would give log(98 / 101) on the ex-date; adjusted removes the
    # 2% distribution: open 98 on factor 1.0 versus close 101 on factor 0.98.
    assert m["overnight_return"].iloc[3] == pytest.approx(
        math.log(98.0 / (101.0 * 0.98))
    )
    assert m["intraday_return"].iloc[0] == pytest.approx(0.0)
    assert m["intraday_return"].iloc[4] == pytest.approx(math.log(103.0 / 100.0))


def test_overnight_plus_intraday_equals_close_to_close():
    m = compute(
        synthetic_bars(200), ["log_return_1d", "overnight_return", "intraday_return"]
    ).values
    total = (m["overnight_return"] + m["intraday_return"]).iloc[1:]
    np.testing.assert_allclose(total, m["log_return_1d"].iloc[1:], rtol=0, atol=1e-12)


@pytest.mark.parametrize("size", [16, 32, 64, 128])
def test_windows_are_lagged_returns_oldest_first(size):
    m = compute(synthetic_bars(300), [f"returns_window_{size}", "log_return_1d"]).values
    r = m["log_return_1d"].to_numpy()
    window = m[
        [f"returns_window_{size}.lag{k}" for k in range(size - 1, -1, -1)]
    ].to_numpy()
    assert np.isnan(window[size - 1]).all()
    assert not np.isnan(window[size:]).any()
    t = 250
    np.testing.assert_array_equal(window[t], r[t - size + 1 : t + 1])


def test_sum_of_window_tail_matches_cumulative_return():
    m = compute(synthetic_bars(150), ["returns_window_64", "cum_return_20"]).values
    tail = m[[f"returns_window_64.lag{k}" for k in range(19, -1, -1)]].sum(axis=1)
    np.testing.assert_allclose(tail.iloc[64:], m["cum_return_20"].iloc[64:], atol=1e-12)


def test_short_history_is_all_warm_up_and_bad_prices_are_refused():
    bars = synthetic_bars(10)
    m = compute(bars, ["returns_window_64", "cum_return_60"]).values
    assert m.isna().all().all()
    single = compute(bars.iloc[:1], ["log_return_1d", "intraday_return"]).values
    assert math.isnan(single["log_return_1d"].iloc[0]) and math.isfinite(
        single["intraday_return"].iloc[0]
    )
    zero = bars.copy()
    zero.iloc[4, zero.columns.get_loc("adjusted_close")] = 0.0
    with pytest.raises(CanonicalFrameError):
        compute(zero, ["log_return_1d"])


def test_every_returns_feature_declares_complete_metadata():
    for spec in DEFAULT_REGISTRY.specs():
        described = spec.describe()
        assert described["units"] and described["availability"] == "close of bar t"
        assert spec.lookback >= 0 and spec.width >= 1
