"""Groups B to G: hand-computed values, identities and numerical edge cases."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from feature_fixtures import synthetic_bars

from src.features import DEFAULT_REGISTRY, FeatureGroup, compute
from src.features.groups._window import ema_weights, safe_ratio, shift


def bars_from(close, *, opens=None, high=None, low=None, volume=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    opens = close.copy() if opens is None else np.asarray(opens, dtype=float)
    high = np.maximum(opens, close) if high is None else np.asarray(high, dtype=float)
    low = np.minimum(opens, close) if low is None else np.asarray(low, dtype=float)
    volume = (
        np.full(len(close), 1000.0)
        if volume is None
        else np.asarray(volume, dtype=float)
    )
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adjusted_close": close,
        },
        index=pd.bdate_range("2015-01-01", periods=len(close)),
    )


def values(bars, *ids):
    return compute(bars, list(ids)).values


SYN = synthetic_bars(400, seed=11)
SYN_VALUES = values(SYN, *DEFAULT_REGISTRY.ids())
GROWTH = bars_from(100 * 1.001 ** np.arange(300))


# ------------------------------------------------------------------ helpers
def test_safe_ratio_and_ema_weights():
    np.testing.assert_array_equal(
        safe_ratio(np.array([1.0, 2.0]), np.array([0.0, 4.0])), [np.nan, 0.5]
    )
    assert safe_ratio(np.array([1.0]), np.array([0.0]), when_zero=7.0)[0] == 7.0
    weights = ema_weights(26)
    assert len(weights) == 104 and weights.sum() == pytest.approx(1.0)
    assert weights[-1] / weights[-2] == pytest.approx(1 / (1 - 2 / 27))
    np.testing.assert_array_equal(shift(np.arange(3.0), 5), [np.nan] * 3)


# -------------------------------------------------------------------- trend
def test_trend_on_constant_growth():
    v = values(
        GROWTH,
        "dist_sma_10",
        "macd_line",
        "momentum_accel_10",
        "roc_10",
        "rsi_14",
        "adx_14",
    )
    t = 250
    sma = np.mean(100 * 1.001 ** np.arange(t - 9, t + 1))
    assert v["dist_sma_10"].iloc[t] == pytest.approx(100 * 1.001**t / sma - 1)
    assert v["macd_line"].iloc[t] > 0
    assert v["momentum_accel_10"].iloc[t] == pytest.approx(0.0, abs=1e-12)
    assert v["roc_10"].iloc[t] == pytest.approx(1.001**10 - 1)
    assert v["rsi_14"].iloc[t] == pytest.approx(100.0)
    # Rising highs and lows only: all directional movement is positive.
    assert v["adx_14"].iloc[t] == pytest.approx(1.0)


def test_trend_on_a_flat_market():
    flat = bars_from(np.full(200, 50.0))
    v = values(flat, "macd_hist", "rsi_14", "adx_14", "stoch_k_14", "dist_ema_26")
    assert (v["macd_hist"].dropna() == 0).all() and (
        v["dist_ema_26"].dropna() == 0
    ).all()
    assert (v["rsi_14"].dropna() == 50.0).all()
    assert (v["adx_14"].dropna() == 0.0).all()
    assert (v["stoch_k_14"].dropna() == 0.5).all()


def test_rsi_and_stochastic_by_hand():
    close = np.array([10, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16.0])
    v = values(bars_from(close), "rsi_14", "stoch_k_14")
    diff = np.diff(close)
    gains, losses = np.maximum(diff, 0).mean(), np.maximum(-diff, 0).mean()
    assert v["rsi_14"].iloc[14] == pytest.approx(100 * gains / (gains + losses))
    assert v["stoch_k_14"].iloc[14] == pytest.approx((16 - 10) / (17 - 10))


def test_trend_identities_on_synthetic_bars():
    v = SYN_VALUES
    np.testing.assert_allclose(v["roc_20"], np.expm1(v["cum_return_20"]), atol=1e-12)
    np.testing.assert_allclose(
        v["macd_hist"], v["macd_line"] - v["macd_signal"], atol=1e-15
    )
    np.testing.assert_allclose(
        v["stoch_d_3"].iloc[20:],
        v["stoch_k_14"].rolling(3).mean().iloc[20:],
        atol=1e-12,
    )


# --------------------------------------------------------------- volatility
def test_volatility_against_numpy():
    v, r = SYN_VALUES, SYN_VALUES["log_return_1d"].to_numpy()
    t = 350
    assert v["realized_vol_20"].iloc[t] == pytest.approx(
        np.std(r[t - 19 : t + 1], ddof=1)
    )
    assert v["vol_ratio_5_20"].iloc[t] == pytest.approx(
        np.std(r[t - 4 : t + 1], ddof=1) / np.std(r[t - 19 : t + 1], ddof=1)
    )
    down = np.minimum(r[t - 19 : t + 1], 0)
    assert v["downside_semivol_20"].iloc[t] == pytest.approx(np.sqrt(np.mean(down**2)))
    close = SYN["adjusted_close"].to_numpy()
    window = close[t - 19 : t + 1]
    assert v["bb_width_20"].iloc[t] == pytest.approx(4 * window.std() / window.mean())
    assert v["drawdown_252"].iloc[t] == pytest.approx(
        close[t] / close[t - 251 : t + 1].max() - 1
    )
    rv = v["realized_vol_20"].to_numpy()
    assert v["vol_percentile_252"].iloc[t] == pytest.approx(
        np.mean(rv[t - 251 : t] < rv[t])
    )


def test_atr_and_max_drawdown_by_hand():
    close = np.array(
        [100, 102, 99, 101, 98, 103, 104, 100, 97, 99, 101, 102, 100, 98, 99.0]
    )
    high, low = close + 1, close - 2
    v = values(bars_from(close, opens=close, high=high, low=low), "atr_14_pct")
    true_range = [
        max(high[i], close[i - 1]) - min(low[i], close[i - 1]) for i in range(1, 15)
    ]
    assert v["atr_14_pct"].iloc[14] == pytest.approx(np.mean(true_range) / close[14])

    path = np.r_[np.full(30, 100.0), 120, 90, 110, np.full(27, 100.0)]
    mdd = values(bars_from(path), "max_drawdown_60")["max_drawdown_60"]
    assert mdd.iloc[59] == pytest.approx(90 / 120 - 1)


def test_constant_growth_has_zero_realized_volatility():
    v = values(GROWTH, "realized_vol_20", "vol_ratio_5_20", "drawdown_252")
    assert v["realized_vol_20"].iloc[100] == pytest.approx(0.0, abs=1e-15)
    assert v["drawdown_252"].iloc[299] == 0.0


# ----------------------------------------------------------- price structure
def test_bar_structure_by_hand_and_zero_range():
    bars = bars_from([11.0, 5.0], opens=[10.0, 5.0], high=[12.0, 5.0], low=[9.0, 5.0])
    v = values(
        bars,
        "hl_range_pct",
        "clv",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "body_ratio",
    )
    assert v["hl_range_pct"].iloc[0] == pytest.approx(3 / 11)
    for column in ("clv", "upper_wick_ratio", "lower_wick_ratio", "body_ratio"):
        assert v[column].iloc[0] == pytest.approx(1 / 3)
        assert v[column].iloc[1] == 0.0


def test_body_and_wicks_sum_to_one():
    v = SYN_VALUES
    total = v["body_ratio"] + v["upper_wick_ratio"] + v["lower_wick_ratio"]
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert (v["dist_high_20"].dropna() <= 1e-12).all() and (
        v["dist_low_20"].dropna() >= -1e-12
    ).all()


# -------------------------------------------------------------------- volume
def test_volume_features_with_constant_volume():
    v = values(GROWTH, *DEFAULT_REGISTRY.ids(FeatureGroup.VOLUME))
    t = 120
    assert v["rel_volume_20"].iloc[t] == pytest.approx(1.0)
    assert math.isnan(v["volume_z_20"].iloc[t])
    assert math.isnan(v["price_volume_corr_20"].iloc[t])
    assert (
        math.isnan(v["dollar_volume_z_60"].iloc[t]) is False
    )  # close grows, so traded value varies
    assert v["volume_change_5"].iloc[t] == pytest.approx(0.0)
    assert v["volume_trend_20"].iloc[t] == pytest.approx(0.0, abs=1e-12)
    assert v["obv_flow_20"].iloc[t] == pytest.approx(1.0)


def test_volume_against_numpy():
    v, vol = SYN_VALUES, SYN["volume"].to_numpy()
    t = 300
    lv = np.log1p(vol[t - 19 : t + 1])
    assert v["volume_z_20"].iloc[t] == pytest.approx(
        (lv[-1] - lv.mean()) / lv.std(ddof=1)
    )
    assert v["volume_trend_20"].iloc[t] == pytest.approx(
        np.polyfit(np.arange(20), lv, 1)[0]
    )
    r = v["log_return_1d"].to_numpy()
    dlv = np.diff(np.log1p(vol))
    expected = np.corrcoef(r[t - 19 : t + 1], dlv[t - 20 : t])[0, 1]
    assert v["price_volume_corr_20"].iloc[t] == pytest.approx(expected)


# ---------------------------------------------------------------- statistics
def test_statistics_against_numpy():
    v, r = SYN_VALUES, SYN_VALUES["log_return_1d"].to_numpy()
    t = 330
    w = r[t - 59 : t + 1]
    c = w - w.mean()
    assert v["skew_60"].iloc[t] == pytest.approx(np.mean(c**3) / np.mean(c**2) ** 1.5)
    assert v["kurtosis_60"].iloc[t] == pytest.approx(
        np.mean(c**4) / np.mean(c**2) ** 2 - 3
    )
    assert v["autocorr_1_60"].iloc[t] == pytest.approx(
        np.corrcoef(w, r[t - 60 : t])[0, 1]
    )
    w20 = r[t - 19 : t + 1]
    assert v["up_fraction_20"].iloc[t] == pytest.approx(np.mean(w20 > 0))
    assert v["efficiency_ratio_20"].iloc[t] == pytest.approx(
        abs(w20.sum()) / np.abs(w20).sum()
    )
    assert v["return_to_vol_20"].iloc[t] == pytest.approx(w20.mean() / w20.std(ddof=1))


def test_statistics_on_constant_growth():
    v = values(GROWTH, "up_fraction_20", "efficiency_ratio_20", "skew_60")
    assert v["up_fraction_20"].iloc[100] == 1.0
    assert v["efficiency_ratio_20"].iloc[100] == pytest.approx(1.0)
    assert math.isnan(v["skew_60"].iloc[100])


# -------------------------------------------------------------------- regime
def test_regimes_follow_their_fixed_thresholds():
    v = SYN_VALUES.dropna()
    pct, ratio, dd = v["vol_percentile_252"], v["vol_ratio_5_20"], v["drawdown_252"]
    np.testing.assert_array_equal(
        v["regime_volatility"], np.where(pct > 2 / 3, 1, np.where(pct < 1 / 3, -1, 0))
    )
    np.testing.assert_array_equal(
        v["regime_vol_expansion"],
        np.where(ratio > 1.25, 1, np.where(ratio < 0.8, -1, 0)),
    )
    np.testing.assert_array_equal(v["regime_trend"], np.sign(v["dist_sma_200"]))
    np.testing.assert_array_equal(v["regime_momentum"], np.sign(v["cum_return_60"]))
    expected_dd = np.select([dd > -0.05, dd > -0.10, dd > -0.20], [0, 1, 2], default=3)
    np.testing.assert_array_equal(v["regime_drawdown"], expected_dd)
    np.testing.assert_allclose(
        v["trend_vol_interaction"], v["dist_sma_50"] / v["realized_vol_20"], rtol=1e-12
    )


def test_drawdown_regime_covers_every_bucket():
    path = np.r_[np.full(260, 100.0), 97, 92, 85, 70]
    regime = values(bars_from(path), "regime_drawdown")["regime_drawdown"]
    assert list(regime.iloc[-5:]) == [0, 0, 1, 2, 3]


# ---------------------------------------------------------- catalogue shape
def test_every_feature_declares_complete_metadata():
    for spec in DEFAULT_REGISTRY.specs():
        assert spec.units and spec.definition and spec.description
        if spec.group in (FeatureGroup.VOLUME,):
            assert not spec.visibility.value.startswith("returns")
        assert spec.blind_safe, spec.id
