"""T4 targets: quantities, losses and gradients, decoding, metrics, builder integration."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from feature_fixtures import synthetic_bars

from src.data.split import SplitConfig
from src.datasets import DatasetError, DatasetSpec, build_dataset
from src.features import compute
from src.features.canonical import to_canonical
from src.targets import (
    NEUTRAL_BAND_K,
    TARGETS,
    TargetError,
    TargetKind,
    get_target,
    softmax,
)
from src.targets import metrics as m
from src.targets.quantities import (
    close_cumulative,
    executed_cumulative,
    future_volatility,
)
from src.targets.targets import ExpectedReturn

LEARNING = Path(__file__).resolve().parents[1] / "learning"
# Wide gaps so 20-session labels fit; the narrow config mimics split.json's 21-bar test gap.
WIDE = SplitConfig(
    train_end=date(2011, 12, 31),
    validation_start=date(2012, 3, 1),
    validation_end=date(2012, 12, 31),
    test_start=date(2013, 3, 1),
)
NARROW_TEST_GAP = WIDE.model_copy(update={"test_start": date(2013, 2, 1)})
BARS = {"AAA": synthetic_bars(1000, seed=4), "BBB": synthetic_bars(1000, seed=5)}
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- quantities
def test_quantities_by_hand():
    bars = to_canonical(BARS["AAA"])
    opens, close = bars["adj_open"].to_numpy(), bars["adj_close"].to_numpy()
    ex5 = executed_cumulative(bars, 5)
    assert ex5[10] == pytest.approx(math.log(opens[16] / opens[11]))
    assert np.isnan(ex5[-6:]).all() and np.isfinite(ex5[:-6]).all()
    cc = close_cumulative(bars, 3)
    assert cc[10] == pytest.approx(math.log(close[13] / close[10]))
    vol = future_volatility(bars, 5)
    r = np.diff(np.log(close))
    assert vol[10] == pytest.approx(np.std(r[10:15], ddof=1))
    assert np.isnan(vol[-5:]).all()
    flat = to_canonical(
        BARS["AAA"].assign(
            adjusted_close=50.0, close=50.0, open=50.0, high=50.0, low=50.0
        )
    )
    assert np.isnan(future_volatility(flat, 5)).all()
    with pytest.raises(ValueError):
        future_volatility(bars, 1)


# ------------------------------------------------------------------ registry
def test_registry_describes_every_target():
    assert {
        "direction_binary_v1",
        "direction_3class_v1",
        "expected_return_h1",
        "volatility_h5",
        "quantiles_h1",
        "multi_horizon_v1",
    } <= set(TARGETS)
    for target in TARGETS.values():
        d = target.describe()
        assert d["output_dim"] == len(d["output_names"]) >= 1
        assert d["interpretation"] and d["definition"] and d["loss"] and d["metrics"]
        assert d["min_bootstrap_block"] >= d["horizon_bars"]
    dev = {t.id for t in TARGETS.values() if t.development_only}
    assert dev == {"expected_return_h20", "volatility_h20", "multi_horizon_v1"}
    with pytest.raises(TargetError):
        get_target("buy_sell_hold")
    with pytest.raises(TargetError):
        ExpectedReturn((1,), loss="hinge")


# ----------------------------------------------- losses, gradients, decoding
def _labels(target, n=40):
    raw = RNG.normal(0, 0.01, (n, len(target.horizons)))
    if target.kind is TargetKind.VOLATILITY:
        raw = np.abs(raw) + 0.002
    sigma = np.full(n, 0.01)
    state = target.fit(raw, sigma)
    return target.encode(raw, sigma, state), state


@pytest.mark.parametrize("target_id", sorted(TARGETS))
def test_gradient_matches_finite_differences(target_id):
    target = get_target(target_id)
    y, state = _labels(target)
    z = RNG.normal(0, 1, (len(y), target.output_dim))
    _, grad = target.loss(z, y, state)
    eps = 1e-6
    for i, j in [(0, 0), (3, target.output_dim - 1), (7, 0)]:
        up, down = z.copy(), z.copy()
        up[i, j] += eps
        down[i, j] -= eps
        numeric = (target.loss(up, y, state)[0] - target.loss(down, y, state)[0]) / (
            2 * eps
        )
        assert grad[i, j] == pytest.approx(numeric, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("target_id", sorted(TARGETS))
def test_output_shape_and_finiteness_are_checked(target_id):
    target = get_target(target_id)
    y, state = _labels(target)
    with pytest.raises(TargetError):
        target.loss(np.zeros((len(y), target.output_dim + 1)), y, state)
    with pytest.raises(TargetError):
        target.decode(np.full((2, target.output_dim), np.nan), state)


def test_classification_probabilities_and_loss():
    target = get_target("direction_3class_v1")
    z = RNG.normal(0, 3, (50, 3))
    decoded = target.decode(z, {})
    total = decoded["p_down"] + decoded["p_neutral"] + decoded["p_up"]
    np.testing.assert_allclose(total, 1.0)
    y = RNG.integers(0, 3, 50)
    assert target.loss(z, y, {})[0] == pytest.approx(m.log_loss(softmax(z), y))


def test_three_class_band_uses_sigma_at_decision_time():
    target = get_target("direction_3class_v1")
    raw = np.array([[-0.004], [-0.001], [0.0], [0.002], [0.006]])
    sigma = np.full(5, 0.01)  # band = 0.25 * 0.01 = 0.0025
    np.testing.assert_array_equal(target.encode(raw, sigma, {}), [0, 1, 1, 1, 2])
    assert NEUTRAL_BAND_K == 0.25
    np.testing.assert_array_equal(target.encode(raw, sigma * 4, {}), [1, 1, 1, 1, 1])


def test_quantiles_never_cross_and_decode_to_return_units():
    target = get_target("quantiles_h1")
    y, state = _labels(target, 200)
    z = RNG.normal(0, 50, (200, 3))
    d = target.decode(z, state)
    assert (d["q10"] <= d["q50"]).all() and (d["q50"] <= d["q90"]).all()
    report = target.evaluate(
        z,
        y,
        np.asarray(state["mean"]) + y[:, None] * state["std"],
        np.full(200, 0.01),
        state,
    )
    assert report["crossing_rate"] == 0.0


def test_volatility_decodes_positive_and_compares_with_the_naive_forecast():
    target = get_target("volatility_h5")
    raw = np.abs(RNG.normal(0, 0.01, (100, 1))) + 0.001
    sigma = raw[:, 0] * 1.1
    state = target.fit(raw, sigma)
    y = target.encode(raw, sigma, state)
    assert (target.decode(RNG.normal(0, 30, (100, 1)), state)["volatility"] > 0).all()
    perfect = target.evaluate(y, y, raw, sigma, state)
    assert perfect["mae"] == pytest.approx(0, abs=1e-12) and perfect[
        "qlike"
    ] == pytest.approx(0, abs=1e-12)
    assert perfect["naive_mae"] > 0 and perfect["mae_ratio_vs_naive"] == pytest.approx(
        0, abs=1e-9
    )


def test_expected_return_round_trip_and_metrics():
    target = get_target("multi_horizon_v1")
    raw = RNG.normal(0.0005, 0.01, (300, 3))
    state = target.fit(raw, np.zeros(300))
    y = target.encode(raw, None, state)
    np.testing.assert_allclose(
        np.column_stack(list(target.decode(y, state).values())), raw
    )
    report = target.evaluate(y, y, raw, None, state)
    assert report["h5"]["mae_bp"] == pytest.approx(0, abs=1e-9)
    assert report["h1"]["r2_vs_train_mean"] == pytest.approx(1.0)
    mse = get_target("expected_return_h1_mse")
    e = np.array([[3.0]])
    assert mse.loss(e, np.zeros((1, 1)), {})[0] == pytest.approx(4.5)
    assert get_target("expected_return_h1").loss(e, np.zeros((1, 1)), {})[
        0
    ] == pytest.approx(2.5)


def test_fit_refuses_unusable_training_labels():
    with pytest.raises(TargetError, match="constant"):
        get_target("expected_return_h1").fit(np.zeros((10, 1)), np.zeros(10))
    with pytest.raises(TargetError, match="two"):
        get_target("expected_return_h1").fit(np.zeros((1, 1)), np.zeros(1))
    with pytest.raises(TargetError, match="no training"):
        get_target("direction_binary_v1").fit(np.zeros((0, 1)), np.zeros(0))


# ------------------------------------------------------------------- metrics
def test_metrics_match_the_team_protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(LEARNING / "mlp"))
    import protocol

    p = RNG.uniform(0, 1, 500)
    p[:50] = 0.5  # ties
    y = (RNG.uniform(0, 1, 500) < 0.55).astype(int)
    pred = (p >= 0.5).astype(int)
    assert m.macro_f1(pred, y) == pytest.approx(protocol.macro_f1(pred, y))
    assert m.auc(p, y) == pytest.approx(protocol.auc(p, y))
    assert m.log_loss(np.column_stack([1 - p, p]), y) == pytest.approx(
        protocol.log_loss(p, y)
    )


def test_metric_edge_cases():
    assert math.isnan(m.auc(np.array([0.2, 0.4]), np.array([1, 1])))
    assert math.isnan(m.rank_correlation(np.ones(5), np.arange(5.0)))
    assert math.isnan(m.directional_accuracy(np.zeros(3), np.ones(3)))
    assert math.isnan(m.r2_out_of_sample(np.ones(3), np.zeros(3), 1.0))
    assert math.isnan(m.balanced_accuracy(np.array([]), np.array([]), 2))
    assert math.isnan(m.expected_calibration_error(np.array([]), np.array([])))
    assert math.isnan(m.accuracy(np.array([]), np.array([])))
    bins = m.reliability_bins(np.array([1.0, 0.95, 0.05]), np.array([1, 0, 0]))
    assert bins[-1]["count"] == 2 and bins[-1]["high"] == 1.0
    assert m.brier(np.array([[0.0, 1.0]]), np.array([1])) == 0.0
    assert m.pinball(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert (
        m.coverage(np.array([0.0, 2.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        == 0.5
    )
    assert m.confusion(np.array([0, 1, 1]), np.array([0, 0, 1]), 2) == [[1, 1], [0, 1]]


# ------------------------------------------------------- builder integration
def test_binary_target_reproduces_the_t3_label():
    old = build_dataset(BARS, DatasetSpec(profile="returns_only_v1"), WIDE)
    new = build_dataset(
        BARS, DatasetSpec(profile="returns_only_v1", target="direction_binary_v1"), WIDE
    )
    for name in ("train", "inner_train", "inner_validation", "validation"):
        pd.testing.assert_frame_equal(old.parts[name].index, new.parts[name].index)
        np.testing.assert_array_equal(old.parts[name].y, new.parts[name].y)
    assert (
        new.manifest["target"]["id"] == "direction_binary_v1"
        and new.manifest["label"] is None
    )
    assert old.manifest["target"] is None


def test_sigma_is_the_point_in_time_realized_volatility():
    ds = build_dataset(
        BARS,
        DatasetSpec(features=("returns_window_16",), target="direction_3class_v1"),
        WIDE,
    )
    assert ds.lookback == 20
    part = ds.parts["validation"]
    feature = compute(
        BARS["AAA"].loc[: str(WIDE.validation_end)], ["realized_vol_20"]
    ).values
    rows = part.index["asset"] == "AAA"
    expected = feature.iloc[part.index.loc[rows, "bar"].to_numpy(), 0].to_numpy()
    np.testing.assert_array_equal(part.sigma[rows.to_numpy()], expected)
    band = NEUTRAL_BAND_K * part.sigma
    r = part.targets[:, 0]
    np.testing.assert_array_equal(
        part.y, np.where(r < -band, 0, np.where(r > band, 2, 1))
    )


def test_fitted_target_state_ignores_validation_rows():
    altered = {k: v.copy() for k, v in BARS.items()}
    for frame in altered.values():
        late = frame.index >= pd.Timestamp(WIDE.validation_start)
        for column in ("open", "high", "low", "close", "adjusted_close"):
            frame.loc[late, column] *= np.linspace(1, 2, late.sum())
    spec = DatasetSpec(profile="returns_only_v1", target="quantiles_h1")
    assert (
        build_dataset(BARS, spec, WIDE).target_state
        == build_dataset(altered, spec, WIDE).target_state
    )


def test_long_horizon_targets_fit_wide_gaps_and_refuse_the_narrow_test_gap():
    ds = build_dataset(
        BARS, DatasetSpec(profile="returns_only_v1", target="multi_horizon_v1"), WIDE
    )
    assert ds.parts["validation"].y.shape[1] == 3
    last_label = ds.parts["validation"].index["bar"].max() + 21
    assert BARS["AAA"].index[last_label].date() <= WIDE.validation_end
    with pytest.raises(DatasetError, match="development-only"):
        build_dataset(
            BARS,
            DatasetSpec(
                profile="returns_only_v1", target="multi_horizon_v1", include_test=True
            ),
            NARROW_TEST_GAP,
        )
    ok = build_dataset(
        BARS,
        DatasetSpec(
            profile="returns_only_v1", target="expected_return_h5", include_test=True
        ),
        NARROW_TEST_GAP,
    )
    assert len(ok.parts["test"]) > 0


def test_label_and_target_cannot_both_be_set():
    with pytest.raises(ValueError, match="not both"):
        DatasetSpec(
            profile="returns_only_v1",
            label="close_to_close_v1",
            target="direction_binary_v1",
        )


@pytest.mark.parametrize("target_id", sorted(TARGETS))
def test_every_target_builds_and_evaluates(target_id):
    target = get_target(target_id)
    ds = build_dataset(
        BARS, DatasetSpec(profile="returns_only_v1", target=target_id), WIDE
    )
    part = ds.parts["validation"]
    assert len(part) > 0 and np.isfinite(part.targets).all()
    z = np.zeros((len(part), target.output_dim))
    report = target.evaluate(z, part.y, part.targets, part.sigma, ds.target_state)
    assert report["count"] == len(part)
    assert ds.manifest["target"]["fitted_state"] == ds.target_state
