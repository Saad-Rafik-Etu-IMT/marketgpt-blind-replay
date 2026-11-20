"""Redundancy declarations, feature profiles and training-only diagnostics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from feature_fixtures import synthetic_bars

from src.data.split import SplitConfig
from src.features import (
    DEFAULT_REGISTRY,
    PROFILES,
    FeatureEngineError,
    FeatureGroup,
    FeatureProfile,
    FeatureProfileError,
    FeatureRegistry,
    FeatureRegistryError,
    FeatureSpec,
    FeatureSpecError,
    compute,
    get_profile,
    redundancy_report,
    training_rows,
    validate_profiles,
)
from src.features.diagnostics import DiagnosticsError


def _const(name, value=1.0):
    return lambda bars, deps: pd.DataFrame(
        {name: np.full(len(bars), value)}, index=bars.index
    )


def spec(feature_id, **overrides):
    fields = dict(
        id=feature_id,
        version="1.0.0",
        group=FeatureGroup.RETURNS,
        description="Toy.",
        definition="1",
        required_fields=("adj_close",),
        lookback=0,
        compute=_const(feature_id),
    )
    fields.update(overrides)
    return FeatureSpec(**fields)


# --------------------------------------------------------- redundancy fields
@pytest.mark.parametrize(
    "overrides",
    [
        {"redundant_with": ("b",)},  # reason missing
        {"redundancy_reason": "orphan reason"},  # no reference
        {"redundant_with": ("a",), "redundancy_reason": "self"},
        {"redundant_with": ("b", "b"), "redundancy_reason": "dup"},
        {"determined_by": ("b",), "redundancy_reason": "needs two"},
    ],
)
def test_redundancy_declarations_are_validated(overrides):
    with pytest.raises(FeatureSpecError):
        spec("a", **overrides)


def test_registry_reports_pairwise_and_joint_redundancy():
    registry = FeatureRegistry(
        [
            spec("a"),
            spec("b"),
            spec("c", redundant_with=("a",), redundancy_reason="c = 2a"),
            spec("d", determined_by=("a", "b"), redundancy_reason="d = a - b"),
        ]
    )
    registry.validate()
    assert registry.redundancies(["a", "b"]) == []
    assert registry.redundancies(["a", "d"]) == []  # d needs both a and b
    assert registry.redundancies(["c", "a"]) == ["c is redundant with a: c = 2a"]
    assert registry.redundancies(["a", "b", "d"]) == [
        "d is determined by a, b: d = a - b"
    ]
    matrix = compute(synthetic_bars(5), ["a", "c"], registry)
    assert matrix.warnings == ("c is redundant with a: c = 2a",)

    broken = FeatureRegistry(
        [spec("x", redundant_with=("ghost",), redundancy_reason="?")]
    )
    with pytest.raises(FeatureRegistryError, match="unknown"):
        broken.validate()


def test_catalogue_declares_the_known_redundancies():
    found = DEFAULT_REGISTRY.redundancies(DEFAULT_REGISTRY.ids())
    text = "\n".join(found)
    for expected in (
        "mean_return_5 is redundant with cum_return_5",
        "roc_20 is redundant with cum_return_20",
        "intraday_return is determined by log_return_1d, overnight_return",
        "macd_hist is determined by macd_line, macd_signal",
        "body_ratio is determined by upper_wick_ratio, lower_wick_ratio",
        "dist_ema_12 is redundant with dist_sma_10",
        "returns_window_16 is redundant with returns_window_64",
        "regime_volatility is redundant with vol_percentile_252",
    ):
        assert expected in text


def test_engine_bounds_allow_rounding_only():
    def nearly(value):
        return FeatureRegistry(
            [spec("r", compute=_const("r", value), hard_bounds=(0.0, 1.0))]
        )

    compute(synthetic_bars(5), ["r"], nearly(1.0 + 1e-12))
    with pytest.raises(FeatureEngineError, match="hard bounds"):
        compute(synthetic_bars(5), ["r"], nearly(1.0 + 1e-6))


def test_start_independence_detects_an_infinite_memory_feature():
    """A classic recursive EMA depends on the first bar: the T2 property catches it."""

    def classic(bars, deps):
        ema = bars["adj_close"].ewm(span=10, adjust=False).mean()
        return pd.DataFrame(
            {"classic_ema": bars["adj_close"] / ema - 1}, index=bars.index
        )

    registry = FeatureRegistry([spec("classic_ema", compute=classic, lookback=39)])
    bars = synthetic_bars(200)
    full = compute(bars, ["classic_ema"], registry).values.iloc[30 + 39 :]
    late = compute(bars.iloc[30:], ["classic_ema"], registry).values.iloc[39:]
    assert not np.allclose(late.to_numpy(), full.to_numpy(), rtol=1e-12, atol=1e-15)


# ------------------------------------------------------------------ profiles
def test_all_profiles_are_valid_and_start_with_the_baseline():
    validate_profiles()
    assert get_profile("returns_only_v1").features == ("returns_window_64",)
    described = get_profile("extended_features").describe()
    assert described["column_count"] > 64 and len(described["fingerprint"]) == 64
    for profile in PROFILES.values():
        if profile.available:
            assert DEFAULT_REGISTRY.redundancies(profile.features) == []
            assert len(set(profile.features)) == len(profile.features)


def test_group_profiles_add_only_their_own_group():
    expected = {
        "returns_plus_trend": FeatureGroup.TREND,
        "returns_plus_volatility": FeatureGroup.VOLATILITY,
        "returns_plus_price_structure": FeatureGroup.PRICE_STRUCTURE,
        "returns_plus_volume": FeatureGroup.VOLUME,
        "returns_plus_statistics": FeatureGroup.STATISTICS,
    }
    for profile_id, group in expected.items():
        added = get_profile(profile_id).features[1:]
        assert added and all(DEFAULT_REGISTRY.get(f).group is group for f in added)


def test_unavailable_and_unknown_profiles():
    assert PROFILES["returns_plus_market_context"].describe()["available"] is False
    with pytest.raises(FeatureProfileError, match="unavailable"):
        get_profile("returns_plus_market_context")
    with pytest.raises(FeatureProfileError, match="unknown"):
        get_profile("nope")


@pytest.mark.parametrize(
    ("profile", "match"),
    [
        (FeatureProfile("p", "1.0.0", "d", ("cum_return_5",)), "baseline"),
        (
            FeatureProfile(
                "p", "1.0.0", "d", ("returns_window_64", "cum_return_5", "roc_5")
            ),
            "redundant",
        ),
        (
            FeatureProfile(
                "p", "1.0.0", "d", ("x",), available=False, unavailable_reason="r"
            ),
            "unavailable",
        ),
        (FeatureProfile("p", "1.0.0", "d", (), available=False), "unavailable"),
    ],
)
def test_profile_rules_are_enforced(profile, match):
    with pytest.raises(FeatureProfileError, match=match):
        validate_profiles({"p": profile})


def test_profile_rules_refuse_unsafe_features():
    from src.features import Visibility

    registry = FeatureRegistry(
        [
            *(DEFAULT_REGISTRY.get(i) for i in ("log_return_1d", "returns_window_64")),
            spec("day", visibility=Visibility.DATED),
        ]
    )
    profile = FeatureProfile("p", "1.0.0", "d", ("returns_window_64", "day"))
    with pytest.raises(FeatureProfileError, match="blind-safe"):
        validate_profiles({"p": profile}, registry)


# --------------------------------------------------------------- diagnostics
SPLIT = SplitConfig(
    train_end=date(2010, 9, 30),
    validation_start=date(2010, 11, 1),
    validation_end=date(2010, 12, 31),
    test_start=date(2011, 2, 1),
)


def test_training_rows_never_include_validation_or_test():
    values = compute(synthetic_bars(320), ["log_return_1d", "cum_return_5"]).values
    rows = training_rows(values, SPLIT)
    assert len(rows) > 0 and rows.index.max() <= pd.Timestamp(SPLIT.train_end)
    assert rows.notna().all().all()


def test_redundancy_report_finds_clones_and_counts_components():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    rows = pd.DataFrame(
        {"a": a, "a_scaled": 3 * a + 1, "a_cubed": a**3, "b": b, "flat": np.ones(500)}
    )
    report = redundancy_report(rows, threshold=0.95)
    names = {(p.first, p.second) for p in report.pairs}
    assert ("a", "a_scaled") in names and ("a", "a_cubed") in names
    assert report.clusters == (("a", "a_scaled", "a_cubed"),)
    assert report.constant_columns == ("flat",)
    assert report.components_for_variance["99%"] <= 3
    pearson = redundancy_report(rows, threshold=0.95, method="pearson")
    assert ("a", "a_cubed") not in {(p.first, p.second) for p in pearson.pairs}
    assert report.to_dict()["clusters"] == [["a", "a_scaled", "a_cubed"]]


@pytest.mark.parametrize(
    ("rows", "kwargs"),
    [
        (pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]}), {}),
        (pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 1.0, 1.0]}), {}),
        (
            pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]}),
            {"method": "kendall"},
        ),
        (
            pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]}),
            {"threshold": 0.0},
        ),
    ],
)
def test_redundancy_report_refuses_degenerate_input(rows, kwargs):
    with pytest.raises(DiagnosticsError):
        redundancy_report(rows, **kwargs)
