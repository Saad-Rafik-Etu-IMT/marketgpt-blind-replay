"""Feature specification, registry, canonical frame and engine contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from feature_fixtures import synthetic_bars

from src.features import (
    BASELINE_FEATURES,
    DEFAULT_REGISTRY,
    FeatureEngineError,
    FeatureGroup,
    FeatureRegistry,
    FeatureRegistryError,
    FeatureSpec,
    FeatureSpecError,
    MissingPolicy,
    Visibility,
    compute,
    vector_at,
)
from src.features.canonical import CanonicalFrameError, to_canonical


def _const(name: str, value: float = 1.0):
    def fn(bars, deps):
        return pd.DataFrame({name: np.full(len(bars), value)}, index=bars.index)

    return fn


def spec(feature_id: str = "toy", **overrides) -> FeatureSpec:
    fields = dict(
        id=feature_id,
        version="1.0.0",
        group=FeatureGroup.RETURNS,
        description="Toy feature.",
        definition="1",
        required_fields=("adj_close",),
        lookback=0,
        compute=_const(feature_id),
    )
    fields.update(overrides)
    return FeatureSpec(**fields)


# --------------------------------------------------------------------- spec
@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "Bad-Id"},
        {"version": "1.0"},
        {"description": " "},
        {"definition": ""},
        {"lookback": -1},
        {"lookback": True},
        {"lookback": 2.0},
        {"required_fields": ("price_level",)},
        {"required_fields": (), "dependencies": ()},
        {"dependencies": ("toy",)},
        {"columns": ("a", "a")},
        {"columns": ("Bad Column",)},
        {"expected_range": (1.0, 1.0)},
        {"hard_bounds": (2.0, -2.0)},
        {"compute": "not callable"},
    ],
)
def test_invalid_specs_are_refused(overrides):
    with pytest.raises(FeatureSpecError):
        spec(**overrides)


def test_spec_defaults_and_description_are_json_ready():
    s = spec(visibility=Visibility.DATED)
    assert s.columns == ("toy",) and s.width == 1
    assert not s.blind_safe
    described = s.describe()
    assert described["visibility"] == "dated" and described["blind_safe"] is False
    assert "compute" not in described


# ----------------------------------------------------------------- registry
def test_duplicates_and_column_clashes_are_refused():
    registry = FeatureRegistry([spec("a")])
    with pytest.raises(FeatureRegistryError, match="already registered"):
        registry.register(spec("a"))
    with pytest.raises(FeatureRegistryError, match="already used"):
        registry.register(spec("b", columns=("a",)))


def test_resolution_orders_dependencies_and_detects_problems():
    registry = FeatureRegistry(
        [
            spec("base", lookback=1),
            spec("mid", required_fields=(), dependencies=("base",), lookback=2),
            spec("top", required_fields=(), dependencies=("mid", "base"), lookback=3),
        ]
    )
    assert [s.id for s in registry.resolve(["top"])] == ["base", "mid", "top"]
    assert registry.lookback(["mid"]) == 2
    with pytest.raises(FeatureRegistryError, match="unknown"):
        registry.resolve(["nope"])
    with pytest.raises(FeatureRegistryError, match="more than once"):
        registry.resolve(["base", "base"])

    cyclic = FeatureRegistry(
        [
            spec("x", required_fields=(), dependencies=("y",)),
            spec("y", required_fields=(), dependencies=("x",)),
        ]
    )
    with pytest.raises(FeatureRegistryError, match="cycle"):
        cyclic.validate()

    short = FeatureRegistry(
        [
            spec("long", lookback=5),
            spec("short", required_fields=(), dependencies=("long",)),
        ]
    )
    with pytest.raises(FeatureRegistryError, match="shorter"):
        short.resolve(["short"])


def test_fingerprint_tracks_versions_order_and_dependencies():
    def build(base_version="1.0.0"):
        return FeatureRegistry(
            [
                spec("base", version=base_version, lookback=1),
                spec("mid", required_fields=(), dependencies=("base",), lookback=1),
                spec("other"),
            ]
        )

    reference = build().fingerprint(["mid", "other"])
    assert reference == build().fingerprint(["mid", "other"])
    assert reference != build().fingerprint(["other", "mid"])
    assert reference != build("1.0.1").fingerprint(["mid", "other"])


def test_default_catalogue_is_valid_and_keeps_the_baseline():
    DEFAULT_REGISTRY.validate()
    assert BASELINE_FEATURES == ("returns_window_64",)
    baseline = DEFAULT_REGISTRY.get("returns_window_64")
    assert baseline.width == 64 and baseline.lookback == 64 and baseline.blind_safe
    assert (
        DEFAULT_REGISTRY.columns(["returns_window_64"])[0] == "returns_window_64.lag63"
    )
    assert DEFAULT_REGISTRY.ids(FeatureGroup.RETURNS) == DEFAULT_REGISTRY.ids()
    assert DEFAULT_REGISTRY.specs(FeatureGroup.CALENDAR) == []
    assert "returns_window_64" in DEFAULT_REGISTRY and len(DEFAULT_REGISTRY) >= 12


# ---------------------------------------------------------------- canonical
def test_canonical_frame_derives_adjusted_ohlc():
    bars = synthetic_bars(70)
    out = to_canonical(bars.reset_index())
    factor = bars["adjusted_close"] / bars["close"]
    np.testing.assert_allclose(out["adj_open"], bars["open"] * factor)
    np.testing.assert_allclose(out["adj_close"], bars["adjusted_close"])
    assert isinstance(out.index, pd.DatetimeIndex) and out.index.name == "date"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda f: f.drop(columns="adjusted_close"),
        lambda f: f.iloc[::-1],
        lambda f: pd.concat([f, f.iloc[[-1]]]),
        lambda f: f.assign(close=f["close"].where(f.index != f.index[3], -1.0)),
        lambda f: f.assign(
            adjusted_close=f["adjusted_close"].where(f.index != f.index[3], np.nan)
        ),
        lambda f: f.assign(volume=f["volume"].where(f.index != f.index[3], -5.0)),
        lambda f: f.assign(high=f["low"] * 0.5),
        lambda f: f.iloc[:0],
        lambda f: f.set_axis(f.index.tz_localize("UTC")),
        lambda f: f.set_axis([object()] * len(f)),
    ],
)
def test_canonical_frame_refuses_bad_bars(corrupt):
    with pytest.raises(CanonicalFrameError):
        to_canonical(corrupt(synthetic_bars(10)))


# ------------------------------------------------------------------- engine
def _engine_registry(*specs: FeatureSpec) -> FeatureRegistry:
    return FeatureRegistry(specs)


def test_engine_masks_warm_up_and_only_passes_declared_fields():
    seen: list[list[str]] = []

    def fn(bars, deps):
        seen.append(list(bars.columns))
        return pd.DataFrame({"toy": bars["adj_close"]}, index=bars.index)

    registry = _engine_registry(spec(lookback=3, compute=fn))
    matrix = compute(synthetic_bars(10), ["toy"], registry)
    assert seen == [["adj_close"]]
    assert matrix.values["toy"].iloc[:3].isna().all()
    assert matrix.values["toy"].iloc[3:].notna().all()
    assert matrix.ready.sum() == 7 and matrix.lookback == 3


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (lambda b, d: b["adj_close"], "DataFrame"),
        (lambda b, d: pd.DataFrame({"toy": b["adj_close"].to_numpy()}), "index"),
        (
            lambda b, d: pd.DataFrame({"wrong": b["adj_close"]}, index=b.index),
            "columns",
        ),
        (_const("toy", np.inf), "infinite"),
        (_const("toy", np.nan), "undefined"),
        (_const("toy", 50.0), "hard bounds"),
    ],
)
def test_engine_refuses_contract_violations(fn, match):
    registry = _engine_registry(spec(compute=fn, hard_bounds=(-10.0, 10.0)))
    with pytest.raises(FeatureEngineError, match=match):
        compute(synthetic_bars(10), ["toy"], registry)


def test_allow_nan_policy_keeps_undefined_values_visible():
    registry = _engine_registry(
        spec(compute=_const("toy", np.nan), missing_policy=MissingPolicy.ALLOW_NAN)
    )
    matrix = compute(synthetic_bars(10), ["toy"], registry)
    assert matrix.values["toy"].isna().all() and not matrix.ready.any()


def test_blind_mode_refuses_unsafe_features_and_dependencies():
    registry = _engine_registry(
        spec("dated", visibility=Visibility.DATED),
        spec("child", required_fields=(), dependencies=("dated",)),
        spec("safe"),
    )
    bars = synthetic_bars(10)
    assert list(compute(bars, ["safe"], registry, blind=True).columns) == ["safe"]
    for feature in ("dated", "child"):
        with pytest.raises(FeatureEngineError, match="blind"):
            compute(bars, [feature], registry, blind=True)


def test_engine_request_and_position_errors():
    bars = synthetic_bars(10)
    with pytest.raises(FeatureEngineError, match="no feature"):
        compute(bars, [])
    with pytest.raises(FeatureEngineError, match="position"):
        vector_at(bars, ["log_return_1d"], 10)


def test_value_fingerprint_is_stable_and_sensitive():
    bars = synthetic_bars(120)
    first = compute(bars, ["returns_window_16", "cum_return_5"])
    assert (
        first.fingerprint()
        == compute(bars, ["returns_window_16", "cum_return_5"]).fingerprint()
    )
    changed = bars.copy()
    changed.iloc[-1, changed.columns.get_loc("adjusted_close")] *= 1.0001
    changed.iloc[-1, changed.columns.get_loc("close")] *= 1.0001
    changed.iloc[-1, changed.columns.get_loc("high")] *= 1.0001
    assert (
        first.fingerprint()
        != compute(changed, ["returns_window_16", "cum_return_5"]).fingerprint()
    )
    assert first.catalogue_fingerprint == DEFAULT_REGISTRY.fingerprint(
        ["returns_window_16", "cum_return_5"]
    )
