"""T3 dataset construction: labels, selection rules, folds, scalers, builder, manifest."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from feature_fixtures import synthetic_bars

from src.data.split import Split, SplitConfig, SplitError, assign_splits
from src.datasets import (
    ContextRule,
    DatasetError,
    DatasetSpec,
    FoldError,
    LabelError,
    LabelSpec,
    ManifestError,
    Scaler,
    ScalerError,
    ScalerKind,
    build_dataset,
    fit_scaler,
    get_label,
    inner_cutoff,
    inner_split,
    read_manifest,
    select_examples,
    walk_forward,
    walk_forward_cutoffs,
    write_manifest,
)
from src.datasets.manifest import code_revision
from src.features.canonical import to_canonical

LEARNING = Path(__file__).resolve().parents[1] / "learning"
CONFIG = SplitConfig(
    train_end=date(2011, 12, 31),
    validation_start=date(2012, 2, 1),
    validation_end=date(2012, 12, 31),
    test_start=date(2013, 2, 1),
)
BARS = {"AAA": synthetic_bars(900, seed=1), "BBB": synthetic_bars(900, seed=2)}
DATES = list(BARS["AAA"].index.date)


SPLIT_LABELS = assign_splits(DATES, CONFIG)


def _split_of(t: int) -> Split | None:
    return SPLIT_LABELS[t]


# --------------------------------------------------------------------- labels
def test_labels_by_hand():
    bars = to_canonical(BARS["AAA"])
    executed = get_label("executed_return_v1").values(bars)
    opens = bars["adj_open"].to_numpy()
    assert executed[10] == pytest.approx(np.log(opens[12] / opens[11]))
    assert np.isnan(executed[-2:]).all() and np.isfinite(executed[:-2]).all()
    close = get_label("close_to_close_v1").values(bars)
    assert close[10] == pytest.approx(
        np.log(bars["adj_close"].iloc[11] / bars["adj_close"].iloc[10])
    )
    assert np.isnan(close[-1])
    with pytest.raises(LabelError):
        get_label("nope")
    with pytest.raises(LabelError):
        LabelSpec("x", "1.0.0", "d", "d", 0, lambda b: np.zeros(len(b)))
    bad = LabelSpec("x", "1.0.0", "d", "d", 1, lambda b: np.zeros(3))
    with pytest.raises(LabelError):
        bad.values(bars)


# ------------------------------------------------------------------ selection
@pytest.mark.parametrize("context", list(ContextRule))
def test_every_label_stays_inside_its_partition(context):
    ex = select_examples(DATES, CONFIG, lookback=100, horizon=2, context=context)
    for split, indices in (
        (Split.TRAIN, ex.train),
        (Split.VALIDATION, ex.validation),
        (Split.TEST, ex.test),
    ):
        assert indices
        for t in indices:
            assert all(_split_of(i) is split for i in range(t, t + 3))
            assert t - 100 >= 0
            if context is ContextRule.OWN_PARTITION and split is not Split.TEST:
                assert all(_split_of(i) is split for i in range(t - 100, t))


def test_available_history_keeps_more_validation_dates_with_the_same_labels():
    own = select_examples(DATES, CONFIG, lookback=100, horizon=2)
    avail = select_examples(
        DATES, CONFIG, lookback=100, horizon=2, context=ContextRule.AVAILABLE_HISTORY
    )
    assert set(own.validation) < set(avail.validation)
    assert own.train == avail.train and own.test == avail.test
    first_validation = min(
        t for t, d in enumerate(DATES) if d >= CONFIG.validation_start
    )
    assert avail.validation[0] == first_validation


def test_selection_refuses_invalid_requests():
    with pytest.raises(SplitError):
        select_examples(DATES, CONFIG, lookback=-1, horizon=1)
    with pytest.raises(SplitError):
        select_examples(DATES, CONFIG, lookback=10, horizon=0)
    # A gap of 21 weekdays cannot hold horizon 20 + embargo 5.
    with pytest.raises(SplitError, match="gap"):
        select_examples(DATES, CONFIG, lookback=10, horizon=20)


# ---------------------------------------------------------------------- folds
@pytest.mark.parametrize("context", list(ContextRule))
def test_inner_split_purges_and_embargoes(context):
    ex = select_examples(DATES, CONFIG, lookback=64, horizon=2, context=context)
    cutoff = inner_cutoff([DATES[t] for t in ex.train])
    fold = inner_split(ex, cutoff, embargo=5, context=context)
    first = min(i for i, d in enumerate(DATES) if d >= cutoff)
    assert fold.train and fold.evaluate
    assert all(t + 2 < first - 5 for t in fold.train)
    start = (
        (lambda t: t - 64) if context is ContextRule.OWN_PARTITION else (lambda t: t)
    )
    assert all(start(t) >= first for t in fold.evaluate)
    assert set(fold.train).isdisjoint(fold.evaluate)


def test_walk_forward_is_chronological_purged_and_disjoint():
    ex = select_examples(DATES, CONFIG, lookback=64, horizon=2)
    cuts = walk_forward_cutoffs(
        [DATES[t] for t in ex.train], folds=3, min_train_fraction=0.4
    )
    folds = walk_forward(ex, cuts, embargo=5, context=ContextRule.OWN_PARTITION)
    assert len(folds) == 3 and len(cuts) == 4
    seen: set[int] = set()
    for k, fold in enumerate(folds):
        first = min(i for i, d in enumerate(DATES) if d >= cuts[k])
        assert fold.train and fold.evaluate
        assert max(fold.train) + 2 < first - 5
        assert min(fold.evaluate) - 64 >= first
        if k < 2:
            stop = min(i for i, d in enumerate(DATES) if d >= cuts[k + 1])
            assert max(fold.evaluate) + 2 < stop
        assert seen.isdisjoint(fold.evaluate)
        seen |= set(fold.evaluate)
    assert len(folds[2].train) > len(folds[0].train)


def test_fold_errors():
    with pytest.raises(FoldError):
        inner_cutoff([])
    with pytest.raises(FoldError):
        inner_cutoff([date(2020, 1, 1)], fraction=1.5)
    with pytest.raises(FoldError):
        walk_forward_cutoffs(DATES[:5], folds=3, min_train_fraction=0.5)
    with pytest.raises(FoldError):
        walk_forward_cutoffs(DATES, folds=0, min_train_fraction=0.5)
    with pytest.raises(FoldError):
        walk_forward_cutoffs(DATES, folds=2, min_train_fraction=1.0)


# --------------------------------------------------------------------- scaler
def test_scaler_kinds():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 3.0, (500, 3))
    x[:, 2] = 7.0
    cols = ("a", "b", "c")
    std = fit_scaler("standard", x, cols)
    z = std.transform(x)
    np.testing.assert_allclose(z[:, :2].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z[:, :2].std(axis=0), 1, atol=1e-12)
    assert std.scale[2] == 1.0
    glob = fit_scaler("global_standard", x, cols)
    assert glob.center == (float(x.mean()),) * 3 and glob.scale == (float(x.std()),) * 3
    rob = fit_scaler("robust", x, cols)
    assert rob.clip == 5.0 and np.abs(rob.transform(x * 100)).max() <= 5.0
    none = fit_scaler("none", x, cols)
    np.testing.assert_array_equal(none.transform(x), x)
    seq = fit_scaler("standard", x.reshape(100, 5, 3), cols)
    assert seq.transform(x.reshape(100, 5, 3)).shape == (100, 5, 3)


def test_scaler_serialization_and_errors():
    x = np.random.default_rng(1).normal(size=(50, 2))
    scaler = fit_scaler("robust", x, ("a", "b"))
    restored = Scaler.from_dict(json.loads(json.dumps(scaler.to_dict())), ("a", "b"))
    np.testing.assert_array_equal(restored.transform(x), scaler.transform(x))
    with pytest.raises(ScalerError, match="columns differ"):
        Scaler.from_dict(scaler.to_dict(), ("b", "a"))
    with pytest.raises(ScalerError, match="invalid"):
        Scaler.from_dict({"kind": "standard"}, ("a", "b"))
    with pytest.raises(ScalerError):
        fit_scaler("standard", np.zeros((0, 2)), ("a", "b"))
    with pytest.raises(ScalerError):
        fit_scaler("standard", np.array([[np.nan, 1.0]]), ("a", "b"))
    with pytest.raises(ScalerError):
        fit_scaler("standard", np.zeros((3, 2)), ("a",))
    with pytest.raises(ScalerError, match="expected"):
        scaler.transform(np.zeros((2, 3)))
    with pytest.raises(ScalerError):
        Scaler(ScalerKind.STANDARD, ("a",), (0.0,), (0.0,))
    with pytest.raises(ScalerError):
        Scaler(ScalerKind.STANDARD, ("a",), (0.0,), (1.0,), clip=-1.0)
    with pytest.raises(ScalerError):
        Scaler(ScalerKind.STANDARD, ("a", "b"), (0.0,), (1.0,))


# ------------------------------------------------------------------- builder
def test_spec_validation():
    with pytest.raises(ValueError, match="exactly one"):
        DatasetSpec()
    with pytest.raises(ValueError, match="exactly one"):
        DatasetSpec(features=("log_return_1d",), profile="returns_only_v1")
    with pytest.raises(ValueError, match="sequence"):
        DatasetSpec(profile="returns_only_v1", sequence_length=3)
    with pytest.raises(ValueError):
        DatasetSpec(profile="returns_only_v1", unknown=1)


def test_builder_parts_scaler_and_metadata():
    ds = build_dataset(
        BARS, DatasetSpec(profile="returns_plus_price_structure"), CONFIG
    )
    assert set(ds.parts) == {"train", "inner_train", "inner_validation", "validation"}
    inner = ds.parts["inner_train"]
    np.testing.assert_allclose(inner.x.mean(axis=0), 0, atol=1e-10)
    for part in ds.parts.values():
        assert part.x.shape == (len(part), len(ds.columns))
        assert set(part.index["asset"]) == {"AAA", "BBB"}
        np.testing.assert_array_equal(part.y, (part.label_values > 0).astype(int))
    validation = ds.parts["validation"].index["date"]
    assert (
        validation.min() >= CONFIG.validation_start
        and validation.max() <= CONFIG.validation_end
    )
    assert ds.parts["train"].index["date"].max() <= CONFIG.train_end
    assert ds.manifest["parts"]["validation"]["rows"] == len(ds.parts["validation"])


def test_scaler_never_sees_validation_or_test_rows():
    altered = {k: v.copy() for k, v in BARS.items()}
    for frame in altered.values():
        late = frame.index >= pd.Timestamp(CONFIG.validation_start)
        for column in ("open", "high", "low", "close", "adjusted_close"):
            frame.loc[late, column] *= 3.0
        frame.loc[late, "volume"] *= 10
    spec = DatasetSpec(profile="extended_features", scaler="robust")
    a, b = build_dataset(BARS, spec, CONFIG), build_dataset(altered, spec, CONFIG)
    assert a.scaler == b.scaler
    np.testing.assert_array_equal(a.parts["train"].x, b.parts["train"].x)


def test_sealed_test_period_is_never_read_unless_requested():
    altered = {k: v.copy() for k, v in BARS.items()}
    for frame in altered.values():
        frame.loc[
            frame.index >= pd.Timestamp(CONFIG.test_start), "adjusted_close"
        ] *= 2.0
    spec = DatasetSpec(profile="returns_only_v1")
    assert (
        build_dataset(BARS, spec, CONFIG).fingerprint
        == build_dataset(altered, spec, CONFIG).fingerprint
    )
    with_test = build_dataset(
        BARS, DatasetSpec(profile="returns_only_v1", include_test=True), CONFIG
    )
    assert len(with_test.parts["test"]) > 0
    assert with_test.parts["test"].index["date"].min() >= CONFIG.test_start


def test_sequence_last_step_equals_tabular_row():
    tab = build_dataset(
        BARS,
        DatasetSpec(profile="core_features", min_lookback=255, scaler="none"),
        CONFIG,
    )
    seq = build_dataset(
        BARS,
        DatasetSpec(
            profile="core_features",
            representation="sequence",
            sequence_length=5,
            scaler="none",
        ),
        CONFIG,
    )
    assert seq.lookback == tab.lookback == 255
    np.testing.assert_array_equal(
        seq.parts["validation"].x[:, -1, :], tab.parts["validation"].x
    )
    np.testing.assert_array_equal(
        seq.parts["validation"].x[1:, -2, :][:5], tab.parts["validation"].x[:5]
    )


def test_common_lookback_aligns_profiles_on_the_same_dates():
    long = build_dataset(BARS, DatasetSpec(profile="returns_plus_volatility"), CONFIG)
    short = build_dataset(
        BARS, DatasetSpec(profile="returns_only_v1", min_lookback=long.lookback), CONFIG
    )
    for name in ("train", "inner_train", "inner_validation", "validation"):
        pd.testing.assert_frame_equal(short.parts[name].index, long.parts[name].index)


def test_undefined_rows_are_dropped_and_counted():
    flat = {k: v.copy() for k, v in BARS.items()}
    for frame in flat.values():
        frame.loc[frame.index[100:160], "volume"] = 1000.0
    ds = build_dataset(
        flat, DatasetSpec(features=("returns_window_16", "volume_z_20")), CONFIG
    )
    assert ds.manifest["parts"]["train"]["dropped_undefined"] > 0
    assert np.isfinite(ds.parts["train"].x).all()


def test_builder_errors():
    with pytest.raises(DatasetError, match="no asset"):
        build_dataset({}, DatasetSpec(profile="returns_only_v1"), CONFIG)
    with pytest.raises(DatasetError, match="primary"):
        build_dataset(
            BARS, DatasetSpec(profile="returns_only_v1", primary_asset="ZZZ"), CONFIG
        )
    short = {"AAA": BARS["AAA"].iloc[:60]}
    with pytest.raises(DatasetError, match="no training"):
        build_dataset(short, DatasetSpec(profile="returns_only_v1"), CONFIG)
    # Constant volume during training: volume_z_20 is undefined on every train row.
    quiet = BARS["AAA"].copy()
    quiet.loc[quiet.index <= pd.Timestamp(CONFIG.train_end), "volume"] = 1000.0
    with pytest.raises(DatasetError, match="no train rows"):
        build_dataset(
            {"AAA": quiet},
            DatasetSpec(features=("volume_z_20",), fit_on="train"),
            CONFIG,
        )


# ------------------------------------------------------------------ manifest
def test_manifest_round_trip_and_tamper_detection(tmp_path):
    ds = build_dataset(BARS, DatasetSpec(profile="returns_plus_trend"), CONFIG)
    again = build_dataset(BARS, DatasetSpec(profile="returns_plus_trend"), CONFIG)
    assert ds.fingerprint == again.fingerprint
    assert ds.manifest["features"]["redundancy_warnings"] == []
    assert ds.manifest["selection"]["context"] == "own_partition"
    path = tmp_path / "manifest.json"
    write_manifest(ds.manifest, path)
    assert read_manifest(path)["dataset_fingerprint"] == ds.fingerprint
    tampered = json.loads(path.read_text())
    tampered["parts"]["validation"]["rows"] += 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(ManifestError, match="fingerprint"):
        read_manifest(path)
    path.write_text(json.dumps({**tampered, "schema_version": 99}))
    with pytest.raises(ManifestError, match="schema"):
        read_manifest(path)
    path.write_text("not json")
    with pytest.raises(ManifestError, match="cannot read"):
        read_manifest(path)
    assert isinstance(code_revision(), str)


def test_fingerprint_changes_with_the_context_rule():
    own = build_dataset(BARS, DatasetSpec(profile="returns_only_v1"), CONFIG)
    avail = build_dataset(
        BARS,
        DatasetSpec(profile="returns_only_v1", context="available_history"),
        CONFIG,
    )
    assert own.fingerprint != avail.fingerprint
    assert len(avail.parts["validation"]) > len(own.parts["validation"])


# ---------------------------------------- parity with the team's study pipeline
@pytest.fixture
def study_pipeline(monkeypatch):
    monkeypatch.syspath_prepend(str(LEARNING / "mlp"))
    monkeypatch.syspath_prepend(str(LEARNING / "analysis"))
    import export_comparison
    import protocol
    import step3_normalize

    return export_comparison, protocol, step3_normalize


def test_matches_export_comparison_inputs_labels_and_folds(study_pipeline):
    export_comparison, protocol, step3 = study_pipeline
    spec = DatasetSpec(
        profile="returns_only_v1", scaler="global_standard", primary_asset="AAA"
    )
    ds = build_dataset(BARS, spec, CONFIG)

    prepared = {k: export_comparison.prepare_asset(v, CONFIG) for k, v in BARS.items()}
    primary = prepared["AAA"]
    cutoff = protocol.inner_cutoff(
        [primary["bar_dates"][t + 1] for t in primary["train"]]
    )
    assert ds.manifest["selection"]["inner_cutoff"] == cutoff.isoformat()

    expected_x, expected_r = {}, {}
    for key, a in prepared.items():
        inner_train, inner_val = protocol.inner_split(
            a["train"], a["bar_dates"], cutoff, embargo=CONFIG.embargo
        )
        for part, indices in (
            ("train", a["train"]),
            ("inner_train", inner_train),
            ("inner_validation", inner_val),
            ("validation", a["val"]),
        ):
            # Team indexing: example t' is decided at the close of bar t' + 1.
            rows = ds.parts[part].index
            mine = rows.loc[rows["asset"] == key, "bar"].to_numpy()
            np.testing.assert_array_equal(mine, np.asarray(indices) + 1)
            expected_x.setdefault(part, []).append(
                export_comparison.windows(a["values"], indices)
            )
            expected_r.setdefault(part, []).append(
                protocol.executed_returns(a["opens"], indices)
            )
    for part in ("train", "inner_train", "inner_validation", "validation"):
        np.testing.assert_allclose(
            ds.parts[part].x_raw, np.concatenate(expected_x[part]), rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            ds.parts[part].label_values,
            np.concatenate(expected_r[part]),
            rtol=1e-12,
            atol=1e-15,
        )
    mean, std = step3.fit_normalizer(np.concatenate(expected_x["inner_train"]))
    assert ds.scaler.center[0] == pytest.approx(mean, rel=1e-12)
    assert ds.scaler.scale[0] == pytest.approx(std, rel=1e-12)
