"""T5 configurable MLP: gradients, parity with W1 and deep_mlp, training, checkpoints."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from feature_fixtures import synthetic_bars

from src.ai.mlp import (
    MLP,
    PRESETS,
    ArchitectureConfig,
    CheckpointError,
    Head,
    TrainConfig,
    TrainingData,
    TrainingDivergedError,
    evaluate_loss,
    load,
    preset,
    save,
    train,
)
from src.data.split import SplitConfig
from src.datasets import DatasetSpec, build_dataset
from src.targets import get_target
from src.targets.metrics import auc

LEARNING = Path(__file__).resolve().parents[1] / "learning"
RNG = np.random.default_rng(3)


def _labels(targets, n, signal=None):
    """Labels per target; ``signal`` (n,) drives every target when given."""
    rng = np.random.default_rng(1)
    base = signal if signal is not None else rng.normal(0, 1, n)
    raw = 0.01 * base[:, None]
    out, states = {}, {}
    for t in targets:
        target = get_target(t)
        r = np.repeat(raw, len(target.horizons), axis=1)
        if target.kind.value == "volatility":
            r = np.abs(r) + 0.003
        sigma = np.full(n, 0.01)
        states[t] = target.fit(r, sigma)
        out[t] = target.encode(r, sigma, states[t])
    return out, states


def _total_loss(model, x, y, states, training, seed=0):
    outputs, cache = model.forward(
        x, training=training, rng=np.random.default_rng(seed)
    )
    total, d = 0.0, {}
    for head in model.config.heads:
        value, grad = get_target(head.target).loss(
            outputs[head.target], y[head.target], states[head.target]
        )
        total += head.weight * value
        d[head.target] = head.weight * grad
    return total, model.backward(cache, d)


# -------------------------------------------------------------------- config
@pytest.mark.parametrize(
    "fields",
    [
        {"hidden": (0,)},
        {"hidden": (8, 4), "dropout": (0.1,)},
        {"dropout": 1.0},
        {"heads": ()},
        {
            "heads": (
                Head(target="direction_binary_v1"),
                Head(target="direction_binary_v1"),
            )
        },
        {"hidden": (), "input_skip": 0.5},
        {"unknown": 1},
    ],
)
def test_invalid_architectures(fields):
    with pytest.raises(ValueError):
        ArchitectureConfig(**fields)
    with pytest.raises(ValueError):
        Head(target="buy_sell_hold")


def test_learning_rate_schedules():
    cos = TrainConfig(
        learning_rate=1.0, schedule="cosine", schedule_floor=0.15, epochs=40
    )
    assert cos.learning_rate_at(1) == pytest.approx(1.0)
    assert cos.learning_rate_at(21) == pytest.approx(0.15 + 0.85 * 0.5)
    warm = TrainConfig(learning_rate=1.0, warmup_epochs=3)
    assert [warm.learning_rate_at(e) for e in (1, 3, 4)] == pytest.approx(
        [0.25, 0.75, 1.0]
    )
    assert set(PRESETS) == {"w1_baseline", "small", "medium", "large", "deep_v1"}
    with pytest.raises(ValueError):
        preset("huge")


# ----------------------------------------------------------------- gradients
CONFIGS = {
    "relu": ArchitectureConfig(hidden=(6, 5)),
    "gelu_batchnorm": ArchitectureConfig(
        hidden=(6, 5), activation="gelu", norm="batch"
    ),
    "tanh_layernorm_residual": ArchitectureConfig(
        hidden=(6, 6, 6), activation="tanh", norm="layer", residual=True
    ),
    "skip_lecun": ArchitectureConfig(
        hidden=(6, 5), input_skip=0.35, output_init="lecun"
    ),
    "multi_head": ArchitectureConfig(
        hidden=(6, 5),
        heads=(
            Head(target="direction_3class_v1"),
            Head(target="expected_return_h1", weight=0.5),
            Head(target="quantiles_h1", weight=2.0),
            Head(target="volatility_h5"),
        ),
    ),
    "linear_no_hidden": ArchitectureConfig(hidden=()),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_backward_matches_finite_differences(name):
    config = CONFIGS[name]
    model = MLP(4, config, seed=2)
    x = RNG.normal(0, 1, (12, 4))
    y, states = _labels([h.target for h in config.heads], 12)
    _, grads = _total_loss(model, x, y, states, training=True)
    assert set(grads) == {
        k for k in model.params if not k.endswith(("running_mean", "running_var"))
    }
    eps = 1e-6
    for key, grad in grads.items():
        for index in [(0,) * grad.ndim, tuple(s - 1 for s in grad.shape)]:
            original = model.params[key][index]
            model.params[key][index] = original + eps
            up = _total_loss(model, x, y, states, training=True)[0]
            model.params[key][index] = original - eps
            down = _total_loss(model, x, y, states, training=True)[0]
            model.params[key][index] = original
            assert grad[index] == pytest.approx(
                (up - down) / (2 * eps), rel=1e-4, abs=1e-8
            ), (key, index)


def test_running_batchnorm_statistics_backward():
    model = MLP(4, CONFIGS["gelu_batchnorm"], seed=1)
    x = RNG.normal(0, 1, (10, 4))
    y, states = _labels(["direction_binary_v1"], 10)
    _, grads = _total_loss(model, x, y, states, training=False)
    eps = 1e-6
    model.params["hidden.0.W"][0, 0] += eps
    up = _total_loss(model, x, y, states, training=False)[0]
    model.params["hidden.0.W"][0, 0] -= 2 * eps
    down = _total_loss(model, x, y, states, training=False)[0]
    assert grads["hidden.0.W"][0, 0] == pytest.approx((up - down) / (2 * eps), rel=1e-4)


# ------------------------------------------------------------------- parity
@pytest.fixture
def learning(monkeypatch):
    monkeypatch.syspath_prepend(str(LEARNING / "mlp"))
    monkeypatch.syspath_prepend(str(LEARNING / "analysis"))
    import deep_mlp
    import step8_train

    return step8_train, deep_mlp


def test_w1_baseline_reproduces_the_learning_sprint_training(learning):
    step8, _ = learning
    x = RNG.normal(0, 1, (300, 64))
    y = (x[:, -1] + RNG.normal(0, 1, 300) > 0).astype(int)
    xv = RNG.normal(0, 1, (80, 64))
    yv = (xv[:, -1] > 0).astype(int)
    reference, ref_history = step8.train(x, y, xv, yv, epochs=6, seed=0)

    architecture, config = preset("w1_baseline")
    model = MLP(64, architecture, seed=0)
    result = train(
        model,
        TrainingData(x, {"direction_binary_v1": y}, xv, {"direction_binary_v1": yv}),
        config.model_copy(update={"epochs": 6}),
    )
    mapping = {
        "W1": "hidden.0.W",
        "b1": "hidden.0.b",
        "W2": "head.direction_binary_v1.W",
        "b2": "head.direction_binary_v1.b",
    }
    for theirs, ours in mapping.items():
        np.testing.assert_allclose(
            result.params[ours], reference[theirs], rtol=0, atol=1e-12
        )
    np.testing.assert_allclose(
        [h["train_loss"] for h in result.history],
        [h["train_loss"] for h in ref_history],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [h["select_loss"] for h in result.history],
        [h["val_loss"] for h in ref_history],
        atol=1e-12,
    )
    assert result.best_epoch == min(ref_history, key=lambda h: h["val_loss"])["epoch"]


def test_deep_v1_matches_deep_mlp_forward_and_backward(learning):
    _, deep = learning
    n_in = 20
    theirs = deep.init_deep_mlp(n_in=n_in, seed=5)
    architecture, _ = preset("deep_v1")
    model = MLP(n_in, architecture, seed=5)
    names = {
        "hidden.0.W": "W1",
        "hidden.0.b": "b1",
        "hidden.0.gamma": "gamma1",
        "hidden.0.beta": "beta1",
        "hidden.1.W": "W2",
        "hidden.1.b": "b2",
        "hidden.1.gamma": "gamma2",
        "hidden.1.beta": "beta2",
        "hidden.2.W": "W3",
        "hidden.2.b": "b3",
        "hidden.2.gamma": "gamma3",
        "hidden.2.beta": "beta3",
        "skip.W": "W_skip",
        "head.direction_binary_v1.W": "W4",
        "head.direction_binary_v1.b": "b4",
        "hidden.0.running_mean": "rm1",
        "hidden.0.running_var": "rv1",
    }
    for ours, key in names.items():
        np.testing.assert_array_equal(
            model.params[ours], theirs[key]
        )  # identical initialization

    x = RNG.normal(0, 1, (32, n_in))
    y = RNG.integers(0, 2, 32)
    probs, cache = deep.deep_mlp_forward(
        x, theirs, training=True, dropout=0.25, rng=np.random.default_rng(9)
    )
    expected = deep.deep_mlp_backward(cache, np.eye(2)[y], np.ones(32), theirs)
    outputs, our_cache = model.forward(x, training=True, rng=np.random.default_rng(9))
    _, dz = get_target("direction_binary_v1").loss(
        outputs["direction_binary_v1"], y, {}
    )
    grads = model.backward(our_cache, {"direction_binary_v1": dz})
    np.testing.assert_allclose(
        outputs["direction_binary_v1"], cache["logits"], atol=1e-12
    )
    for ours, key in names.items():
        if key in expected:
            np.testing.assert_allclose(
                grads[ours], expected[key], rtol=1e-10, atol=1e-13, err_msg=key
            )
    np.testing.assert_allclose(
        model.params["hidden.2.running_var"], theirs["rv3"], atol=1e-15
    )
    eval_probs, _ = deep.deep_mlp_forward(x, theirs, training=False)
    np.testing.assert_allclose(
        get_target("direction_binary_v1").decode(
            model.predict(x)["direction_binary_v1"], {}
        )["p_up"],
        eval_probs[:, 1],
        atol=1e-12,
    )


# ----------------------------------------------------------------- training
def _signal_data(n=1200, informative=True):
    x = RNG.normal(0, 1, (n, 10))
    signal = x[:, 0] - 0.5 * x[:, 1] if informative else RNG.normal(0, 1, n)
    labels = (signal + 0.3 * RNG.normal(0, 1, n) > 0).astype(int)
    cut = int(n * 0.75)
    return TrainingData(
        x[:cut],
        {"direction_binary_v1": labels[:cut]},
        x[cut:],
        {"direction_binary_v1": labels[cut:]},
    )


def test_learns_a_known_signal_and_finds_nothing_in_noise():
    architecture, config = preset("small")
    data = _signal_data()
    model = MLP(10, architecture, seed=0)
    train(model, data, config.model_copy(update={"epochs": 15}))
    p_up = get_target("direction_binary_v1").decode(
        model.predict(data.x_select)["direction_binary_v1"], {}
    )["p_up"]
    assert (
        auc(p_up, data.y_select["direction_binary_v1"]) > 0.8
    )  # Bayes-optimal is about 0.9

    noise = _signal_data(informative=False)
    model = MLP(10, architecture, seed=0)
    train(model, noise, config.model_copy(update={"epochs": 15}))
    p_up = get_target("direction_binary_v1").decode(
        model.predict(noise.x_select)["direction_binary_v1"], {}
    )["p_up"]
    assert abs(auc(p_up, noise.y_select["direction_binary_v1"]) - 0.5) < 0.1


def test_training_is_deterministic_and_seed_dependent():
    data = _signal_data(400)
    architecture, config = preset("deep_v1")
    config = config.model_copy(update={"epochs": 3})
    a = train(MLP(10, architecture, seed=1), data, config)
    b = train(MLP(10, architecture, seed=1), data, config)
    c = train(
        MLP(10, architecture, seed=1), data, config.model_copy(update={"seed": 2})
    )
    for key in a.params:
        np.testing.assert_array_equal(a.params[key], b.params[key])
    assert any(not np.array_equal(a.params[k], c.params[k]) for k in a.params)


def test_early_stopping_restores_the_best_epoch():
    data = _signal_data(600, informative=False)
    architecture, _ = preset("large")
    config = TrainConfig(optimizer="adamw", learning_rate=1e-2, epochs=60, patience=3)
    model = MLP(10, architecture, seed=0)
    result = train(model, data, config)
    assert result.stopped_early and len(result.history) < 60
    best = min(result.history, key=lambda h: h["select_loss"])
    assert best["epoch"] == result.best_epoch and best["selected"]
    current = evaluate_loss(model, data.x_select, data.y_select, {})["total"]
    assert current == pytest.approx(best["select_loss"])


def test_divergence_is_detected():
    data = _signal_data(200)
    data = TrainingData(
        data.x_train * 1e150, data.y_train, data.x_select, data.y_select
    )
    with pytest.raises(TrainingDivergedError) as info:
        train(
            MLP(10, ArchitectureConfig(hidden=(8,)), seed=0),
            data,
            TrainConfig(optimizer="sgd", learning_rate=10.0, epochs=3),
        )
    assert "non-finite" in str(info.value)


def test_training_input_errors():
    data = _signal_data(100)
    model = MLP(
        10,
        ArchitectureConfig(hidden=(4,), heads=(Head(target="expected_return_h1"),)),
        seed=0,
    )
    with pytest.raises(ValueError, match="no labels"):
        train(model, data, TrainConfig(epochs=1))
    empty = TrainingData(data.x_train[:0], data.y_train, data.x_select, data.y_select)
    with pytest.raises(ValueError, match="non-empty"):
        train(model, empty, TrainConfig(epochs=1))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 7)))
    with pytest.raises(ValueError):
        MLP(0, ArchitectureConfig())


# ---------------------------------------------------------------- multi-task
CONFIG = SplitConfig(
    train_end=date(2011, 12, 31),
    validation_start=date(2012, 3, 1),
    validation_end=date(2012, 12, 31),
    test_start=date(2013, 3, 1),
)
BARS = {"AAA": synthetic_bars(1000, seed=8), "BBB": synthetic_bars(1000, seed=9)}


def test_multi_task_training_records_head_losses_and_gradient_conflicts():
    targets = ("direction_binary_v1", "expected_return_h5", "volatility_h5")
    common = dict(
        profile="returns_plus_price_structure", min_horizon=6, min_lookback=64
    )
    datasets = [
        build_dataset(BARS, DatasetSpec(target=t, **common), CONFIG) for t in targets
    ]
    data = TrainingData.from_datasets(datasets)
    architecture = ArchitectureConfig(
        hidden=(16, 8), heads=tuple(Head(target=t) for t in targets)
    )
    result = train(
        MLP(data.x_train.shape[1], architecture, seed=0), data, TrainConfig(epochs=3)
    )
    record = result.history[0]
    assert set(record["select_loss_per_head"]) == set(targets)
    assert record["select_loss"] == pytest.approx(
        sum(record["select_loss_per_head"].values())
    )
    assert len(record["gradient_cosine"]) == 3
    assert all(-1 <= v <= 1 for v in record["gradient_cosine"].values())

    misaligned = build_dataset(
        BARS,
        DatasetSpec(target="volatility_h5", profile="returns_plus_price_structure"),
        CONFIG,
    )
    with pytest.raises(ValueError, match="disagree"):
        TrainingData.from_datasets([datasets[0], misaligned])
    with pytest.raises(ValueError, match="one dataset per target"):
        TrainingData.from_datasets([datasets[0], datasets[0]])
    with pytest.raises(ValueError):
        TrainingData.from_datasets([])


# --------------------------------------------------------------- checkpoints
def test_checkpoint_round_trip_and_refusals(tmp_path):
    ds = build_dataset(
        BARS,
        DatasetSpec(profile="returns_plus_price_structure", target="quantiles_h1"),
        CONFIG,
    )
    data = TrainingData.from_datasets([ds])
    architecture, config = preset("medium", ("quantiles_h1",))
    model = MLP(len(ds.columns), architecture, seed=0)
    result = train(model, data, config.model_copy(update={"epochs": 2}))
    path = tmp_path / "models" / "quantiles"
    manifest = save(
        path,
        model,
        config,
        dataset=ds,
        target_states=data.target_states,
        history=result.history,
        best_epoch=result.best_epoch,
    )
    assert manifest["features"]["columns"] == list(ds.columns)
    assert manifest["dataset_fingerprint"] == ds.fingerprint

    loaded = load(path, expected_columns=ds.columns)
    x_raw = ds.parts["validation"].x_raw
    np.testing.assert_array_equal(
        loaded.predict_raw_features(x_raw)["quantiles_h1"],
        model.predict(ds.parts["validation"].x)["quantiles_h1"],
    )
    decoded = loaded.decode_raw_features(x_raw)["quantiles_h1"]
    assert (decoded["q10"] <= decoded["q90"]).all()

    with pytest.raises(CheckpointError, match="columns"):
        load(path, expected_columns=tuple(reversed(ds.columns)))
    weights = dict(np.load(path.with_suffix(".npz")))
    weights["hidden.0.b"] = weights["hidden.0.b"] + 1.0
    np.savez(path.with_suffix(".npz"), **weights)
    with pytest.raises(CheckpointError, match="hash"):
        load(path)
    np.savez(path.with_suffix(".npz"), evil=np.array([object()], dtype=object))
    with pytest.raises(CheckpointError, match="cannot read weights"):
        load(path)
    stored = json.loads(path.with_suffix(".json").read_text())
    path.with_suffix(".json").write_text(json.dumps({**stored, "schema_version": 9}))
    with pytest.raises(CheckpointError, match="version"):
        load(path)
    with pytest.raises(CheckpointError, match="cannot read manifest"):
        load(tmp_path / "missing")


def test_checkpoint_refuses_shapes_that_do_not_match_the_architecture(tmp_path):
    ds = build_dataset(BARS, DatasetSpec(profile="returns_only_v1"), CONFIG)
    architecture, config = preset("small")
    model = MLP(64, architecture, seed=0)
    path = tmp_path / "small"
    save(path, model, config, dataset=ds, target_states={}, history=[], best_epoch=0)
    stored = json.loads(path.with_suffix(".json").read_text())
    stored["architecture"]["hidden"] = [16]
    path.with_suffix(".json").write_text(json.dumps(stored))
    with pytest.raises(CheckpointError, match="shapes"):
        load(path)
