"""Training protocol from the 2026-11-20 audit: labels, inner split, policy, export."""

from datetime import date
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.split import SplitConfig

MLP = Path(__file__).resolve().parents[1] / "learning" / "mlp"


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.syspath_prepend(str(MLP))
    return import_module("protocol"), import_module("deep_mlp")


def frame(start="2014-01-01", end="2016-12-30", seed=41):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    r = rng.normal(0, 0.008, len(dates)) + 0.002 * np.sin(np.arange(len(dates)) / 10)
    closes = 100 * np.exp(np.cumsum(r))
    opens = closes * np.exp(rng.normal(0, 0.002, len(dates)))
    return pd.DataFrame(
        dict(
            open=opens,
            close=closes,
            adjusted_close=closes,
            volume=1000,
            high=np.maximum(opens, closes) * 1.001,
            low=np.minimum(opens, closes) * 0.999,
        ),
        index=dates,
    )


SMALL = SplitConfig(
    train_end=date(2016, 6, 30),
    validation_start=date(2016, 8, 1),
    validation_end=date(2016, 12, 30),
    test_start=date(2017, 2, 1),
)


def test_executed_label_is_next_open_to_following_open(mods):
    protocol, _ = mods
    opens = np.array([10.0, 11, 12, 13, 14, 15])
    # example t is decided at the close of bar t+1, filled at bar t+2's open
    np.testing.assert_allclose(
        protocol.executed_returns(opens, [0, 1]), np.log([13 / 12, 14 / 13])
    )


def test_inner_split_keeps_windows_and_embargo_on_their_side(mods):
    protocol, _ = mods
    bar_dates = list(pd.bdate_range("2010-01-01", periods=400).date)
    train = tuple(range(63, 396))
    cutoff = bar_dates[300]
    inner_train, inner_val = protocol.inner_split(train, bar_dates, cutoff, embargo=5)
    assert inner_train and inner_val
    assert max(t + 3 for t in inner_train) < 300 - 5
    assert min(t - 63 for t in inner_val) >= 300
    assert not set(inner_train) & set(inner_val)


def test_hold_policy_respects_minimum_holding_period(mods):
    protocol, _ = mods
    policy = protocol.HoldPolicy(flat_below=0.5, min_hold=3)
    positions = [policy.step(p) for p in [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]]
    assert positions == [1, 1, 1, 1, 0, 0, 0, 0]
    always = protocol.HoldPolicy(flat_below=0.0)
    assert all(always.step(p) == 1 for p in [0.0, 0.2, 0.9])


def test_policy_stays_long_without_signal_and_goes_flat_with_it(mods):
    protocol, _ = mods
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 2000)
    noise = rng.uniform(0.4, 0.6, 2000)
    assert protocol.fit_policy([(noise, r)])["flat_below"] == 0.0
    informative = np.where(r < -0.005, 0.3, 0.7)
    assert protocol.fit_policy([(informative, r)], min_hold=1)["flat_below"] > 0.3


def test_temperature_softens_overconfident_noise(mods):
    _, deep = mods
    rng = np.random.default_rng(1)
    diff = rng.normal(0, 2.0, 4000)  # confident logits
    y = rng.integers(0, 2, 4000)  # unrelated labels
    assert deep.fit_temperature(diff, y) > 4.0


def test_training_retains_lowest_selection_loss_epoch(mods):
    _, deep = mods
    rng = np.random.default_rng(2)
    xt, xs = rng.normal(0, 0.01, (200, 64)), rng.normal(0, 0.01, (60, 64))
    yt, ys = rng.integers(0, 2, 200), rng.integers(0, 2, 60)
    params, history, _ = deep.train_deep_mlp(xt, yt, xs, ys, seed=0, epochs=4)
    best = min(history, key=lambda h: h["val_loss"])
    assert [h["epoch"] for h in history if h["selected"]] == [best["epoch"]]
    assert float(params["temperature"]) > 0


def test_pooled_export_end_to_end(mods, tmp_path):
    export_comparison = import_module("export_comparison")
    from src.eval.execution import load_benchmark_profile

    report = export_comparison.export(
        frame(),
        "synthetic",
        0,
        tmp_path / "MLP-reference-0.npz",
        epochs=1,
        research=True,
        config=SMALL,
        profile=load_benchmark_profile(),
        name="A",
        pool={"A": frame(), "B": frame(seed=7)},
    )
    protocol = report["protocol"]
    assert protocol["assets"] == ["A", "B"]
    assert protocol["innerValidationStart"] < str(SMALL.train_end)
    assert report["pooled"]["count"] > len(report["cases"])
    # ``next`` is the executed label; the close-to-close return is kept beside it
    assert all("nextClose" in c for c in report["cases"])
    assert sum(c["selected"] for c in report["curves"]) == 1
    # The model's targets never switch faster than the minimum holding period
    targets = [c["actions"][0]["target"] for c in report["cases"]]
    runs, run = [], 1
    for a, b in zip(targets, targets[1:], strict=False):
        if a == b:
            run += 1
        else:
            runs.append(run)
            run = 1
    assert all(r >= protocol["decisionRule"]["min_hold"] for r in runs[1:])


def test_study_defaults_and_pooled_validation(client):
    from src.api.routes_studies import StudyConfig

    assert StudyConfig().seeds == [0, 1, 2, 3, 4]
    response = client.post("/api/studies", json={"source": "synthetic", "pooled": True})
    assert response.status_code == 422


def test_cuda_request_fails_fast_without_a_usable_gpu(mods):
    torch = pytest.importorskip("torch")
    resolve = import_module("transformer_reference").resolve_device
    assert resolve("cpu").type == "cpu"
    with pytest.raises(ValueError):
        resolve("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve("cuda")


def test_auto_device_falls_back_to_cpu_without_a_usable_gpu(mods, monkeypatch):
    torch = pytest.importorskip("torch")
    resolve = import_module("transformer_reference").resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve("cuda")


def test_gpu_arch_check_accepts_older_minor_kernels(mods):
    runs_on = import_module("transformer_reference").build_runs_on
    cu118 = ["sm_37", "sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert runs_on(cu118, 5, 2)  # Tesla M60/M40 use the sm_50 kernels
    assert not runs_on(["sm_80", "sm_90"], 5, 2)
    assert runs_on(["compute_70"], 8, 6)  # PTX is JIT-compiled for newer cards
    assert not runs_on(["compute_90"], 5, 2)
