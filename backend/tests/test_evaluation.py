"""T7 scientific evaluation: statistics, baselines, ablation, experiment evaluation, API."""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import routes_experiments
from src.evaluation import (
    UNAVAILABLE,
    EvaluationError,
    baseline_outputs,
    block_bootstrap_mean,
    drift,
    evaluate_experiment,
    paired_difference,
    psi,
    read_evaluation,
    row_losses,
    verdict,
)
from src.evaluation.evaluate import ablation, trading_simulation
from src.experiments import ExperimentSpec, run_folder, write_spec
from src.main import create_app
from src.services.study_artifacts import read_json
from src.targets import get_target
from src.targets import metrics as m

RNG = np.random.default_rng(0)
DATES = np.array([date(2015, 1, 1) + timedelta(days=i) for i in range(400)])


# --------------------------------------------------------------------- stats
def test_block_bootstrap_covers_the_true_mean():
    hits = 0
    for seed in range(200):
        values = np.random.default_rng(seed).normal(0.0, 1.0, 300)
        low, high = block_bootstrap_mean(
            values, DATES[:300], block=10, n_boot=300, seed=seed
        )
        hits += low <= 0 <= high
    assert 0.88 <= hits / 200 <= 0.99


def test_bootstrap_keeps_dates_together():
    values = RNG.normal(0, 1, 200)
    one = block_bootstrap_mean(values, DATES[:200], block=10, n_boot=200)
    two = block_bootstrap_mean(
        np.r_[values, values], np.r_[DATES[:200], DATES[:200]], block=10, n_boot=200
    )
    np.testing.assert_allclose(one, two)
    assert np.isnan(block_bootstrap_mean(np.zeros(0), np.zeros(0), block=10)[0])


def test_verdicts_and_paired_difference():
    assert verdict(-0.2, -0.1) == "better" and verdict(0.1, 0.2) == "worse"
    assert (
        verdict(-0.1, 0.1) == "no measurable difference"
        and verdict(np.nan, 1) == "undefined"
    )
    better = paired_difference(
        np.zeros(300), np.ones(300) + RNG.normal(0, 0.1, 300), DATES[:300], block=10
    )
    assert (
        better["verdict"] == "better"
        and better["rows"] == 300
        and better["dates"] == 300
    )
    noise = paired_difference(
        RNG.normal(0, 1, 300), RNG.normal(0, 1, 300), DATES[:300], block=10
    )
    assert noise["verdict"] in {"no measurable difference", "better", "worse"}


def test_psi_and_drift():
    ref = RNG.normal(0, 1, 2000)
    assert psi(ref, RNG.normal(0, 1, 2000)) < 0.05 < psi(ref, RNG.normal(1.5, 1, 2000))
    assert np.isnan(psi(np.ones(100), np.ones(100)))
    x_ref = np.column_stack([ref, RNG.normal(0, 1, 2000), np.ones(2000)])
    x_cur = np.column_stack(
        [RNG.normal(0, 1, 500), RNG.normal(2, 1, 500), np.ones(500)]
    )
    report = drift(x_ref, x_cur, ("a", "b", "flat"))
    assert (
        report[0]["column"] == "b" and report[0]["standardized_mean_difference"] > 1.5
    )
    assert (
        next(d for d in report if d["column"] == "flat")["standardized_mean_difference"]
        is None
    )


def _state_and_labels(target, raw, sigma):
    state = target.fit(raw, sigma)
    return state, target.encode(raw, sigma, state)


def test_row_losses_match_the_target_metrics():
    n = 200
    sigma = np.abs(RNG.normal(0.01, 0.002, n))
    raw = RNG.normal(0, 0.01, (n, 1))
    binary = get_target("direction_binary_v1")
    state, y = _state_and_labels(binary, raw, sigma)
    z = RNG.normal(0, 1, (n, 2))
    assert row_losses(binary, z, y, raw, state).mean() == pytest.approx(
        binary.loss(z, y, state)[0]
    )
    regression = get_target("expected_return_h1")
    state, y = _state_and_labels(regression, raw, sigma)
    z = RNG.normal(0, 1, (n, 1))
    expected = np.abs(raw[:, 0] - regression.decode(z, state)["return_h1"]).mean()
    assert row_losses(regression, z, y, raw, state).mean() == pytest.approx(expected)
    volatility = get_target("volatility_h5")
    raw_vol = np.abs(raw) + 0.002
    state, y = _state_and_labels(volatility, raw_vol, sigma)
    expected = m.qlike(raw_vol[:, 0], volatility.decode(z, state)["volatility"])
    assert row_losses(volatility, z, y, raw_vol, state).mean() == pytest.approx(
        expected
    )
    quantiles = get_target("quantiles_h1")
    state, y = _state_and_labels(quantiles, raw, sigma)
    z = RNG.normal(0, 1, (n, 3))
    d = quantiles.decode(z, state)
    expected = np.mean(
        [m.pinball(raw[:, 0], d[f"q{q}"], q / 100) for q in (10, 50, 90)]
    )
    assert row_losses(quantiles, z, y, raw, state).mean() == pytest.approx(expected)


# ----------------------------------------------------------------- baselines
class Part(SimpleNamespace):
    def __len__(self):
        return len(self.x)


def _parts(target, n=1500, signal=True):
    x = RNG.normal(0, 1, (n, 5))
    sigma = np.abs(0.01 + 0.003 * x[:, 1])
    base = 0.01 * x[:, 0] if signal else 0
    raw = (base + 0.01 * RNG.normal(0, 1, n))[:, None]
    if target.kind.value == "volatility":
        raw = sigma[:, None] * np.exp(0.1 * RNG.normal(0, 1, (n, 1)))
    state = target.fit(raw[:900], sigma[:900])
    y = target.encode(raw, sigma, state)
    cut = [slice(0, 900), slice(900, 1200), slice(1200, n)]
    return state, [Part(x=x[c], y=y[c], targets=raw[c], sigma=sigma[c]) for c in cut]


def test_classification_baselines():
    target = get_target("direction_binary_v1")
    state, (tr, se, ev) = _parts(target)
    outputs, info = baseline_outputs(target, state, tr, se, ev)
    assert set(outputs) == {"class_frequency", "logistic_regression"}
    shares = state["train_class_shares"]
    assert row_losses(
        target, outputs["class_frequency"], ev.y, ev.targets, state
    ).mean() == pytest.approx(m.log_loss(np.tile(shares, (len(ev.y), 1)), ev.y))
    p = target.decode(outputs["logistic_regression"], state)["p_up"]
    assert m.auc(p, ev.y) > 0.65 and info["logistic_regression"]["weight_decay"] in (
        0.0,
        1e-3,
        1e-2,
        1e-1,
    )
    assert info["unavailable"] == UNAVAILABLE


def test_regression_and_volatility_baselines():
    target = get_target("expected_return_h1")
    state, (tr, se, ev) = _parts(target)
    outputs, info = baseline_outputs(target, state, tr, se, ev)
    assert np.all(outputs["train_mean"] == 0)
    np.testing.assert_allclose(
        target.decode(outputs["zero_return"], state)["return_h1"], 0, atol=1e-15
    )
    ridge = target.decode(outputs["ridge"], state)["return_h1"]
    assert (
        np.corrcoef(ridge, ev.targets[:, 0])[0, 1] > 0.5 and info["ridge"]["alpha"] > 0
    )
    vol = get_target("volatility_h5")
    state, (tr, se, ev) = _parts(vol)
    outputs, _ = baseline_outputs(vol, state, tr, se, ev)
    np.testing.assert_allclose(
        vol.decode(outputs["naive_realized_vol_20"], state)["volatility"], ev.sigma
    )


def test_quantile_baselines():
    target = get_target("quantiles_h1")
    state, (tr, se, ev) = _parts(target, signal=False)
    outputs, _ = baseline_outputs(target, state, tr, se, ev)
    d = target.decode(outputs["train_quantiles"], state)
    np.testing.assert_allclose(d["q10"], np.quantile(tr.targets[:, 0], 0.1), atol=1e-9)
    n = target.decode(outputs["normal_scaled_by_vol_20"], state)
    np.testing.assert_allclose(
        n["q90"] - n["q50"], 1.2815515655446004 * ev.sigma, rtol=1e-6
    )
    assert (
        target.decode(outputs["linear_quantile_regression"], state)["q10"]
        <= n["q90"] + 1
    ).all()


# ------------------------------------------------------------------ ablation
def _losses(rows, seeds, gap, index):
    return {
        "rows": rows,
        "seeds": seeds,
        "gap": gap,
        "dates": index["date"].to_numpy(),
        "block": 10,
        "index": index,
    }


def test_ablation_categories_are_known_in_advance():
    n = 400
    index = pd.DataFrame({"asset": "A", "date": DATES[:n], "bar": np.arange(n)})
    ref_rows = 0.69 + RNG.normal(0, 0.05, n)
    spec = SimpleNamespace(
        feature_profiles=("returns_only_v1", "signal", "noise", "mixed")
    )

    def seeds(delta):
        return {s: ref_rows + delta + RNG.normal(0, 0.01, n) for s in range(3)}

    mixed_seeds = {0: ref_rows - 0.05, 1: ref_rows + 0.05, 2: ref_rows - 0.01}
    losses = {
        ("returns_only_v1", "small", "t"): _losses(ref_rows, seeds(0), 0.0, index),
        ("signal", "small", "t"): _losses(ref_rows - 0.05, seeds(-0.05), 0.0, index),
        ("noise", "small", "t"): _losses(
            ref_rows + RNG.normal(0, 0.05, n), seeds(0), -0.05, index
        ),
        ("mixed", "small", "t"): _losses(
            ref_rows + RNG.normal(0, 0.05, n), mixed_seeds, 0.0, index
        ),
    }
    report = ablation(spec, losses, n_boot=300)
    found = {f["feature_profile"]: f for f in report["findings"]}
    assert found["signal"]["category"] == "improves"
    assert found["noise"]["category"] == "no measurable improvement"
    assert any("overfitting" in flag for flag in found["noise"]["flags"])
    assert any("unstable" in flag for flag in found["mixed"]["flags"])
    assert "signal (t, small)" in report["summary"]

    nothing = {k: v for k, v in losses.items() if k[0] != "signal"}
    assert report["reference"] == "returns_only_v1"
    assert ablation(spec, nothing, n_boot=300)["summary"].startswith(
        "No feature profile"
    )
    shifted = dict(losses)
    shifted[("noise", "small", "t")] = _losses(
        ref_rows, seeds(0), 0.0, index.iloc[::-1].reset_index(drop=True)
    )
    with pytest.raises(EvaluationError, match="identical rows"):
        ablation(spec, shifted, n_boot=50)


def test_trading_simulation_by_hand():
    index = pd.DataFrame({"asset": ["A"] * 4, "date": DATES[:4], "bar": range(4)})
    part = Part(
        x=np.zeros((4, 1)),
        targets=np.array([[0.01], [-0.02], [0.03], [0.01]]),
        index=index,
    )
    result = trading_simulation(np.array([0.6, 0.4, 0.7, 0.7]), part, {}, cost=0.001)
    model = result["model_ensemble"]
    assert model["total_log_return_equal_weight"] == pytest.approx(
        0.01 + 0.03 + 0.01 - 0.001 * 3
    )
    assert model["exposure"] == 0.75 and model[
        "turnover_per_decision"
    ] == pytest.approx(3 / 4)
    assert result["always_long"]["total_log_return_equal_weight"] == pytest.approx(
        0.03 - 0.001
    )
    assert (
        result["model_ensemble"]["max_drawdown_log"] <= 0
        and "not the replay engine" in result["note"]
    )


# ------------------------------------------------------- experiment evaluation
SPEC = dict(
    name="eval",
    feature_profiles=("returns_only_v1", "returns_plus_volatility"),
    presets=("small",),
    targets=("direction_binary_v1", "volatility_h5", "quantiles_h1"),
    epochs=2,
)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    folder = data / "experiments" / str(uuid4())
    write_spec(folder, ExperimentSpec(**SPEC))
    assert run_folder(folder, data)["state"] == "complete"
    return data, folder


def test_evaluate_experiment_end_to_end(finished):
    data, folder = finished
    report = evaluate_experiment(folder, data, n_boot=200)
    assert len(report["results"]) == 6
    first = next(r for r in report["results"] if r["target"] == "direction_binary_v1")
    assert set(first["baselines"]) == {"class_frequency", "logistic_regression"}
    for baseline in first["baselines"].values():
        assert baseline["comparison"]["verdict"] in {
            "better",
            "worse",
            "no measurable difference",
        }
    assert (
        first["calibration"]["reliability_up"]
        and "trading" in first
        and first["importance"]
    )
    assert set(first["regimes"]) == {"volatility_tercile", "trend_60d"}
    assert (
        sum(v["rows"] for v in first["regimes"]["volatility_tercile"].values())
        == first["rows"]
    )
    vol = next(r for r in report["results"] if r["target"] == "volatility_h5")
    assert "naive_realized_vol_20" in vol["baselines"]
    assert (
        report["ablation"]["reference"] == "returns_only_v1"
        and len(report["ablation"]["findings"]) == 3
    )
    assert set(report["drift"]) == set(SPEC["feature_profiles"])
    assert (
        read_evaluation(folder, "inner_validation")["spec_fingerprint"]
        == report["spec_fingerprint"]
    )
    assert not (data / "experiments" / "validation_reads.json").exists()


def test_validation_evaluation_is_logged(finished):
    data, folder = finished
    evaluate_experiment(folder, data, "validation", n_boot=50)
    reads = read_json(data / "experiments" / "validation_reads.json")["reads"]
    assert reads[-1]["run"] == "evaluation" and reads[-1]["experiment"] == folder.name


def test_evaluation_refusals(finished, tmp_path):
    data, folder = finished
    with pytest.raises(EvaluationError, match="part"):
        evaluate_experiment(folder, data, "test")
    with pytest.raises(EvaluationError, match="part"):
        read_evaluation(folder, "test")
    with pytest.raises(FileNotFoundError):
        read_evaluation(tmp_path, "validation")
    queued = tmp_path / "experiments" / "q"
    write_spec(queued, ExperimentSpec(**SPEC))
    with pytest.raises(EvaluationError, match="not finished"):
        evaluate_experiment(queued, tmp_path)
    copy = tmp_path / "experiments" / "copy"
    write_spec(
        copy, ExperimentSpec(**{**SPEC, "feature_profiles": ("returns_only_v1",)})
    )
    run_folder(copy, tmp_path)
    report_path = next((copy / "runs").glob("*.json"))
    stored = json.loads(report_path.read_text())
    stored["dataset_fingerprints"]["direction_binary_v1"] = "changed"
    report_path.write_text(json.dumps(stored))
    with pytest.raises(EvaluationError, match="dataset changed"):
        evaluate_experiment(copy, tmp_path)
    for path in (copy / "runs").glob("*.json"):
        path.write_text(json.dumps({**json.loads(path.read_text()), "state": "failed"}))
    with pytest.raises(EvaluationError, match="no completed run"):
        evaluate_experiment(copy, tmp_path)


def test_api_evaluate_and_read(finished, monkeypatch):
    data, folder = finished
    monkeypatch.setattr(routes_experiments, "data_dir", lambda: data)
    monkeypatch.setattr(routes_experiments, "worker", None)
    client = TestClient(create_app())
    response = client.post(f"/api/experiments/{folder.name}/evaluate")
    assert response.status_code == 200 and response.json()["results"] == 6
    assert (
        client.get(
            f"/api/experiments/{folder.name}/evaluation/inner_validation"
        ).json()["part"]
        == "inner_validation"
    )
    assert (
        client.get(f"/api/experiments/{folder.name}/evaluation/test").status_code == 404
    )
    assert (
        client.post(
            f"/api/experiments/{folder.name}/evaluate", params={"part": "test"}
        ).status_code
        == 422
    )
    queued = data / "experiments" / str(uuid4())
    write_spec(queued, ExperimentSpec(**SPEC))
    assert client.post(f"/api/experiments/{queued.name}/evaluate").status_code == 409
    monkeypatch.setattr(routes_experiments, "_running", lambda: True)
    monkeypatch.setattr(routes_experiments, "active_id", folder.name)
    assert client.post(f"/api/experiments/{folder.name}/evaluate").status_code == 409
