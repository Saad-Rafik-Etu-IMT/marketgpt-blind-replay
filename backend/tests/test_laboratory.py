"""Cross-boundary regressions for the automated research laboratory."""

from datetime import date, timedelta
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import routes_experiments
from src.datasets import ContextRule
from src.evaluation import evaluate_experiment
from src.evaluation.evaluate import classification_curves
from src.experiments import ExperimentSpec, Runner, run_folder, write_spec
from src.experiments.campaigns import campaigns
from src.experiments.holdout import evaluate, freeze
from src.experiments.models import load_registered
from src.experiments.replay import replay_run
from src.experiments.reporting import export_report
from src.experiments.runner import chronological_masks
from src.experiments.sources import synthetic_frames
from src.main import create_app
from src.services.study_artifacts import read_json


def test_search_is_bounded_and_paired():
    spec = ExperimentSpec(
        name="search",
        profile="research",
        epochs=2,
        feature_profiles=("returns_only_v1", "returns_plus_volatility"),
        presets=("small", "medium"),
        search={"budget": 3, "seed": 42},
    )
    assert len(spec.runs()) == 30
    assert (
        spec.candidates()
        == ExperimentSpec.model_validate(spec.model_dump()).candidates()
    )
    assert len({r.key for r in spec.runs()}) == 30
    assert all(spec.train_config(r.preset, r.seed).epochs == 2 for r in spec.runs())
    with pytest.raises(ValueError, match="grid exceeds"):
        ExperimentSpec(name="bad grid", search={"method": "grid", "budget": 2})
    assert all(
        not ExperimentSpec.model_validate(c["spec"]).report_validation
        for c in campaigns()
    )


def test_positive_control_contains_causal_signal():
    noise = synthetic_frames()["SYN_A"]
    signal = synthetic_frames(signal=True)["SYN_A"]
    assert (
        abs(
            np.diff(np.log(noise.close)).reshape(-1).size
            - np.diff(np.log(signal.close)).size
        )
        == 0
    )
    assert np.log(signal.close).diff().autocorr() > 0.6
    assert abs(np.log(noise.close).diff().autocorr()) < 0.1


def test_fold_never_fits_future_only_asset():
    index = pd.DataFrame(
        {
            "asset": ["A"] * 12 + ["B"] * 3,
            "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(12)]
            + [date(2020, 2, 1) + timedelta(days=i) for i in range(3)],
            "bar": list(range(12)) + [0, 1, 2],
        }
    )
    fit, test = chronological_masks(
        index, date(2020, 1, 7), date(2020, 1, 12), 1, 1, 2, ContextRule.OWN_PARTITION
    )
    assert not fit[-3:].any()
    assert not test[-3:].any()
    assert index.loc[fit, "date"].max() < index.loc[test, "date"].min()
    assert not (fit & test).any()


def test_classification_curves_handle_ties_and_absent_class():
    points = classification_curves(
        np.array([0.8, 0.8, 0.2, 0.2]), np.array([1, 0, 0, 1])
    )
    assert points["roc"] == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    assert points["precision_recall"][-1] == [1.0, 0.5]
    assert classification_curves(np.ones(3), np.ones(3))["roc"] == []


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    data = tmp_path_factory.mktemp("research")
    folder = data / "experiments" / str(uuid4())
    write_spec(folder, ExperimentSpec(name="lab smoke", epochs=1, presets=("small",)))
    status = run_folder(folder, data)
    assert status["state"] == "complete"
    return data, folder, "returns_only_v1__small__s0"


def test_data_identity_is_frozen(registered):
    data, folder, _ = registered
    runner = Runner(folder, data)
    changed = synthetic_frames()
    changed["SYN_A"].iloc[5, changed["SYN_A"].columns.get_loc("volume")] += 1
    with pytest.raises(ValueError, match="Development data or split changed"):
        runner.verify_data(changed)


def test_registered_replay_uses_engine_and_exports(registered, monkeypatch):
    data, folder, key = registered
    report = replay_run(folder, data, key, "inner_validation")
    assert len(report["players"]) == 3
    model = report["players"][0]
    assert model["decisions"] > 0
    event = model["events"][0]
    assert (
        event["portfolio"]["bar_index"] == event["decision"]["decision_bar_index"] + 1
    )
    assert 0 <= event["decision"]["probabilities"]["long"] <= 1
    if event["execution"] and event["execution"]["fill_bar"] is not None:
        assert event["execution"]["fill_bar"] > event["decision"]["decision_bar_index"]
    monkeypatch.setattr(routes_experiments, "data_dir", lambda: data)
    monkeypatch.setattr(routes_experiments, "worker", None)
    client = TestClient(create_app())
    prefix = f"/api/experiments/{folder.name}/runs/{key}"
    assert client.get(prefix + "/replay").json()["run"] == key
    facts = client.post(prefix + "/replay/explain?index=0").json()["facts"]
    assert len(facts) == 3 and all("equity" not in f for f in facts)
    assert (
        client.post(prefix + "/replay/explain?index=0&outcome=true").status_code == 200
    )
    assert client.get(prefix + "/progress").json()["history"][0]["selected"]
    assert client.get(f"/api/experiments/{folder.name}/export/csv").status_code == 200
    assert "run,state,partition,target,metric,value" in export_report(folder, "csv")[0]
    assert "Development evidence" in export_report(folder, "md")[0]


def test_queue_persists_and_dispatches_waiting_job(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_experiments, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        routes_experiments, "worker", SimpleNamespace(poll=lambda: None)
    )
    monkeypatch.setattr(routes_experiments, "active_id", "existing")
    launched = []
    monkeypatch.setattr(
        routes_experiments, "_launch", lambda folder: launched.append(folder.name)
    )
    client = TestClient(create_app())
    correlation = client.get("/api/experiments/sources/synthetic/correlation").json()
    assert correlation["rows"] > 0
    assert len(correlation["matrix"]) == len(correlation["columns"])
    body = {"name": "queued", "epochs": 1}
    queued = client.post("/api/experiments?enqueue=true", json=body)
    assert queued.status_code == 202
    campaign = queued.json()["id"]
    assert (
        client.get(f"/api/experiments/{campaign}").json()["status"]["state"] == "queued"
    )
    assert not launched
    monkeypatch.setattr(routes_experiments, "worker", None)
    assert client.post("/api/experiments/queue/dispatch").json()["id"] == campaign
    assert launched == [campaign]


def test_prospective_holdout_is_frozen_and_not_revealed(tmp_path):
    raw = tmp_path / "raw" / "yfinance"
    raw.mkdir(parents=True)
    synthetic_frames()["SYN_A"].to_csv(raw / "AAA.csv")
    folder = tmp_path / "experiments" / str(uuid4())
    write_spec(folder, ExperimentSpec(name="freeze", sources=("AAA",), epochs=1))
    assert run_folder(folder, tmp_path)["state"] == "complete"
    key = "returns_only_v1__w1_baseline__s0"
    start, end = date.today() + timedelta(days=1), date.today() + timedelta(days=60)
    plan = freeze(folder, tmp_path, key, start, end)
    with pytest.raises(FileExistsError):
        freeze(folder, tmp_path, key, start, end)
    with pytest.raises(ValueError, match="Authorization"):
        evaluate(folder, tmp_path, "invalid")
    with pytest.raises(ValueError, match="not ended"):
        evaluate(folder, tmp_path, plan["authorization_id"])
    assert not (folder / "holdout_consumed.json").exists()
    assert not (folder / "holdout_result.json").exists()


def test_prospective_holdout_once_on_generated_fixture(tmp_path, monkeypatch):
    from src.experiments import holdout

    raw = tmp_path / "raw" / "yfinance"
    raw.mkdir(parents=True)
    frame = synthetic_frames()["SYN_A"]
    frame.to_csv(raw / "AAA.csv")
    folder = tmp_path / "experiments" / str(uuid4())
    write_spec(folder, ExperimentSpec(name="fixture only", sources=("AAA",), epochs=1))
    assert run_folder(folder, tmp_path)["state"] == "complete"
    start = date.today() + timedelta(days=1)
    end = start + timedelta(days=90)
    plan = freeze(folder, tmp_path, "returns_only_v1__w1_baseline__s0", start, end)

    # This is newly generated fixture data, never the repository's sealed data.
    future = synthetic_frames(signal=True)["SYN_A"].copy()
    future.index = pd.bdate_range("2020-01-02", periods=len(future), name="date")
    future = future.loc[future.index <= pd.Timestamp(end)]
    future.to_csv(raw / "AAA.csv", mode="a", header=False)

    class AfterPeriod(date):
        @classmethod
        def today(cls):
            return end + timedelta(days=1)

    monkeypatch.setattr(holdout, "date", AfterPeriod)
    result = evaluate(folder, tmp_path, plan["authorization_id"])
    assert result["metrics"]["direction_binary_v1"]["rows"] > 0
    with pytest.raises(FileExistsError):
        evaluate(folder, tmp_path, plan["authorization_id"])


def test_registered_transformer_trains_and_evaluates(tmp_path):
    pytest.importorskip("torch")
    folder = tmp_path / "experiments" / str(uuid4())
    write_spec(
        folder,
        ExperimentSpec(name="attention", presets=("transformer_reference",), epochs=1),
    )
    assert run_folder(folder, tmp_path)["state"] == "complete"
    key = "returns_only_v1__transformer_reference__s0"
    model = load_registered(folder / "checkpoints" / key)
    assert model.manifest["device"] == "cpu"
    outputs = model.predict_raw_features(np.zeros((3, 64)))["direction_binary_v1"]
    assert outputs.shape == (3, 2)
    assert np.isfinite(outputs).all()
    report = evaluate_experiment(folder, tmp_path, n_boot=10)
    assert report["results"][0]["preset"] == "transformer_reference"
    assert read_json(folder / "runs" / f"{key}.json")["best_epoch"] == 1
