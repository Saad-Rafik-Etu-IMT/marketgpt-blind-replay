"""T6 experiment orchestration: spec, sources, estimate, runner, worker and API."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.ai.mlp import TrainingDivergedError, load
from src.api import routes_experiments
from src.data.split import load_split_config
from src.experiments import (
    MAX_RUNS,
    ExperimentSpec,
    SourceError,
    estimate,
    health,
    load_frames,
    run_folder,
    synthetic_frames,
    worker,
    write_spec,
)
from src.experiments import runner as runner_module
from src.main import create_app
from src.services.study_artifacts import read_json

FAST = dict(
    name="test",
    feature_profiles=("returns_only_v1", "returns_plus_price_structure"),
    presets=("small",),
    epochs=2,
)


def _run(tmp_path: Path, **fields) -> tuple[Path, dict]:
    spec = ExperimentSpec(**{**FAST, **fields})
    folder = tmp_path / "experiments" / str(uuid4())
    write_spec(folder, spec)
    return folder, run_folder(folder, tmp_path)


# ---------------------------------------------------------------------- spec
@pytest.mark.parametrize(
    "fields",
    [
        {"sources": ("../etc",)},
        {"sources": ("SPY", "SPY")},
        {"primary_asset": "QQQ"},
        {"feature_profiles": ("returns_plus_market_context",)},
        {"feature_profiles": ("nope",)},
        {"presets": ("huge",)},
        {"targets": ("buy_sell_hold",)},
        {"targets": ("direction_binary_v1", "direction_binary_v1")},
        {"epochs": 11},
        {"profile": "research", "seeds": (0, 1)},
        {"seeds": (-1,)},
        {"name": "bad/name"},
        {
            "profile": "research",
            "feature_profiles": tuple(
                p
                for p in (
                    "returns_only_v1",
                    "returns_plus_trend",
                    "returns_plus_volatility",
                    "returns_plus_volume",
                    "returns_plus_statistics",
                    "core_features",
                    "extended_features",
                    "returns_plus_price_structure",
                )
            ),
            "presets": ("small", "medium"),
        },
    ],
)
def test_invalid_specs_are_refused(fields):
    with pytest.raises(ValueError):
        ExperimentSpec(**{**FAST, **fields})


def test_profiles_cap_epochs_seeds_and_patience():
    research = ExperimentSpec(
        name="r", profile="research", presets=("w1_baseline", "small")
    )
    assert research.resolved_seeds == (0, 1, 2, 3, 4) and len(research.runs()) == 10
    w1 = research.train_config("w1_baseline", seed=3)
    assert (w1.epochs, w1.patience, w1.seed) == (40, 7, 3)
    dev = ExperimentSpec(name="d", presets=("deep_v1",))
    assert dev.train_config("deep_v1", 0).epochs == 10 and not dev.reads_validation
    confirm = ExperimentSpec(name="c", profile="confirmation")
    assert (
        confirm.reads_validation and confirm.experiment_profile.walk_forward_folds == 3
    )
    assert len(confirm.runs()) == 10 <= MAX_RUNS
    assert (
        research.fingerprint()
        != ExperimentSpec(
            name="r2", profile="research", presets=("w1_baseline", "small")
        ).fingerprint()
    )


# ------------------------------------------------------------------ sources
def test_synthetic_source_covers_the_split():
    frames = synthetic_frames()
    split = load_split_config()
    for frame in frames.values():
        assert frame.index.min() < pd.Timestamp(split.train_end) < frame.index.max()
        assert frame.index.max() >= pd.Timestamp(split.validation_end)
    assert set(load_frames(("synthetic",), Path("/nonexistent"))) == {"SYN_A", "SYN_B"}


def test_local_sources_health_and_errors(tmp_path):
    raw = tmp_path / "raw" / "yfinance"
    raw.mkdir(parents=True)
    synthetic_frames()["SYN_A"].iloc[:50].to_csv(raw / "AAA.csv")
    (raw / "BAD.csv").write_text("not,a,price\n1,2,3\n")
    digest = hashlib.sha256((raw / "AAA.csv").read_bytes()).hexdigest()
    (raw / "provenance.json").write_text(
        json.dumps({"tickers": [{"ticker": "AAA", "sha256": digest}]})
    )
    report = {h["source"]: h for h in health(tmp_path)}
    assert (
        report["AAA"]["status"] == "ok"
        and report["AAA"]["rows"] == 50
        and report["AAA"]["provenance_match"]
    )
    assert (
        report["BAD"]["status"] == "unreadable"
        and report["synthetic"]["status"] == "ok"
    )
    assert len(load_frames(("AAA",), tmp_path)["AAA"]) == 50
    with pytest.raises(SourceError, match="unknown"):
        load_frames(("ZZZ",), tmp_path)
    with pytest.raises(SourceError, match="mixed"):
        load_frames(("synthetic", "AAA"), tmp_path)
    with pytest.raises(SourceError, match="unreadable"):
        load_frames(("BAD",), tmp_path)
    assert health(tmp_path / "empty")[-1]["source"] == "synthetic"


def test_estimate_reports_runs_and_budget():
    spec = ExperimentSpec(
        name="e",
        profile="research",
        feature_profiles=("core_features",),
        presets=("large",),
    )
    result = estimate(spec, {"synthetic": 8000})
    assert result["runs"] == 5 and result["per_run"][0]["parameters"] > 0
    assert result["within_budget"] and result["estimated_seconds"] > 0


# ------------------------------------------------------------------- runner
def test_runner_end_to_end_without_reading_validation(tmp_path):
    folder, status = _run(tmp_path, targets=("direction_binary_v1", "volatility_h5"))
    assert status["state"] == "complete" and status["done"] == status["total"] == 2
    report = read_json(folder / "runs" / "returns_plus_price_structure__small__s0.json")
    assert set(report["inner_validation"]) == {"direction_binary_v1", "volatility_h5"}
    assert (
        "validation" not in report
        and not (tmp_path / "experiments" / "validation_reads.json").exists()
    )
    assert report["rows"]["validation"] > 0
    rows = {
        read_json(folder / "runs" / f"{fp}__small__s0.json")["rows"]["inner_train"]
        for fp in FAST["feature_profiles"]
    }
    assert len(rows) == 1  # every feature profile on the same rows
    model = load(folder / report["checkpoint"])
    assert (
        model.manifest["dataset_fingerprint"]
        == report["dataset_fingerprints"]["direction_binary_v1"]
    )
    assert len(list((folder / "datasets").glob("*.json"))) == 4


def test_resume_skips_completed_runs(tmp_path):
    folder, _ = _run(tmp_path)
    path = folder / "runs" / "returns_only_v1__small__s0.json"
    before = path.stat().st_mtime_ns
    path.unlink()
    (folder / "runs" / "returns_plus_price_structure__small__s0.json").write_text("{}")
    other = (
        (folder / "runs" / "returns_plus_price_structure__small__s0.json")
        .stat()
        .st_mtime_ns
    )
    status = run_folder(folder, tmp_path)
    assert status["state"] == "complete" and path.stat().st_mtime_ns != before
    assert (
        folder / "runs" / "returns_plus_price_structure__small__s0.json"
    ).stat().st_mtime_ns != other


def test_a_failing_run_does_not_stop_the_experiment(tmp_path, monkeypatch):
    real = runner_module.Runner._run

    def flaky(self, run, datasets):
        if run.feature_profile == "returns_only_v1":
            raise TrainingDivergedError("boom", [])
        return real(self, run, datasets)

    monkeypatch.setattr(runner_module.Runner, "_run", flaky)
    folder, status = _run(tmp_path)
    assert status["state"] == "complete_with_failures" and status["failed"] == [
        "returns_only_v1__small__s0"
    ]
    assert read_json(folder / "runs" / "returns_only_v1__small__s0.json")[
        "error"
    ].startswith("TrainingDivergedError")


def test_cancel_time_budget_and_crash(tmp_path, monkeypatch):
    spec = ExperimentSpec(**FAST)
    folder = tmp_path / "experiments" / "c1"
    write_spec(folder, spec)
    (folder / "cancel").write_text("x")
    assert run_folder(folder, tmp_path)["state"] == "cancelled"

    clock = iter(range(0, 10**6, 10_000))
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: float(next(clock)))
    folder, status = _run(tmp_path, max_minutes=1)
    assert status["state"] == "stopped" and "budget" in status["message"]
    monkeypatch.undo()

    folder, status = _run(tmp_path, sources=("MISSING",))
    assert status["state"] == "failed" and (folder / "worker_error.txt").is_file()


def test_tampered_spec_is_refused(tmp_path):
    folder = tmp_path / "experiments" / "t1"
    write_spec(folder, ExperimentSpec(**FAST))
    stored = read_json(folder / "spec.json")
    stored["spec"]["epochs"] = 1
    (folder / "spec.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="fingerprint"):
        run_folder(folder, tmp_path)


def test_confirmation_reads_validation_once_per_run_and_walks_forward(tmp_path):
    folder, status = _run(
        tmp_path,
        profile="confirmation",
        feature_profiles=("returns_only_v1",),
        epochs=1,
        seeds=tuple(range(10)),
    )
    assert status["state"] == "complete" and status["done"] == 10
    report = read_json(folder / "runs" / "returns_only_v1__small__s0.json")
    assert "validation" in report
    folds = report["walk_forward"]
    assert len(folds) == 3
    assert [f["start"] for f in folds] == sorted(f["start"] for f in folds)
    assert folds[0]["train_rows"] < folds[1]["train_rows"] < folds[2]["train_rows"]
    assert all(f["evaluated_rows"] > 0 for f in folds)
    reads = read_json(tmp_path / "experiments" / "validation_reads.json")["reads"]
    assert len(reads) == 10 and {r["experiment"] for r in reads} == {folder.name}


def test_worker_entry_point(tmp_path, monkeypatch, capsys):
    assert worker.main([]) == 2
    assert worker.main([str(tmp_path)]) == 2
    folder = tmp_path / "experiments" / "w1"
    write_spec(
        folder, ExperimentSpec(**{**FAST, "feature_profiles": ("returns_only_v1",)})
    )
    monkeypatch.setattr(
        worker, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    assert worker.main([str(folder)]) == 0
    assert "complete (1/1)" in capsys.readouterr().out


# ---------------------------------------------------------------------- API
@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_experiments, "data_dir", lambda: tmp_path)
    launched: list[Path] = []

    def fake_launch(folder):
        launched.append(folder)
        routes_experiments.active_id = folder.name
        run_folder(folder, tmp_path)

    monkeypatch.setattr(routes_experiments, "_launch", fake_launch)
    monkeypatch.setattr(routes_experiments, "worker", None)
    return TestClient(create_app()), launched


def test_api_catalogue_sources_and_preview(api):
    client, _ = api
    catalog = client.get("/api/experiments/catalog").json()
    assert {
        "features",
        "feature_profiles",
        "targets",
        "presets",
        "experiment_profiles",
    } <= set(catalog)
    assert client.get("/api/experiments/features/rsi_14").json()["id"] == "rsi_14"
    assert client.get("/api/experiments/features/nope").status_code == 404
    assert client.get("/api/experiments/sources").json()[-1]["source"] == "synthetic"
    preview = client.get(
        "/api/experiments/sources/synthetic/preview",
        params={"feature": "rsi_14", "last": 30},
    ).json()
    assert (
        len(preview["dates"]) == 30
        and preview["dates"][-1] <= load_split_config().validation_end.isoformat()
    )
    assert (
        client.get(
            "/api/experiments/sources/ZZZ/preview", params={"feature": "rsi_14"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/experiments/sources/synthetic/preview", params={"feature": "nope"}
        ).status_code
        == 404
    )


def test_api_lifecycle_and_results(api):
    client, launched = api
    body = {
        **FAST,
        "feature_profiles": list(FAST["feature_profiles"]),
        "presets": ["small"],
    }
    assert client.post("/api/experiments/estimate", json=body).json()["runs"] == 2
    created = client.post("/api/experiments", json=body)
    assert created.status_code == 202
    experiment = created.json()["id"]
    assert client.get("/api/experiments").json()[0]["id"] == experiment
    detail = client.get(f"/api/experiments/{experiment}").json()
    assert detail["status"]["state"] == "complete"
    runs = client.get(f"/api/experiments/{experiment}/runs").json()
    assert {r["state"] for r in runs} == {"complete"} and len(runs) == 2
    key = runs[0]["key"]
    assert client.get(f"/api/experiments/{experiment}/runs/{key}").json()["key"] == key
    assert (
        client.get(f"/api/experiments/{experiment}/runs/..%2Fspec").status_code == 404
    )
    assert client.get(f"/api/experiments/{experiment}/runs/a__b__s9").status_code == 404
    compared = client.get(f"/api/experiments/{experiment}/compare").json()
    assert (
        len(compared) == 2
        and compared[0]["metrics"]["direction_binary_v1.log_loss"]["n"] == 1
    )
    assert (
        client.get(
            f"/api/experiments/{experiment}/compare", params={"part": "test"}
        ).status_code
        == 422
    )
    assert client.get(f"/api/experiments/{uuid4()}").status_code == 404
    assert (
        client.post(f"/api/experiments/{experiment}/cancel").json()["state"]
        == "complete"
    )
    assert client.post(f"/api/experiments/{experiment}/resume").status_code == 409
    assert len(launched) == 1


def test_api_cancel_then_resume(api, tmp_path):
    client, launched = api
    folder = tmp_path / "experiments" / str(uuid4())
    write_spec(
        folder, ExperimentSpec(**{**FAST, "feature_profiles": ("returns_only_v1",)})
    )
    assert (
        client.post(f"/api/experiments/{folder.name}/cancel").json()["state"]
        == "cancelled"
    )
    assert run_folder(folder, tmp_path)["state"] == "cancelled"
    assert client.post(f"/api/experiments/{folder.name}/resume").status_code == 202
    assert not (folder / "cancel").exists() and len(launched) == 1
    assert read_json(folder / "status.json")["state"] == "complete"


def test_api_refusals(api, monkeypatch):
    client, _ = api
    body = {**FAST, "feature_profiles": ["returns_only_v1"], "presets": ["small"]}
    assert (
        client.post("/api/experiments", json={**body, "sources": ["ZZZ"]}).status_code
        == 422
    )
    assert (
        client.post("/api/experiments", json={**body, "epochs": 99}).status_code == 422
    )
    monkeypatch.setattr(
        routes_experiments,
        "estimate",
        lambda spec, rows: {
            "within_budget": False,
            "estimated_seconds": 1e9,
            "budget_seconds": 1,
        },
    )
    assert client.post("/api/experiments", json=body).status_code == 422
    monkeypatch.setattr(routes_experiments, "_running", lambda: True)
    monkeypatch.setattr(
        routes_experiments, "estimate", lambda spec, rows: {"within_budget": True}
    )
    assert client.post("/api/experiments", json=body).status_code == 409


def test_api_resume_refuses_a_complete_experiment(api, tmp_path):
    client, _ = api
    folder = tmp_path / "experiments" / str(uuid4())
    write_spec(
        folder, ExperimentSpec(**{**FAST, "feature_profiles": ("returns_only_v1",)})
    )
    run_folder(folder, tmp_path)
    assert client.post(f"/api/experiments/{folder.name}/resume").status_code == 409
    assert date.fromisoformat(read_json(folder / "status.json")["updated_at"][:10])
