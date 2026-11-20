"""Regression protection for publication, run identity, and study progress."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import routes_studies
from src.main import create_app
from src.services import study_artifacts as artifacts


def report():
    return dict(
        model="MLP",
        variant="reference",
        seed=0,
        macroF1=0.7,
        curves=[dict(validationLoss=0.5)],
        cases=[dict(p=0.7, next=0.01)],
        trace=dict(events=[1, 2, 3]),
    )


def test_summary_matches_legacy_and_avoids_full_report_read(tmp_path, monkeypatch):
    cfg = dict(models=["MLP"], variants=["reference"], seeds=[0, 1])
    path = tmp_path / "MLP-reference-0.json"
    artifacts.write_json(path, report())
    expected = artifacts.overview_runs(tmp_path, cfg)
    artifacts.write_json(
        tmp_path / "MLP-reference-0.overview",
        artifacts.overview_entry(path.stem, report()),
    )
    original = artifacts.read_json

    def read(path):
        assert path.suffix == ".overview"
        return original(path)

    monkeypatch.setattr(artifacts, "read_json", read)
    assert artifacts.overview_runs(tmp_path, cfg) == expected
    assert expected[0]["accuracy"] == 1


def test_corrupted_summary_falls_back_to_original_report(tmp_path):
    cfg = dict(models=["MLP"], variants=["reference"], seeds=[0])
    artifacts.write_json(tmp_path / "MLP-reference-0.json", report())
    expected = artifacts.overview_runs(tmp_path, cfg)
    (tmp_path / "MLP-reference-0.overview").write_text("{")
    assert artifacts.overview_runs(tmp_path, cfg) == expected


def test_worker_launch_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        routes_studies, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    monkeypatch.setattr(routes_studies, "worker", None)

    def fail(*args, **kwargs):
        raise OSError("cannot launch")

    monkeypatch.setattr(routes_studies.subprocess, "Popen", fail)
    client = TestClient(create_app())
    response = client.post("/api/studies", json={"seeds": [1, 2]})
    assert response.status_code == 503
    status = client.get("/api/studies").json()["studies"][0]
    assert (status["state"], status["done"], status["total"]) == ("failed", 0, 2)


def test_interrupted_worker_retains_completed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_studies, "worker", None)
    artifacts.write_json(
        tmp_path / "status.json",
        dict(state="running", message="Training", done=2, total=3),
    )
    value = routes_studies.status(tmp_path)
    assert (value["state"], value["done"], value["total"]) == ("interrupted", 2, 3)


def test_worker_freezes_configuration_and_preserves_failure_progress(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "learning/mlp")
    )
    import run_study

    artifacts.write_json(
        tmp_path / "config.json",
        dict(
            source="synthetic",
            models=["MLP"],
            variants=["reference"],
            seeds=[0, 1],
            epochs=1,
        ),
    )
    config, profile = object(), object()
    monkeypatch.setattr(run_study, "load_split_config", lambda: config)
    monkeypatch.setattr(run_study, "load_benchmark_profile", lambda: profile)
    monkeypatch.setattr(run_study, "synthetic_frame", lambda: None)
    monkeypatch.setattr(
        run_study, "dataset_timeline", lambda f, c: {"dataset": "fixed"}
    )
    calls = []

    def export(*args, **kwargs):
        calls.append((kwargs["config"], kwargs["profile"]))
        if len(calls) == 2:
            raise ValueError("second run failed")
        return {**report(), "dataset": "fixed"}

    monkeypatch.setattr(run_study, "export", export)
    with pytest.raises(ValueError, match="second run failed"):
        run_study.run(tmp_path)
    assert calls == [(config, profile)] * 2
    status = json.loads((tmp_path / "status.json").read_text())
    assert (status["state"], status["done"], status["total"]) == ("failed", 1, 2)
    assert (tmp_path / "MLP-reference-0.overview").is_file()
