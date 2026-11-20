import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import routes_studies
from src.data.split import load_split_config
from src.main import create_app


def test_jury_reveals_only_after_fixed_vote(monkeypatch, tmp_path):
    study = str(uuid4())
    folder = tmp_path / "studies" / study
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(
        json.dumps(dict(models=["MLP"], variants=["reference"], seeds=[0]))
    )
    case = dict(
        p=0.7,
        next=-0.03,
        actions=[dict(target="long", price=100)],
        occlusion=[0.01] * 8,
    )
    (folder / "MLP-reference-0.json").write_text(
        json.dumps(dict(cases=[case], returns=[0] * 64))
    )
    monkeypatch.setattr(
        routes_studies, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    monkeypatch.setenv("MRTL_JURY_STUDY", study)
    monkeypatch.setenv("MRTL_JURY_RUN", "MLP-reference-0")
    monkeypatch.setenv("MRTL_JURY_ONLY", "1")
    client = TestClient(create_app())
    assert client.get("/api/studies").status_code == 404
    assert client.get(f"/api/studies/{study}/competition").status_code == 404
    assert client.get(f"/api/studies/{study}/timeline").status_code == 404
    challenge = client.post("/api/jury/challenge").json()
    assert set(challenge) == {"inputs", "p", "targets", "occlusion", "token"}
    url = "/api/jury/vote/" + challenge["token"]
    assert client.post(url, json={"up": True}).json()["next"] == -0.03
    assert client.post(url, json={"up": True}).status_code == 200
    assert client.post(url, json={"up": False}).status_code == 409


@pytest.fixture
def recorded_study(monkeypatch, tmp_path):
    study = str(uuid4())
    folder = tmp_path / "studies" / study
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(
        json.dumps(
            dict(models=["MLP", "Transformer"], variants=["reference"], seeds=[0])
        )
    )
    case = dict(
        p=0.72,
        next=-0.03,
        period="2018-01",
        actions=[dict(target="long", status="filled")] * 3,
        occlusion=[0.08] + [0.01] * 7,
    )
    initial = dict(equity=100000, cash=100000)
    event = dict(
        decision=dict(order_side="buy", action="submit_buy"),
        execution=dict(fill_quantity=1, fill_price=100, fee=0, status="filled"),
        portfolio=dict(equity=99995, cash=99900),
    )
    baseline = dict(name="buy_and_hold", initial=initial, events=[event, event])
    for model in ("MLP", "Transformer"):
        trace = dict(
            profileHash="same",
            market=[dict(close=100), dict(close=99)],
            profile={},
            firstDecision=64,
            players=[
                dict(name=model, initial=initial, events=[event, event]),
                baseline,
                baseline,
            ],
        )
        run = dict(
            dataset="same",
            accounting="raw-price",
            origin="synthetic",
            checkpoint=model,
            macroF1=0.4,
            strategies=[dict(name=model), dict(name="ma"), dict(name="hold")],
            cases=[case, case],
            returns=[0] * 65,
            trace=trace,
        )
        (folder / f"{model}-reference-0.json").write_text(json.dumps(run))
    monkeypatch.setattr(
        routes_studies, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    return study, folder


def test_study_picker_identifies_source_models_and_recorded_replay(recorded_study):
    study, folder = recorded_study
    config = json.loads((folder / "config.json").read_text())
    config.update(source="SPY", epochs=2)
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "status.json").write_text(
        json.dumps(dict(state="complete", message="Ready", done=2, total=2))
    )
    client = TestClient(create_app())
    summary = client.get("/api/studies").json()["studies"][0]
    assert summary["id"] == study
    assert summary["source"] == "SPY"
    assert summary["models"] == ["MLP", "Transformer"]
    assert summary["epochs"] == 2
    assert not summary["hasTrace"]
    (folder / "timeline.json").write_text("{}")
    assert client.get("/api/studies").json()["studies"][0]["hasTrace"]


def test_competition_joins_exact_recorded_runs(recorded_study):
    study, folder = recorded_study
    client = TestClient(create_app())
    url = f"/api/studies/{study}/competition?seed=0"
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert len(data["players"]) == 4
    assert [p["run"] for p in data["players"]] == [
        "MLP-reference-0",
        "Transformer-reference-0",
        None,
        None,
    ]
    assert data["players"][0]["events"][0]["portfolio"]["equity"] == 99995
    path = folder / "Transformer-reference-0.json"
    report = json.loads(path.read_text())
    report["trace"]["profileHash"] = "different"
    path.write_text(json.dumps(report))
    assert client.get(url).status_code == 409


def test_measured_explanations_respect_phase(recorded_study, monkeypatch):
    monkeypatch.delenv("MRTL_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("MRTL_GGUF_MODEL", raising=False)
    study, _ = recorded_study
    client = TestClient(create_app())
    url = f"/api/studies/{study}/MLP-reference-0/explain/0"
    before = client.post(url + "?mode=standard").json()
    text = " ".join(before["facts"])
    assert "72.0%" in text and "-8.0 percentage points" in text
    assert "99,995" not in text
    assert before["source"] == "Measured template"
    after = client.post(url + "?mode=standard&phase=outcome").json()
    assert "99,995.00" in " ".join(after["facts"])
    assert "-5.00" in " ".join(after["facts"])


def test_human_scores_pair_participants_and_reject_duplicate_exposure(recorded_study):
    study, _ = recorded_study
    client = TestClient(create_app())
    response = dict(
        participant="P01",
        case=1,
        format="facts",
        source="Facts",
        probability_percent=72,
        block=1,
    )
    llm = dict(
        **{
            **response,
            "case": 2,
            "format": "llm",
            "source": "Measured template",
            "fallback": True,
            "block": 3,
        }
    )
    url = f"/api/studies/{study}/MLP-reference-0/human-evaluation"
    assert client.get(url).json() is None
    result = client.post(url, json={"responses": [response, llm]})
    assert result.status_code == 200
    summary = result.json()["summary"]
    assert summary["pairs"] == [dict(participant="P01", facts=1.0, llm=0.5)]
    assert summary["fallback"] == 1
    assert client.get(url).json()["summary"] == summary
    assert client.post(url, json={"responses": [response, response]}).status_code == 422
    assert client.get(url).json()["summary"] == summary


def test_timeline_does_not_include_test_values(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "learning" / "mlp")
    )
    from export_comparison import dataset_timeline, synthetic_frame

    frame = synthetic_frame()
    config = load_split_config()
    timeline = dataset_timeline(frame, config)
    test_row = frame.iloc[[-1]].copy()
    test_row.index = pd.DatetimeIndex([pd.Timestamp(config.test_start)])
    test_row["adjusted_close"] = 1e9
    assert dataset_timeline(pd.concat([frame, test_row]), config) == timeline
    assert all(p["split"] != "test" for p in timeline["points"])
