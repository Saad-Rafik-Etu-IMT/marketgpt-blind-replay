import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from src.api import routes_studies
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
    challenge = client.post("/api/jury/challenge").json()
    assert set(challenge) == {"inputs", "p", "targets", "occlusion", "token"}
    url = "/api/jury/vote/" + challenge["token"]
    assert client.post(url, json={"up": True}).json()["next"] == -0.03
    assert client.post(url, json={"up": True}).status_code == 200
    assert client.post(url, json={"up": False}).status_code == 409
