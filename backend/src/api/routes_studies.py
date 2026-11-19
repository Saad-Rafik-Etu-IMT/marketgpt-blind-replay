"""Local validation studies and server-held blind challenge outcomes."""

import json
import os
import secrets
import sqlite3
import subprocess
import sys
from contextlib import closing
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import BACKEND_DIR, get_settings

router = APIRouter(prefix="/studies", tags=["research"])
lock = Lock()
worker = None
active_id = None
Variant = Literal["reference", "no-position", "no-dropout", "no-decay", "fixed-lr"]


class StudyConfig(BaseModel):
    source: str = Field(default="synthetic", pattern=r"^[a-zA-Z0-9_-]{1,40}$")
    seeds: list[int] = Field(default=[0], min_length=1, max_length=5)
    models: list[Literal["MLP", "Transformer"]] = Field(
        default=["MLP"], min_length=1, max_length=2
    )
    variants: list[Variant] = Field(default=["reference"], min_length=1, max_length=5)
    epochs: int = Field(default=8, ge=1, le=60)


def root():
    return get_settings().data_dir / "studies"


def sources():
    raw = get_settings().data_dir / "raw" / "yfinance"
    return {p.stem: p for p in raw.glob("*.csv") if p.is_file()}


def read(path):
    if not path.is_file():
        raise HTTPException(404, "Study artifact unavailable")
    return json.loads(path.read_text(encoding="utf-8"))


def status(folder):
    value = read(folder / "status.json")
    if value["state"] in {"running", "queued"} and (
        active_id != folder.name or worker is None or worker.poll() is not None
    ):
        value = dict(
            state="interrupted",
            message="Worker stopped. Start a new study.",
            done=0,
            total=0,
        )
    return value


@router.get("")
def list_studies():
    return dict(
        sources=["synthetic", *sorted(sources())],
        transformer=find_spec("torch") is not None,
        studies=[
            dict(id=p.parent.name, **status(p.parent))
            for p in sorted(
                root().glob("*/status.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ],
    )


@router.post("", status_code=202)
def start(config: StudyConfig):
    global worker, active_id
    spec = config.model_dump()
    for key in ("models", "variants", "seeds"):
        spec[key] = list(dict.fromkeys(spec[key]))
    if any(s < 0 or s > 2**31 - 1 for s in spec["seeds"]):
        raise HTTPException(422, "Seed outside supported range")
    if "Transformer" in spec["models"] and find_spec("torch") is None:
        raise HTTPException(
            422, "Install the CPU torch package in the backend environment"
        )
    if config.source != "synthetic":
        if config.source not in sources():
            raise HTTPException(422, "Unknown local dataset")
        spec["csv"] = str(sources()[config.source].resolve())
    with lock:
        if worker is not None and worker.poll() is None:
            raise HTTPException(409, "A study is already training")
        folder = root() / str(uuid4())
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(spec))
        (folder / "status.json").write_text(
            json.dumps(dict(state="queued", message="Starting", done=0, total=0))
        )
        with (folder / "worker.log").open("w") as log:
            worker = subprocess.Popen(
                [
                    sys.executable,
                    str(BACKEND_DIR / "learning/mlp/run_study.py"),
                    str(folder.resolve()),
                ],
                cwd=BACKEND_DIR,
                stdout=log,
                stderr=log,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
            )
        active_id = folder.name
        return {"id": folder.name}


def report(study, key):
    folder = root() / str(study)
    config = read(folder / "config.json")
    keys = [
        f"{m}-{v}-{s}"
        for m in config["models"]
        for v in (config["variants"] if m == "Transformer" else ["reference"])
        for s in config["seeds"]
    ]
    if key not in keys:
        raise HTTPException(404, "Unknown model run")
    return read(folder / (key + ".json"))


@router.get("/results")
def results_catalog():
    rows = []
    for path in sorted(root().glob("*/summary-*.meta")):
        rows.append(read(path))
    return rows


@router.get("/contributions")
def contributions():
    paths = {
        "Data preparation": "backend/src/data",
        "Model training": "backend/learning/mlp",
        "Replay and evaluation": "backend/src/eval",
        "Cockpit": "frontend",
    }
    items = []
    for area, path in paths.items():
        history = subprocess.run(
            ["git", "log", "--format=%an%x09%H", "--", path],
            cwd=BACKEND_DIR.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        authors = {}
        for line in history.stdout.splitlines():
            name, _, sha = line.partition("\t")
            authors.setdefault(name, sha)
        items.append(
            dict(
                area=area,
                authors=[dict(name=name, commit=sha) for name, sha in authors.items()],
            )
        )
    return items


@router.get("/{study}")
def overview(study: UUID):
    folder = root() / str(study)
    cfg = read(folder / "config.json")
    cfg.pop("csv", None)
    runs = []
    for path in sorted(folder.glob("*-*.json")):
        r = read(path)
        runs.append(
            dict(
                key=path.stem,
                model=r["model"],
                variant=r["variant"],
                seed=r["seed"],
                f1=r["macroF1"],
                loss=min(x["validationLoss"] for x in r["curves"]),
                curves=r["curves"],
            )
        )
    return dict(config=cfg, status=status(folder), runs=runs)


@router.get("/{study}/{key}/evaluation")
def evaluation(study: UUID, key: str):
    r = report(study, key)
    cases = r.pop("cases")
    r["rejections"] = [
        sum(
            c["actions"][i]["status"] in {"rejected", "pre_trade_rejected"}
            for c in cases
        )
        for i in range(len(r["strategies"]))
    ]
    r.pop("returns")
    r["samples"] = [
        dict(index=i, p=c["p"], up=c["next"] > 0, period=c["period"])
        for i, c in enumerate(cases)
    ]
    return r


def observe(r, i):
    if not 0 <= i < len(r["cases"]):
        raise HTTPException(404, "Unknown case")
    c = r["cases"][i]
    return dict(
        inputs=r["returns"][i : i + 64],
        p=c["p"],
        targets=[a["target"] for a in c["actions"]],
        occlusion=c["occlusion"],
    )


@router.get("/{study}/{key}/case/{index}")
def inspect(study: UUID, key: str, index: int):
    r = report(study, key)
    observation = observe(r, index)
    case = r["cases"][index]
    return dict(**observation, outcome=dict(next=case["next"], actions=case["actions"]))


@router.post("/{study}/{key}/challenge")
def challenge(study: UUID, key: str):
    r = report(study, key)
    index = secrets.randbelow(len(r["cases"]))
    token = uuid4()
    with closing(sqlite3.connect(root() / str(study) / "jury.sqlite")) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS challenges "
            "(token TEXT PRIMARY KEY, run_key TEXT, case_idx INTEGER, up INTEGER)"
        )
        db.execute(
            "INSERT INTO challenges VALUES (?, ?, ?, NULL)", (str(token), key, index)
        )
    return dict(token=token, **observe(r, index))


class Vote(BaseModel):
    up: bool


@router.post("/{study}/{key}/vote/{token}")
def vote(study: UUID, key: str, token: UUID, request: Vote):
    database = root() / str(study) / "jury.sqlite"
    if not database.is_file():
        raise HTTPException(404, "Unknown challenge")
    with closing(sqlite3.connect(database)) as db, db:
        # The conditional update serializes competing votes across both listeners.
        db.execute(
            "UPDATE challenges SET up = ? WHERE token = ? AND run_key = ? AND up IS NULL",
            (request.up, str(token), key),
        )
        row = db.execute(
            "SELECT case_idx, up FROM challenges WHERE token = ? AND run_key = ?",
            (str(token), key),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Unknown challenge")
        index, up = row
        if bool(up) != request.up:
            raise HTTPException(409, "Vote already recorded")
    r = report(study, key)
    return dict(index=index, vote=bool(up), **r["cases"][index])


@router.post("/{study}/{key}/explain/{index}")
def explain(study: UUID, key: str, index: int):
    obs = observe(report(study, key), index)
    block = max(range(8), key=lambda j: abs(obs["occlusion"][j]))
    facts = [
        f"The estimated upward probability was {obs['p']:.1%}.",
        f"The position target was {obs['targets'][0]}.",
        f"Replacing offsets {8*block-63} to {8*block-56} with the training mean changed P(up) by {-obs['occlusion'][block]:+.3f}.",
    ]
    model = os.getenv("MRTL_OLLAMA_MODEL", "")
    gguf = os.getenv("MRTL_GGUF_MODEL")
    if gguf:
        try:
            from src.services.local_explanation import verbalize

            generated = verbalize(gguf, facts)
            return dict(
                source=Path(gguf).name,
                check="Constrained generation; all three facts verified",
                facts=generated,
            )
        except (ImportError, OSError, ValueError, RuntimeError):
            model = model or ""
    if model:
        try:
            prompt = (
                "Write a concise decision explanation using these sentences verbatim. Order them by importance. Return JSON with a sentences array containing each sentence exactly once. "
                + json.dumps(facts)
            )
            body = json.dumps(
                dict(
                    model=model,
                    prompt=prompt,
                    stream=False,
                    format="json",
                    options=dict(temperature=0, num_predict=180),
                )
            ).encode()
            with urlopen(
                Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=30,
            ) as res:
                sentences = json.loads(json.load(res)["response"])["sentences"]
            if sorted(sentences) == sorted(facts):
                return dict(
                    source=model,
                    check="All cited sentences matched measured facts",
                    facts=sentences,
                )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return dict(
        source="Measured template",
        check=(
            "Local generation unavailable or failed fact validation"
            if model or gguf
            else "Local model is not configured"
        ),
        facts=facts,
    )
