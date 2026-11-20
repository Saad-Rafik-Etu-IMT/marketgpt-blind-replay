"""Local validation studies and server-held blind challenge outcomes."""

import json
import math
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
from src.eval.human_explanations import Responses, score_responses

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
                hasTrace="trace" in r,
                accuracy=sum((c["p"] >= 0.5) == (c["next"] > 0) for c in r["cases"])
                / len(r["cases"]),
            )
        )
    return dict(config=cfg, status=status(folder), runs=runs)


@router.get("/{study}/timeline")
def timeline(study: UUID):
    return read(root() / str(study) / "timeline.json")


@router.get("/{study}/competition")
def competition(study: UUID, seed: int = 0, variant: Variant = "reference"):
    """Join recorded autonomous sessions on the exact same market clock."""
    folder = root() / str(study)
    cfg = read(folder / "config.json")
    candidates = [
        f"{model}-{variant if model == 'Transformer' else 'reference'}-{seed}"
        for model in cfg["models"]
        if seed in cfg["seeds"] and (model == "MLP" or variant in cfg["variants"])
    ]
    runs = [
        (key, read(folder / (key + ".json")))
        for key in candidates
        if (folder / (key + ".json")).is_file()
    ]
    recorded = [(key, r) for key, r in runs if "trace" in r]
    if not recorded:
        raise HTTPException(
            404,
            "No recorded engine trace for this selection. Train a new study to record one.",
        )
    _, anchor = recorded[0]
    trace = anchor["trace"]
    players = []
    for key, run in recorded:
        other = run["trace"]
        if (
            run["dataset"] != anchor["dataset"]
            or run["accounting"] != anchor["accounting"]
            or other["profileHash"] != trace["profileHash"]
            or other["market"] != trace["market"]
            or other["firstDecision"] != trace["firstDecision"]
            or len(run["cases"]) != len(anchor["cases"])
            or run["strategies"][1:] != anchor["strategies"][1:]
            or [c["next"] for c in run["cases"]] != [c["next"] for c in anchor["cases"]]
        ):
            raise HTTPException(
                409,
                "Recorded runs do not share identical market and execution settings",
            )
        players.append(
            dict(
                **other["players"][0],
                run=key,
                sourceRun=key,
                playerIndex=0,
                checkpoint=run["checkpoint"],
                macroF1=run["macroF1"],
                occlusion=[c["occlusion"] for c in run["cases"]],
            )
        )
    players.extend(
        dict(
            **player,
            run=None,
            sourceRun=recorded[0][0],
            playerIndex=idx,
            checkpoint=None,
            macroF1=None,
        )
        for idx, player in enumerate(trace["players"][1:], start=1)
    )
    return dict(
        market=trace["market"],
        firstDecision=trace["firstDecision"],
        profile=trace["profile"],
        profileHash=trace["profileHash"],
        accounting=anchor["accounting"],
        origin=anchor["origin"],
        dataset=anchor["dataset"],
        players=players,
        seed=seed,
        variant=variant,
        missing=[key for key in candidates if key not in {key for key, _ in recorded}],
    )


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
    r.pop("trace", None)
    r["samples"] = [
        dict(index=i, p=c["p"], up=c["next"] > 0, period=c["period"])
        for i, c in enumerate(cases)
    ]
    return r


@router.get("/{study}/{key}/human-evaluation")
def human_evaluation(study: UUID, key: str):
    report(study, key)
    path = root() / str(study) / "explanations" / (key + ".json")
    return read(path) if path.is_file() else None


@router.post("/{study}/{key}/human-evaluation")
def save_human_evaluation(study: UUID, key: str, responses: Responses):
    r = report(study, key)
    try:
        summary = score_responses(responses, r["cases"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    path = root() / str(study) / "explanations" / (key + ".json")
    artifact = dict(
        summary=summary,
        dataset=r["dataset"],
        checkpoint=r["checkpoint"],
        **responses.model_dump(),
    )
    with lock:
        path.parent.mkdir(exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
        temp.replace(path)
    return artifact


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
def explain(
    study: UUID,
    key: str,
    index: int,
    mode: Literal["standard", "scientific"] = "scientific",
    phase: Literal["decision", "outcome"] = "decision",
    player: int = 0,
    output: Literal["facts", "llm"] = "llm",
):
    r = report(study, key)
    obs = observe(r, index)
    if not 0 <= player < len(obs["targets"]):
        raise HTTPException(404, "Unknown player")
    target = obs["targets"][player]
    if phase == "outcome":
        if "trace" not in r:
            raise HTTPException(404, "Recorded engine trace unavailable")
        event_log = r["trace"]["players"][player]
        event = event_log["events"][index]
        outcome = event["execution"]
        before = (
            event_log["initial"]
            if index == 0
            else event_log["events"][index - 1]["portfolio"]
        )
        after = event["portfolio"]
        execution = (
            f"The {event['decision']['order_side']} order filled {outcome['fill_quantity']} units at {outcome['fill_price']:.3f}; fee {outcome['fee']:.2f}."
            if outcome and outcome["fill_quantity"] is not None
            else f"Execution status: {outcome['status'] if outcome else event['decision']['action'].replace('_', ' ')}."
        )
        facts = [
            f"The next adjusted close changed by {math.expm1(r['cases'][index]['next']):+.2%}.",
            execution,
            f"Portfolio value changed by {after['equity']-before['equity']:+,.2f} to {after['equity']:,.2f} in virtual currency, including execution effects.",
        ]
        if mode == "standard":
            change = math.expm1(r["cases"][index]["next"])
            facts[0] = (
                f"The next session's adjusted closing price {'rose' if change > 0 else 'fell' if change < 0 else 'stayed unchanged'} by {abs(change):.2%}."
            )
            if outcome and outcome["fill_quantity"] is not None:
                facts[1] = (
                    f"The simulator {'bought' if event['decision']['order_side'] == 'buy' else 'sold'} {outcome['fill_quantity']} units at {outcome['fill_price']:.3f} each; the recorded fee was {outcome['fee']:.2f}."
                )
    elif player:
        rule = (
            "This player compares average relative prices over the last 20 and 50 closes."
            if player == 1
            else "This player aims to stay invested throughout the replay."
        )
        facts = [
            rule,
            f"The target was {'to be invested' if target == 'long' else 'to hold cash'}.",
            "This rule does not estimate a probability of tomorrow's price rising.",
        ]
    else:
        block = max(range(8), key=lambda j: abs(obs["occlusion"][j]))
        delta = -obs["occlusion"][block]
        if mode == "standard":
            facts = [
                f"The AI estimates a {obs['p']:.1%} chance of the price rising next session and a {1-obs['p']:.1%} chance of it staying unchanged or falling.",
                f"The visible price changed {math.expm1(sum(obs['inputs'][-20:])):+.2%} over the last 20 returns. Its target is {'to be invested' if target == 'long' else 'to hold cash'}. The estimate can be wrong.",
                f"Replacing observations {8*block+1} to {8*block+8} of the 64-session input with the average learned during training changes the rising estimate by {100*delta:+.1f} percentage points, the largest measured change across the eight blocks.",
            ]
        else:
            facts = [
                f"The estimated upward probability was {obs['p']:.1%}.",
                f"The position target was {target}.",
                f"Replacing offsets {8*block-63} to {8*block-56} with the training mean changed P(up) by {delta:+.3f}.",
            ]
    if output == "facts":
        return dict(
            source="Fact sheet",
            check="Measured directly from the recorded decision",
            facts=facts,
        )
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
