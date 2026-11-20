"""Experiments API (tranche T6): catalogue, data sources, cost, lifecycle, results.

Separate from ``/studies``: existing studies, their worker and saved artifacts
are untouched. One experiment worker runs at a time (bounded concurrency).
Identifiers are UUIDs and run keys match a strict pattern; no client value is
ever used as a filesystem path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from importlib.util import find_spec
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from src.ai.mlp import PRESETS, preset
from src.config import BACKEND_DIR, get_settings
from src.data.split import load_split_config
from src.datasets import ContextRule, ScalerKind
from src.evaluation import EvaluationError, evaluate_experiment, read_evaluation
from src.experiments import (
    EXPERIMENT_PROFILES,
    MAX_RUNS,
    SYNTHETIC,
    ExperimentSpec,
    SourceError,
    estimate,
    health,
    load_frames,
    write_spec,
)
from src.experiments.campaigns import campaigns
from src.experiments.holdout import freeze
from src.experiments.replay import replay_run
from src.experiments.reporting import export_report
from src.experiments.sources import SYNTHETIC_SIGNAL
from src.experiments.spec import RUN_KEY, TRANSFORMERS
from src.features import (
    DEFAULT_REGISTRY,
    PROFILES,
    FeatureGroup,
    FeatureRegistryError,
    compute,
)
from src.features.diagnostics import training_rows
from src.services.local_explanation import explain_facts
from src.services.study_artifacts import read_json, write_json
from src.targets import TARGETS

router = APIRouter(prefix="/experiments", tags=["experiments"])
lock = Lock()
worker: subprocess.Popen[bytes] | None = None
active_id: str | None = None
PREVIEW_MAX = 2000


def data_dir() -> Path:
    return get_settings().data_dir


def root() -> Path:
    return data_dir() / "experiments"


def folder_of(experiment: UUID) -> Path:
    folder = root() / str(experiment)
    if not (folder / "spec.json").is_file():
        raise HTTPException(404, "Unknown experiment")
    return folder


def _running() -> bool:
    return worker is not None and worker.poll() is None


def _status(folder: Path) -> dict[str, Any]:
    value = read_json(folder / "status.json")
    if value.get("state") == "running" and (active_id != folder.name or not _running()):
        value = read_json(folder / "status.json")
        if value.get("state") in {"queued", "running"}:
            value = {
                **value,
                "state": "interrupted",
                "message": "Worker stopped; resume to continue",
            }
    return value


# ------------------------------------------------------------------ catalogue
@router.get("/catalog")
def catalog() -> dict[str, Any]:
    transformer = find_spec("torch") is not None
    return {
        "feature_groups": [g.value for g in FeatureGroup],
        "features": [s.describe() for s in DEFAULT_REGISTRY.specs()],
        "feature_profiles": [p.describe() for p in PROFILES.values()],
        "targets": [t.describe() for t in TARGETS.values()],
        "presets": {
            **{
                name: {
                    "architecture": a.model_dump(mode="json"),
                    "training": c.model_dump(mode="json"),
                }
                for name in PRESETS
                for a, c in [preset(name)]
            },
            **(
                {
                    name: {
                        "architecture": "reference attention encoder",
                        "variant": variant,
                    }
                    for name, variant in TRANSFORMERS.items()
                }
                if transformer
                else {}
            ),
        },
        "transformer_available": transformer,
        "experiment_profiles": [vars(p) for p in EXPERIMENT_PROFILES.values()],
        "contexts": [c.value for c in ContextRule],
        "scalers": [s.value for s in ScalerKind],
        "max_runs": MAX_RUNS,
        "campaigns": campaigns(transformer),
    }


@router.get("/features/{feature_id}")
def feature(feature_id: str) -> dict[str, Any]:
    try:
        spec = DEFAULT_REGISTRY.get(feature_id)
    except FeatureRegistryError as exc:
        raise HTTPException(404, "Unknown feature") from exc
    return {
        **spec.describe(),
        "redundancies": DEFAULT_REGISTRY.redundancies(
            [feature_id, *spec.redundant_with]
        ),
    }


@router.get("/sources")
def sources() -> list[dict[str, Any]]:
    return health(data_dir())


@router.get("/sources/{source}/preview")
def preview(
    source: str, feature: str, last: int = Query(default=250, ge=1, le=PREVIEW_MAX)
) -> dict[str, Any]:
    """Feature values up to ``validation_end`` only: the sealed test period is never shown."""
    try:
        spec = DEFAULT_REGISTRY.get(feature)
    except FeatureRegistryError as exc:
        raise HTTPException(404, "Unknown feature") from exc
    try:
        frames = load_frames((source,), data_dir())
    except SourceError as exc:
        raise HTTPException(404, "Unknown source") from exc
    frame = next(iter(frames.values()))
    frame = frame.loc[frame.index <= str(load_split_config().validation_end)]
    values = compute(frame, [feature]).values.iloc[-last:]
    return {
        "source": source,
        "feature": spec.describe(),
        "dates": [d.date().isoformat() for d in values.index],
        "adjusted_close": [
            float(v) if np.isfinite(v) else None
            for v in frame.loc[values.index, "adjusted_close"]
        ],
        "columns": list(values.columns),
        "values": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in values.to_numpy()
        ],
    }


@router.get("/sources/{source}/correlation")
def correlation(source: str, profile: str = "core_features") -> dict[str, Any]:
    """One scalar per feature, measured exclusively on training dates."""
    try:
        selected = PROFILES[profile]
        if not selected.available:
            raise ValueError("Unavailable feature profile")
        frame = next(iter(load_frames((source,), data_dir()).values()))
        split = load_split_config()
        frame = frame.loc[frame.index <= str(split.train_end)]
        representatives = [
            f
            for f in DEFAULT_REGISTRY.resolve(selected.features)
            if len(f.columns) == 1
        ][:12]
        if len(representatives) < 2:
            raise ValueError("Choose a profile with at least two scalar features")
        matrix = compute(frame, [f.id for f in representatives]).values
        rows = training_rows(matrix, split)
        if len(rows) < 3:
            raise ValueError("Insufficient complete training rows")
        values = rows.corr(method="spearman").to_numpy()
        return {
            "columns": list(rows.columns),
            "rows": len(rows),
            "method": "Spearman",
            "matrix": [
                [float(v) if np.isfinite(v) else None for v in row] for row in values
            ],
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


# ----------------------------------------------------------------- lifecycle
def _rows(spec: ExperimentSpec) -> dict[str, int]:
    if spec.sources in {(SYNTHETIC,), (SYNTHETIC_SIGNAL,)}:
        return {spec.sources[0]: 2 * 3912}
    known = {h["source"]: h.get("rows", 0) for h in health(data_dir())}
    missing = [s for s in spec.sources if s not in known]
    if missing:
        raise HTTPException(422, f"Unknown local source(s): {missing}")
    return {s: known[s] for s in spec.sources}


@router.post("/estimate")
def estimate_cost(spec: ExperimentSpec) -> dict[str, Any]:
    return estimate(spec, _rows(spec))


def _launch(folder: Path) -> None:
    global worker, active_id
    with (folder / "worker.log").open("a") as log:
        worker = subprocess.Popen(
            [sys.executable, "-m", "src.experiments.worker", str(folder.resolve())],
            cwd=BACKEND_DIR,
            stdout=log,
            stderr=log,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        )
    active_id = folder.name
    Thread(target=_watch, args=(worker, folder), daemon=True).start()


def _mark_launch_failure(folder: Path) -> None:
    write_json(
        folder / "status.json",
        {
            **read_json(folder / "status.json"),
            "state": "failed",
            "message": "Worker could not start",
        },
    )


def _dispatch_pending() -> str | None:
    """Caller holds lock; only one worker is launched at a time."""
    if _running():
        return None
    for path in sorted(
        root().glob("*/status.json"), key=lambda p: p.stat().st_mtime_ns
    ):
        if read_json(path).get("state") != "queued":
            continue
        if (path.parent / "cancel").exists():
            continue
        try:
            _launch(path.parent)
            return path.parent.name
        except OSError:
            _mark_launch_failure(path.parent)
    return None


def _watch(process: subprocess.Popen[bytes], folder: Path) -> None:
    """Dispatch persistent queued jobs even after the user closes the browser."""
    process.wait()
    with lock:
        if worker is not process:
            return
        status = read_json(folder / "status.json")
        if status.get("state") in {"queued", "running"}:
            write_json(
                folder / "status.json",
                {
                    **status,
                    "state": "interrupted",
                    "message": "Worker stopped; resume to continue",
                },
            )
        _dispatch_pending()


@router.post("/queue/dispatch", status_code=202)
def dispatch() -> dict[str, Any]:
    """Explicitly restart a durable waiting queue after application restart."""
    with lock:
        return {"id": _dispatch_pending()}


@router.post("", status_code=202)
def create(spec: ExperimentSpec, enqueue: bool = False) -> dict[str, Any]:
    if set(spec.presets) & TRANSFORMERS.keys() and find_spec("torch") is None:
        raise HTTPException(
            422, "Install optional PyTorch to enable Transformer campaigns"
        )
    cost = estimate(spec, _rows(spec))
    if not cost["within_budget"]:
        raise HTTPException(
            422,
            f"Estimated {cost['estimated_seconds']} s exceeds the {cost['budget_seconds']} s budget",
        )
    with lock:
        if _running() and not enqueue:
            raise HTTPException(409, "An experiment is already running")
        folder = root() / str(uuid4())
        write_spec(folder, spec)
        try:
            if not _running():
                _launch(folder)
        except OSError as exc:
            _mark_launch_failure(folder)
            raise HTTPException(503, "Experiment worker could not start") from exc
    return {"id": folder.name, "estimate": cost}


@router.get("")
def list_experiments() -> list[dict[str, Any]]:
    out = []
    if not root().is_dir():
        return out
    for path in sorted(
        root().glob("*/spec.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        folder = path.parent
        stored = read_json(path)
        out.append(
            {
                "id": folder.name,
                "name": stored["spec"]["name"],
                "profile": stored["spec"]["profile"],
                **_status(folder),
            }
        )
    return out


@router.get("/{experiment}")
def detail(experiment: UUID) -> dict[str, Any]:
    folder = folder_of(experiment)
    stored = read_json(folder / "spec.json")
    return {"id": folder.name, **stored, "status": _status(folder)}


@router.post("/{experiment}/cancel")
def cancel(experiment: UUID) -> dict[str, Any]:
    folder = folder_of(experiment)
    (folder / "cancel").write_text("requested", encoding="utf-8")
    with lock:
        if active_id != folder.name or not _running():
            status = read_json(folder / "status.json")
            if status.get("state") in {"queued", "running"}:
                write_json(
                    folder / "status.json",
                    {**status, "state": "cancelled", "message": "cancelled by request"},
                )
    return _status(folder)


@router.post("/{experiment}/resume", status_code=202)
def resume(experiment: UUID, enqueue: bool = False) -> dict[str, Any]:
    folder = folder_of(experiment)
    with lock:
        if _running() and (not enqueue or active_id == folder.name):
            raise HTTPException(409, "An experiment is already running")
        if _status(folder).get("state") == "complete":
            raise HTTPException(409, "Experiment already complete")
        (folder / "cancel").unlink(missing_ok=True)
        write_json(
            folder / "status.json",
            {
                **read_json(folder / "status.json"),
                "state": "queued",
                "message": "Waiting to resume",
            },
        )
        try:
            if not _running():
                _launch(folder)
        except OSError as exc:
            _mark_launch_failure(folder)
            raise HTTPException(503, "Experiment worker could not start") from exc
    return {"id": folder.name}


# ------------------------------------------------------------------- results
def _reports(folder: Path) -> list[dict[str, Any]]:
    runs = folder / "runs"
    if not runs.is_dir():
        return []
    return [read_json(p) for p in sorted(runs.glob("*.json")) if RUN_KEY.match(p.stem)]


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "key",
        "state",
        "feature_profile",
        "preset",
        "seed",
        "parameter_count",
        "best_epoch",
        "epochs_run",
        "stopped_early",
        "seconds",
        "error",
    )
    out = {k: report[k] for k in keep if k in report}
    out["inner_validation"] = report.get("inner_validation")
    if "validation" in report:
        out["validation"] = report["validation"]
    return out


@router.get("/{experiment}/runs")
def runs(experiment: UUID) -> list[dict[str, Any]]:
    return [_summary(r) for r in _reports(folder_of(experiment))]


@router.get("/{experiment}/datasets")
def dataset_manifests(experiment: UUID) -> list[dict[str, Any]]:
    folder = folder_of(experiment)
    return [read_json(p) for p in sorted((folder / "datasets").glob("*.json"))]


@router.get("/{experiment}/runs/{key}")
def run_report(experiment: UUID, key: str) -> dict[str, Any]:
    if not RUN_KEY.match(key):
        raise HTTPException(404, "Unknown run")
    path = folder_of(experiment) / "runs" / f"{key}.json"
    if not path.is_file():
        raise HTTPException(404, "Unknown run")
    return read_json(path)


@router.get("/{experiment}/runs/{key}/progress")
def run_progress(experiment: UUID, key: str) -> dict[str, Any]:
    folder = folder_of(experiment)
    if not RUN_KEY.fullmatch(key):
        raise HTTPException(404, "Unknown run")
    report = folder / "runs" / f"{key}.json"
    if report.is_file() and read_json(report).get("state") == "complete":
        return {"history": read_json(report).get("history", [])}
    path = folder / "progress" / f"{key}.json"
    if path.is_file():
        return read_json(path)
    return {"history": read_json(report).get("history", []) if report.is_file() else []}


@router.post("/{experiment}/runs/{key}/replay")
def replay(
    experiment: UUID,
    key: str,
    part: str = Query(
        default="inner_validation", pattern="^(inner_validation|validation)$"
    ),
    asset: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$"),
) -> dict[str, Any]:
    folder = folder_of(experiment)
    if not RUN_KEY.fullmatch(key):
        raise HTTPException(404, "Unknown run")
    with lock:
        if _running():
            raise HTTPException(409, "Wait until the training worker finishes")
        try:
            return replay_run(folder, data_dir(), key, part, asset)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc)) from exc


def _recorded_replay(
    experiment: UUID, key: str, part: str, asset: str | None = None
) -> dict[str, Any]:
    folder = folder_of(experiment)
    if not RUN_KEY.fullmatch(key) or part not in {"inner_validation", "validation"}:
        raise HTTPException(404, "Unknown replay")
    paths = sorted((folder / "replays").glob(f"{key}__{part}__*.json"))
    if asset is not None:
        paths = [p for p in paths if p.name == f"{key}__{part}__{asset}.json"]
    if not paths:
        raise HTTPException(404, "Generate the replay first")
    return read_json(paths[0])


@router.get("/{experiment}/runs/{key}/replay")
def recorded_replay(
    experiment: UUID,
    key: str,
    part: str = "inner_validation",
    asset: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$"),
) -> dict[str, Any]:
    return _recorded_replay(experiment, key, part, asset)


@router.post("/{experiment}/runs/{key}/replay/explain")
def replay_explanation(
    experiment: UUID,
    key: str,
    index: int = Query(ge=0),
    part: str = "inner_validation",
    outcome: bool = False,
    generate: bool = False,
    asset: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$"),
) -> dict[str, Any]:
    artifact = _recorded_replay(experiment, key, part, asset)
    player = artifact["players"][0]
    if index >= len(player["events"]):
        raise HTTPException(404, "Unknown decision")
    event = player["events"][index]
    decision = event["decision"]
    p = decision["probabilities"]["long"]
    facts = [
        f"The model assigned a {p:.1%} probability to a positive next-open to following-open return.",
        f"The fixed threshold policy selected {decision['target']} (threshold: 50%).",
        f"The engine recorded action {decision['action']}.",
    ]
    if outcome:
        facts.extend(
            [
                f"The next-close portfolio equity was {event['portfolio']['equity']:,.2f} virtual currency units.",
                f"Execution status: {event['execution']['status'] if event['execution'] else 'no order'}.",
            ]
        )
    return (
        explain_facts(facts)
        if generate and not outcome
        else {
            "source": "Measured facts",
            "check": "Recorded evidence; not a causal explanation",
            "facts": facts,
        }
    )


class HoldoutPlan(BaseModel):
    key: str
    start: date
    end: date


@router.post("/{experiment}/holdout/lock", status_code=201)
def lock_holdout(experiment: UUID, plan: HoldoutPlan) -> dict[str, Any]:
    with lock:
        if _running():
            raise HTTPException(409, "Wait until the worker finishes")
        try:
            return freeze(
                folder_of(experiment), data_dir(), plan.key, plan.start, plan.end
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc


@router.get("/{experiment}/holdout")
def holdout(experiment: UUID) -> dict[str, Any]:
    folder = folder_of(experiment)
    path = folder / "holdout_plan.json"
    return {
        "plan": read_json(path) if path.is_file() else None,
        "consumed": (folder / "holdout_consumed.json").is_file(),
        "result_available": (folder / "holdout_result.json").is_file(),
    }


@router.get("/{experiment}/export/{kind}")
def export(experiment: UUID, kind: str) -> Response:
    try:
        body, media = export_report(folder_of(experiment), kind)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(
        body,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="experiment-{experiment}.{kind}"'
        },
    )


def _numbers(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, dict):
            flat.update(_numbers(value, f"{prefix}{name}."))
        elif (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and name != "count"
        ):
            flat[f"{prefix}{name}"] = float(value)
    return flat


@router.get("/{experiment}/compare")
def compare(
    experiment: UUID,
    part: str = Query(
        default="inner_validation", pattern="^(inner_validation|validation)$"
    ),
) -> list[dict[str, Any]]:
    """Mean, std, min and max over seeds of every numeric metric, per (feature profile, preset)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for report in _reports(folder_of(experiment)):
        if report.get("state") == "complete" and part in report:
            groups.setdefault((report["feature_profile"], report["preset"]), []).append(
                report
            )
    out = []
    for (fp, ps), reports in sorted(groups.items()):
        flats = [_numbers(r[part]) for r in reports]
        names = sorted(set().union(*flats))
        stats = {}
        for name in names:
            values = np.array([f[name] for f in flats if name in f], dtype=float)
            stats[name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "min": float(values.min()),
                "max": float(values.max()),
                "n": int(len(values)),
            }
        out.append(
            {
                "feature_profile": fp,
                "preset": ps,
                "seeds": sorted(r["seed"] for r in reports),
                "metrics": stats,
            }
        )
    return out


# ---------------------------------------------------------------- evaluation
@router.post("/{experiment}/evaluate")
def evaluate(
    experiment: UUID,
    part: str = Query(
        default="inner_validation", pattern="^(inner_validation|validation)$"
    ),
) -> dict[str, Any]:
    """Synchronous T7 evaluation; ``part=validation`` is logged in the validation-read ledger."""
    folder = folder_of(experiment)
    with lock:
        if _running():
            raise HTTPException(409, "Wait until the training worker finishes")
        try:
            report = evaluate_experiment(folder, data_dir(), part)
        except EvaluationError as exc:
            raise HTTPException(409, str(exc)) from exc
    return {
        "part": part,
        "summary": report["ablation"]["summary"],
        "results": len(report["results"]),
    }


@router.get("/{experiment}/evaluation/{part}")
def evaluation(experiment: UUID, part: str) -> dict[str, Any]:
    folder = folder_of(experiment)
    try:
        return read_evaluation(folder, part)
    except (EvaluationError, FileNotFoundError) as exc:
        raise HTTPException(404, "No evaluation for this part") from exc
