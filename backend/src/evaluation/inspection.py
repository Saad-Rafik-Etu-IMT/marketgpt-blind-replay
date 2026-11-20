"""Checkpoint-backed development prediction inspection.

No training, threshold promotion, portfolio execution or final holdout read.
"""

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.evaluate import (
    PARTS,
    EvaluationError,
    read_evaluation,
    regime_labels,
)
from src.experiments.models import load_registered
from src.experiments.runner import Runner
from src.experiments.sources import load_frames
from src.services.study_artifacts import read_json
from src.targets import TargetKind, softmax


def _context(folder: Path, data_dir: Path, key: str, part: str, target: str):
    if part not in PARTS:
        raise EvaluationError("Only development partitions are inspectable")
    if not (folder / "evaluation" / f"{part}.json").is_file():
        raise EvaluationError("Generate the development evaluation first")
    report = read_json(folder / "runs" / f"{key}.json")
    if report["state"] != "complete" or target not in report["dataset_fingerprints"]:
        raise EvaluationError("Completed checkpoint and matching target required")
    evidence = read_evaluation(folder, part)
    if evidence.get("spec_fingerprint") != report["spec_fingerprint"] or not any(
        e["feature_profile"] == report["feature_profile"]
        and e["preset"] == report["preset"]
        and e["target"] == target
        and report["seed"] in e["seeds"]
        for e in evidence["results"]
    ):
        raise EvaluationError(
            "Generate matching evidence for this registered checkpoint"
        )
    runner = Runner(folder, data_dir)
    frames = load_frames(runner.spec.sources, data_dir)
    runner.verify_data(frames)
    datasets = runner._datasets(frames, report["feature_profile"])  # noqa: SLF001
    dataset = next(ds for ds in datasets if ds.target.id == target)
    if dataset.fingerprint != report["dataset_fingerprints"][target]:
        raise EvaluationError("Dataset changed since training")
    if (
        dataset.target.kind is not TargetKind.CLASSIFICATION
        or dataset.target.output_dim != 2
    ):
        raise EvaluationError("Binary classification targets only")
    loaded = load_registered(
        folder / report["checkpoint"], expected_columns=dataset.columns
    )
    if part == "validation":
        runner.record_validation_read("prediction_inspection", dataset.fingerprint)
    return runner, frames, dataset, loaded


def prediction_cases(
    folder: Path, data_dir: Path, key: str, part: str, target: str
) -> dict[str, Any]:
    runner, frames, dataset, loaded = _context(folder, data_dir, key, part, target)
    p = dataset.parts[part]
    probs = softmax(loaded.model.predict(p.x)[target])[:, 1]
    # Regime labels must align with the selected target's own index.
    regimes = regime_labels([dataset], part, frames, runner.split)
    return {
        "run": key,
        "target": target,
        "part": part,
        "dataset": dataset.fingerprint,
        "cases": [
            {
                "index": i,
                "asset": str(row.asset),
                "date": str(row.date)[:10],
                "bar": int(row.bar),
                "probability": float(probs[i]),
                "actual": int(p.y[i]),
                "volatility": str(regimes["volatility_tercile"][i]),
                "trend": str(regimes["trend_60d"][i]),
            }
            for i, row in enumerate(p.index.itertuples(index=False))
        ],
    }


def inspect_prediction(
    folder: Path,
    data_dir: Path,
    key: str,
    part: str,
    target: str,
    index: int,
    column: int | None = None,
    replacement: float = 0.0,
) -> dict[str, Any]:
    _, _, dataset, loaded = _context(folder, data_dir, key, part, target)
    p = dataset.parts[part]
    if not 0 <= index < len(p):
        raise EvaluationError("Unknown development observation")
    x = p.x[index : index + 1].copy()
    count = len(dataset.columns)
    # Sequential replacement in registered column order. The deltas telescope
    # exactly, but depend on ordering and do NOT imply causal importance.
    changes = np.zeros((count + 1, *x.shape[1:]), dtype=x.dtype)
    for i in range(1, count + 1):
        changes[i, ..., :i] = x[0, ..., :i]
    probs = softmax(loaded.model.predict(changes)[target])[:, 1]
    features = [
        {
            "column": name,
            "raw": float(p.x_raw[index, ..., i].reshape(-1)[-1]),
            "scaled": float(x[0, ..., i].reshape(-1)[-1]),
            "delta": float(probs[i + 1] - probs[i]),
        }
        for i, name in enumerate(dataset.columns)
    ]
    result: dict[str, Any] = {
        "index": index,
        "dataset": dataset.fingerprint,
        "probability": float(probs[-1]),
        "reference": float(probs[0]),
        "features": features,
        "sequence": x.ndim == 3,
        "method": "Sequential replacement from zero in model-input scale; order-dependent, non-causal.",
    }
    if column is not None:
        if (
            not 0 <= column < count
            or not np.isfinite(replacement)
            or abs(replacement) > 3
        ):
            raise EvaluationError("Counterfactual outside supported bounds")
        modified = x.copy()
        modified[..., column] = replacement
        result["counterfactual"] = {
            "column": dataset.columns[column],
            "replacement": replacement,
            "probability": float(softmax(loaded.model.predict(modified)[target])[0, 1]),
        }
    if hasattr(loaded.model, "forward"):
        _, cache = loaded.model.forward(x, training=False)
        result["activations"] = [
            {"layer": i + 1, "values": h[0, :64].tolist(), "width": h.shape[1]}
            for i, h in enumerate(
                [b["h_in"] for b in cache["blocks"][1:]] + [cache["h"]]
            )
        ]
    return result
