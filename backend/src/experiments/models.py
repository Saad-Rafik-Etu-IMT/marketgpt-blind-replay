"""Registered-model adapter for the existing reference attention encoder."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.ai.mlp import load
from src.ai.mlp.checkpoint import LoadedModel
from src.datasets.manifest import code_revision
from src.datasets.preprocessing import Scaler
from src.experiments.spec import TRANSFORMERS
from src.services.study_artifacts import read_json, write_json

TARGET = "direction_binary_v1"
COMPATIBILITY = "transformer-reference-v1"


class TransformerPredictor:
    def __init__(self, predict, parameters: int):
        self._predict = predict
        self.parameter_count = parameters
        self.n_in = 64

    def predict(self, x):
        if len(x) == 0:
            return {TARGET: np.empty((0, 2))}
        p = np.concatenate(
            [self._predict(x[i : i + 256]) for i in range(0, len(x), 256)]
        )
        return {TARGET: np.log(np.clip(np.column_stack([1 - p, p]), 1e-12, 1.0))}


@dataclass
class TrainedTransformer:
    model: TransformerPredictor
    history: list[dict[str, Any]]
    best_epoch: int
    stopped_early: bool
    device: str


def train_reference(data, config, candidate, checkpoint: Path, device: str, progress):
    from learning.mlp.transformer_reference import fit_transformer, resolve_device

    resolved = str(resolve_device(device))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def record(h):
        if progress:
            progress({**h, "select_loss": h["val_loss"]})

    predict, history, parameters = fit_transformer(
        data.x_train,
        data.y_train[TARGET],
        data.x_select,
        data.y_select[TARGET],
        config.seed,
        TRANSFORMERS[candidate],
        config.epochs,
        checkpoint,
        device=resolved,
        on_epoch=record,
        patience=config.patience,
    )
    converted = [{**h, "select_loss": h["val_loss"]} for h in history]
    return TrainedTransformer(
        TransformerPredictor(predict, parameters),
        converted,
        next(h["epoch"] for h in converted if h["selected"]),
        len(converted) < config.epochs,
        resolved,
    )


def register_transformer(path, trained, config, datasets, candidate):
    ds = datasets[0]
    weights = path.with_suffix(".pt")
    write_json(
        path.with_suffix(".json"),
        {
            "schema_version": 1,
            "compatibility": COMPATIBILITY,
            "architecture": {
                "encoder": "reference",
                "variant": TRANSFORMERS[candidate],
                "embedding": 16,
                "heads": 2,
            },
            "training": config.model_dump(mode="json"),
            "device": trained.device,
            "code_revision": code_revision(),
            "dataset_fingerprint": ds.fingerprint,
            "features": {"columns": list(ds.columns), "ids": list(ds.feature_ids)},
            "scaler": ds.scaler.to_dict(),
            "targets": {TARGET: {"state": ds.target_state or {}}},
            "history": trained.history,
            "best_epoch": trained.best_epoch,
            "parameter_count": trained.model.parameter_count,
            "n_in": 64,
            "weights": {
                "file": weights.name,
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            },
        },
    )


def load_registered(path: Path, expected_columns=None):
    manifest = read_json(path.with_suffix(".json"))
    if manifest.get("compatibility") != COMPATIBILITY:
        return load(path, expected_columns)
    columns = tuple(manifest["features"]["columns"])
    if expected_columns is not None and columns != tuple(expected_columns):
        raise ValueError("Feature columns differ from registered Transformer")
    weights = path.with_suffix(".pt")
    if (
        hashlib.sha256(weights.read_bytes()).hexdigest()
        != manifest["weights"]["sha256"]
    ):
        raise ValueError("Transformer checkpoint hash does not match manifest")
    from learning.mlp.transformer_reference import load_predictor

    model = TransformerPredictor(load_predictor(weights), manifest["parameter_count"])
    return LoadedModel(
        model=model,
        scaler=Scaler.from_dict(manifest["scaler"], columns),
        columns=columns,
        target_states={TARGET: manifest["targets"][TARGET]["state"]},
        manifest=manifest,
    )
