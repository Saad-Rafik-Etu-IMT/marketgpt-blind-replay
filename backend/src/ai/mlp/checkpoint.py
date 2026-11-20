"""Safe model artifacts: ``<name>.npz`` weights + ``<name>.json`` manifest.

Weights are plain float arrays loaded with ``allow_pickle=False``: no code is
ever deserialized. The manifest records everything needed to audit and reuse
the model: architecture and training configuration, seed, feature ids,
versions and column order, scaler, fitted target states, dataset fingerprint
and split, history, best epoch, code revision and a SHA-256 of the weights.

``load`` refuses a manifest from another schema, weights whose hash or shapes
differ from the manifest, and (when given) a feature column order that is not
exactly the one the model was trained on. Inputs are never reordered,
dropped or substituted.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.ai.mlp.config import ArchitectureConfig, TrainConfig
from src.ai.mlp.network import MLP, Params
from src.datasets.manifest import code_revision
from src.datasets.preprocessing import Scaler

SCHEMA_VERSION = 1
COMPATIBILITY = "mlp-v1"


class CheckpointError(ValueError):
    """Missing, tampered or incompatible model artifact."""


def weights_digest(params: Params) -> str:
    digest = hashlib.sha256()
    for name in sorted(params):
        array = np.ascontiguousarray(params[name], dtype=np.float64)
        digest.update(name.encode() + str(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedModel:
    model: MLP
    scaler: Scaler
    columns: tuple[str, ...]
    target_states: dict[str, dict[str, Any]]
    manifest: dict[str, Any]

    def predict_raw_features(self, x_raw: np.ndarray) -> dict[str, np.ndarray]:
        """Scale raw feature rows with the training scaler, then predict raw head outputs."""
        return self.model.predict(self.scaler.transform(x_raw))

    def decode_raw_features(
        self, x_raw: np.ndarray
    ) -> dict[str, dict[str, np.ndarray]]:
        return self.model.decode(self.scaler.transform(x_raw), self.target_states)


def save(
    path: Path,
    model: MLP,
    train_config: TrainConfig,
    *,
    dataset: Any,
    target_states: dict[str, dict[str, Any]],
    history: list[dict[str, Any]],
    best_epoch: int,
) -> dict[str, Any]:
    """Write ``path.npz`` and ``path.json``; returns the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = weights_digest(model.params)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "compatibility": COMPATIBILITY,
        "model_id": str(uuid.UUID(digest[:32])),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_revision": code_revision(),
        "architecture": model.config.model_dump(mode="json"),
        "training": train_config.model_dump(mode="json"),
        "seed": train_config.seed,
        "n_in": model.n_in,
        "parameter_count": model.parameter_count,
        "features": {
            "ids": list(dataset.feature_ids),
            "versions": dataset.manifest["features"]["versions"],
            "columns": list(dataset.columns),
            "catalogue_fingerprint": dataset.manifest["features"][
                "catalogue_fingerprint"
            ],
        },
        "representation": dataset.spec.representation.value,
        "sequence_length": dataset.spec.sequence_length,
        "scaler": dataset.scaler.to_dict(),
        "targets": {t: {"state": s} for t, s in target_states.items()},
        "dataset_fingerprint": dataset.fingerprint,
        "split": dataset.manifest["split"],
        "selection": dataset.manifest["selection"],
        "history": history,
        "best_epoch": best_epoch,
        "weights": {
            "file": path.with_suffix(".npz").name,
            "sha256": digest,
            "shapes": {k: list(v.shape) for k, v in model.params.items()},
        },
    }
    np.savez(
        path.with_suffix(".npz"),
        **{k: np.asarray(v, dtype=np.float64) for k, v in model.params.items()},
    )
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json) + "\n",
        encoding="utf-8",
    )
    return manifest


def _json(value: object) -> object:
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def load(
    path: Path, expected_columns: tuple[str, ...] | list[str] | None = None
) -> LoadedModel:
    try:
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(
            f"cannot read manifest {path.with_suffix('.json')}: {exc}"
        ) from exc
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("compatibility") != COMPATIBILITY
    ):
        raise CheckpointError("unsupported model artifact version")
    columns = tuple(manifest["features"]["columns"])
    if expected_columns is not None and tuple(expected_columns) != columns:
        raise CheckpointError(
            "feature columns differ from the model's training columns (order included)"
        )
    try:
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            params = {k: archive[k].astype(np.float64) for k in archive.files}
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"cannot read weights: {exc}") from exc
    if weights_digest(params) != manifest["weights"]["sha256"]:
        raise CheckpointError("weights do not match the manifest hash")
    architecture = ArchitectureConfig.model_validate(manifest["architecture"])
    model = MLP(int(manifest["n_in"]), architecture)
    expected_shapes = {k: tuple(v.shape) for k, v in model.params.items()}
    if expected_shapes != {k: tuple(v.shape) for k, v in params.items()}:
        raise CheckpointError("weight shapes do not match the architecture")
    model.params = params
    scaler = Scaler.from_dict(manifest["scaler"], columns)
    states = {t: v["state"] or {} for t, v in manifest["targets"].items()}
    return LoadedModel(
        model=model,
        scaler=scaler,
        columns=columns,
        target_states=states,
        manifest=manifest,
    )
