"""Dataset manifest: everything needed to audit and rebuild a dataset.

The ``dataset_fingerprint`` covers the input data, feature versions, label,
split, selection rules, scaler and the exact arrays of every part. It ignores
the creation time and the code revision, so the same inputs rebuilt by
different code commits produce the same fingerprint when the arrays agree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import BACKEND_DIR
from src.features.canonical import RAW_FIELDS

if TYPE_CHECKING:
    from src.data.split import SplitConfig
    from src.datasets.builder import DatasetSpec, Part
    from src.datasets.labels import LabelSpec
    from src.datasets.preprocessing import Scaler
    from src.features.registry import FeatureRegistry
    from src.targets import Target

SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """A manifest is unreadable, from another schema or inconsistent."""


def frame_fingerprint(canonical: pd.DataFrame) -> str:
    """SHA-256 of the dates and raw OHLCV + adjusted close actually used."""
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.index.asi8, dtype=np.int64).tobytes())
    digest.update(canonical[list(RAW_FIELDS)].to_numpy(dtype=np.float64).tobytes())
    return digest.hexdigest()


def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str(a.dtype).encode() + str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def code_revision() -> str:
    """Current git commit of the repository, or ``unknown`` outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BACKEND_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _json_default(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def build_manifest(
    *,
    spec: DatasetSpec,
    config: SplitConfig,
    registry: FeatureRegistry,
    feature_ids: tuple[str, ...],
    columns: tuple[str, ...],
    lookback: int,
    label: LabelSpec,
    cutoff: date,
    canonical: dict[str, pd.DataFrame],
    parts: dict[str, Part],
    dropped: dict[str, int],
    scaler: Scaler,
    target: Target | None = None,
    target_state: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "spec": spec.model_dump(mode="json"),
        "data": {
            name: {
                "fingerprint": frame_fingerprint(frame),
                "rows": len(frame),
                "first": frame.index[0].date().isoformat(),
                "last": frame.index[-1].date().isoformat(),
            }
            for name, frame in canonical.items()
        },
        "features": {
            "requested": list(feature_ids),
            "versions": {s.id: s.version for s in registry.resolve(feature_ids)},
            "columns": list(columns),
            "catalogue_fingerprint": registry.fingerprint(feature_ids),
            "redundancy_warnings": registry.redundancies(feature_ids),
        },
        "label": None if target is not None else label.describe(),
        "target": (
            None
            if target is None
            else {**target.describe(), "fitted_state": target_state}
        ),
        "split": config.model_dump(mode="json"),
        "selection": {
            "context": spec.context.value,
            "effective_lookback": lookback,
            "label_horizon": label.horizon if target is None else target.horizon_bars,
            "embargo": config.embargo,
            "inner_cutoff": cutoff.isoformat(),
            "inner_fraction": spec.inner_fraction,
            "test_included": spec.include_test,
        },
        "parts": {
            name: {
                "rows": len(part),
                "up_share": float(part.y.mean()) if len(part) else None,
                "per_asset": {
                    str(k): int(v)
                    for k, v in part.index["asset"].value_counts().sort_index().items()
                },
                "first": part.index["date"].min().isoformat() if len(part) else None,
                "last": part.index["date"].max().isoformat() if len(part) else None,
                "dropped_undefined": dropped[name],
                "arrays": array_fingerprint(
                    part.x_raw,
                    part.label_values,
                    *(
                        []
                        # T3 datasets (no target) keep their T3 fingerprint.
                        if target is None or part.targets is None
                        else [part.targets, np.asarray(part.y, dtype=np.float64)]
                    ),
                ),
            }
            for name, part in parts.items()
        },
        "scaler": scaler.to_dict(),
    }
    body["dataset_fingerprint"] = hashlib.sha256(
        _canonical_json(body).encode()
    ).hexdigest()
    body["code_revision"] = code_revision()
    body["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return body


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(path: Path) -> dict[str, object]:
    """Load and integrity-check a manifest (schema and fingerprint)."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ManifestError("unsupported manifest schema version")
    body = {
        k: v
        for k, v in manifest.items()
        if k not in ("dataset_fingerprint", "code_revision", "created_at")
    }
    if hashlib.sha256(_canonical_json(body).encode()).hexdigest() != manifest.get(
        "dataset_fingerprint"
    ):
        raise ManifestError("manifest content does not match its fingerprint")
    return manifest
