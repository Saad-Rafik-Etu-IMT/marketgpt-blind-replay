"""Point-in-time feature engine: batch and replay-time computation.

Guarantee: the value at row ``t`` depends only on bars ``0 .. t``. The engine
enforces the parts of this it can check mechanically:

- a feature only receives the source fields it declares;
- warm-up rows (``t < lookback``) are masked to NaN, never filled;
- non-finite values after warm-up and hard-bound violations are refused;
- ``vector_at`` recomputes from bars ``0 .. t`` only, which is exactly what
  replay sees, and the test suite checks it equals the batch row ``t``.

The remaining part (no feature reads future rows) is verified by the generic
future-mutation tests in ``tests/test_feature_point_in_time.py``, which run on
every registered feature.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.canonical import to_canonical
from src.features.registry import FeatureRegistry
from src.features.spec import FeatureSpec, MissingPolicy


class FeatureEngineError(ValueError):
    """A feature broke its declared contract or the request is not allowed."""


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature values for one asset, one row per bar, columns in request order."""

    values: pd.DataFrame
    feature_ids: tuple[str, ...]
    columns: tuple[str, ...]
    lookback: int
    catalogue_fingerprint: str

    @property
    def ready(self) -> pd.Series:
        """True where every column is defined (warm-up done, no undefined value)."""
        return self.values.notna().all(axis=1)

    def fingerprint(self) -> str:
        """SHA-256 of the exact values, NaN positions, dates and column order."""
        array = self.values.to_numpy(dtype=np.float64)
        missing = np.isnan(array)
        digest = hashlib.sha256()
        digest.update(json.dumps(list(self.columns)).encode("utf-8"))
        digest.update(np.asarray(self.values.index.asi8, dtype=np.int64).tobytes())
        digest.update(np.packbits(missing).tobytes())
        digest.update(np.where(missing, 0.0, array).tobytes())
        return digest.hexdigest()


def _default_registry() -> FeatureRegistry:
    from src.features.catalog import DEFAULT_REGISTRY

    return DEFAULT_REGISTRY


def _check_output(
    spec: FeatureSpec, result: object, index: pd.DatetimeIndex
) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame):
        raise FeatureEngineError(f"{spec.id}: compute must return a DataFrame")
    if not result.index.equals(index):
        raise FeatureEngineError(f"{spec.id}: output index differs from the input bars")
    if tuple(result.columns) != spec.columns:
        raise FeatureEngineError(
            f"{spec.id}: output columns {list(result.columns)} != declared {list(spec.columns)}"
        )
    out = result.astype(np.float64)
    out.iloc[: spec.lookback] = np.nan
    after = out.iloc[spec.lookback :].to_numpy()
    if np.isinf(after).any():
        raise FeatureEngineError(f"{spec.id}: infinite value after warm-up")
    if spec.missing_policy is MissingPolicy.WARMUP_NAN and np.isnan(after).any():
        row = int(np.argwhere(np.isnan(after))[0][0]) + spec.lookback
        raise FeatureEngineError(
            f"{spec.id}: undefined value after warm-up at {index[row].date()}"
        )
    low, high = spec.hard_bounds
    finite = after[np.isfinite(after)]
    if (low is not None and (finite < low).any()) or (
        high is not None and (finite > high).any()
    ):
        raise FeatureEngineError(
            f"{spec.id}: value outside hard bounds {spec.hard_bounds}"
        )
    return out


def compute(
    frame: pd.DataFrame,
    feature_ids: Sequence[str],
    registry: FeatureRegistry | None = None,
    *,
    blind: bool = False,
) -> FeatureMatrix:
    """Compute the requested features for every bar of one asset.

    ``blind=True`` refuses any feature (or dependency) that is not blind-safe.
    """
    registry = registry if registry is not None else _default_registry()
    feature_ids = tuple(feature_ids)
    if not feature_ids:
        raise FeatureEngineError("no feature requested")
    resolved = registry.resolve(feature_ids)
    if blind:
        unsafe = [s.id for s in resolved if not s.blind_safe]
        if unsafe:
            raise FeatureEngineError(f"not allowed in blind mode: {unsafe}")
    bars = to_canonical(frame)
    outputs: dict[str, pd.DataFrame] = {}
    for spec in resolved:
        inputs = bars[list(spec.required_fields)].copy()
        dependencies = {d: outputs[d].copy() for d in spec.dependencies}
        outputs[spec.id] = _check_output(
            spec, spec.compute(inputs, dependencies), bars.index
        )
    values = pd.concat([outputs[f] for f in feature_ids], axis=1)
    return FeatureMatrix(
        values=values,
        feature_ids=feature_ids,
        columns=tuple(values.columns),
        lookback=registry.lookback(feature_ids),
        catalogue_fingerprint=registry.fingerprint(feature_ids),
    )


def vector_at(
    frame: pd.DataFrame,
    feature_ids: Sequence[str],
    position: int,
    registry: FeatureRegistry | None = None,
    *,
    blind: bool = False,
) -> pd.Series:
    """Replay-time vector at bar ``position``, computed from bars ``0 .. position`` only."""
    if not 0 <= position < len(frame):
        raise FeatureEngineError(f"position {position} outside 0..{len(frame) - 1}")
    visible = frame.iloc[: position + 1]
    return compute(visible, feature_ids, registry, blind=blind).values.iloc[-1]
