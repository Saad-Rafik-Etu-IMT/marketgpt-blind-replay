"""Train-only scalers, serialized as plain JSON (no pickle).

Kinds:

- ``none``: identity.
- ``global_standard``: one mean and one std over every value of the training
  matrix; reproduces the W1 baseline (``step3_normalize.fit_normalizer``).
- ``standard``: per-column mean and population std; a zero std becomes 1.
- ``robust``: per-column median and interquartile range, std then 1 as
  fallbacks, clipped to +-5 after scaling (``deep_mlp.fit_robust_scaler``).

``fit`` receives training rows only; the builder never passes anything else.
For 3-D sequence inputs the statistics are per feature (last axis), pooled
over examples and time steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

SCALE_FLOOR = 1e-7


class ScalerError(ValueError):
    """Invalid scaler definition, input or serialized state."""


class ScalerKind(str, Enum):
    NONE = "none"
    GLOBAL_STANDARD = "global_standard"
    STANDARD = "standard"
    ROBUST = "robust"


@dataclass(frozen=True)
class Scaler:
    kind: ScalerKind
    columns: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float | None = None

    def __post_init__(self) -> None:
        if len(self.center) != len(self.columns) or len(self.scale) != len(
            self.columns
        ):
            raise ScalerError("one center and one scale per column are required")
        values = np.array([*self.center, *self.scale], dtype=np.float64)
        if not np.isfinite(values).all() or min(self.scale, default=1.0) <= 0:
            raise ScalerError("scaler statistics must be finite with positive scales")
        if self.clip is not None and self.clip <= 0:
            raise ScalerError("clip must be positive")

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(self.columns):
            raise ScalerError(
                f"expected {len(self.columns)} features, got {x.shape[-1]}"
            )
        out = (x - np.asarray(self.center)) / np.asarray(self.scale)
        return np.clip(out, -self.clip, self.clip) if self.clip is not None else out

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "columns": list(self.columns),
            "center": list(self.center),
            "scale": list(self.scale),
            "clip": self.clip,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, object], expected_columns: Sequence[str]
    ) -> Scaler:
        """Rebuild a saved scaler; refuses a different feature order or schema."""
        try:
            scaler = cls(
                kind=ScalerKind(data["kind"]),
                columns=tuple(data["columns"]),  # type: ignore[arg-type]
                center=tuple(float(v) for v in data["center"]),  # type: ignore[union-attr]
                scale=tuple(float(v) for v in data["scale"]),  # type: ignore[union-attr]
                clip=None if data.get("clip") is None else float(data["clip"]),  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ScalerError(f"invalid serialized scaler: {exc}") from exc
        if scaler.columns != tuple(expected_columns):
            raise ScalerError(
                "scaler columns differ from the requested feature columns"
            )
        return scaler


def fit_scaler(
    kind: ScalerKind | str, train: np.ndarray, columns: Sequence[str]
) -> Scaler:
    kind = ScalerKind(kind)
    x = np.asarray(train, dtype=np.float64)
    if x.ndim not in (2, 3) or x.shape[-1] != len(columns):
        raise ScalerError(
            "training matrix must be [N, F] or [N, T, F] with F = len(columns)"
        )
    if len(x) == 0:
        raise ScalerError("cannot fit a scaler on zero training rows")
    if not np.isfinite(x).all():
        raise ScalerError("training matrix contains non-finite values")
    flat = x.reshape(-1, x.shape[-1])
    width = flat.shape[1]
    clip = None
    if kind is ScalerKind.NONE:
        center, scale = np.zeros(width), np.ones(width)
    elif kind is ScalerKind.GLOBAL_STANDARD:
        std = float(flat.std())
        center = np.full(width, float(flat.mean()))
        scale = np.full(width, std if std > SCALE_FLOOR else 1.0)
    elif kind is ScalerKind.STANDARD:
        std = flat.std(axis=0)
        center, scale = flat.mean(axis=0), np.where(std > SCALE_FLOOR, std, 1.0)
    else:
        q75, q25 = np.percentile(flat, [75, 25], axis=0)
        iqr, std = q75 - q25, flat.std(axis=0)
        center = np.median(flat, axis=0)
        scale = np.where(iqr > SCALE_FLOOR, iqr, np.where(std > SCALE_FLOOR, std, 1.0))
        clip = 5.0
    return Scaler(
        kind=kind,
        columns=tuple(columns),
        center=tuple(float(v) for v in center),
        scale=tuple(float(v) for v in scale),
        clip=clip,
    )
