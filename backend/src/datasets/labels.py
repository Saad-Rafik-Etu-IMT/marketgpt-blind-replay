"""Labels for decision bar ``t`` (decided at the close of bar ``t``).

A label reads only bars ``t + 1 .. t + horizon``; the example selector checks
those bars lie inside the example's own partition. Tranche T3 provides the two
labels the project already uses; the full target registry (three classes,
regression, quantiles, volatility, multi-horizon) is tranche T4.

- ``executed_return_v1`` (default): what a long decision earns in the replay,
  filled at the next open and marked at the following open,
  ``log(adj_open[t+2] / adj_open[t+1])``. Same as
  ``learning/mlp/protocol.py::executed_returns``.
- ``close_to_close_v1``: ``log(adj_close[t+1] / adj_close[t])``, the W1 label.

The binary class is ``up = value > 0``; a zero return is "not up".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


class LabelError(ValueError):
    """Unknown label or invalid label definition."""


@dataclass(frozen=True)
class LabelSpec:
    id: str
    version: str
    description: str
    definition: str
    horizon: int
    compute: Callable[[pd.DataFrame], np.ndarray]

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise LabelError(f"{self.id}: horizon must be >= 1 bar")

    def values(self, canonical: pd.DataFrame) -> np.ndarray:
        """Label value per bar; NaN where bars ``t + 1 .. t + horizon`` are missing."""
        out = np.asarray(self.compute(canonical), dtype=np.float64)
        if out.shape != (len(canonical),):
            raise LabelError(f"{self.id}: one value per bar expected")
        out[len(out) - self.horizon :] = np.nan
        return out

    def describe(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "definition": self.definition,
            "horizon": self.horizon,
            "binary_class": "up = value > 0",
        }


def _executed(bars: pd.DataFrame) -> np.ndarray:
    opens = bars["adj_open"].to_numpy()
    out = np.full(len(opens), np.nan)
    out[:-2] = np.log(opens[2:] / opens[1:-1])
    return out


def _close_to_close(bars: pd.DataFrame) -> np.ndarray:
    close = bars["adj_close"].to_numpy()
    out = np.full(len(close), np.nan)
    out[:-1] = np.log(close[1:] / close[:-1])
    return out


LABELS: dict[str, LabelSpec] = {
    spec.id: spec
    for spec in (
        LabelSpec(
            id="executed_return_v1",
            version="1.0.0",
            description="Return a long decision earns in the replay: next open to following open.",
            definition="log(adj_open[t+2] / adj_open[t+1])",
            horizon=2,
            compute=_executed,
        ),
        LabelSpec(
            id="close_to_close_v1",
            version="1.0.0",
            description="Next-session close-to-close return (W1 learning sprint label).",
            definition="log(adj_close[t+1] / adj_close[t])",
            horizon=1,
            compute=_close_to_close,
        ),
    )
}
DEFAULT_LABEL = "executed_return_v1"


def get_label(label_id: str) -> LabelSpec:
    try:
        return LABELS[label_id]
    except KeyError:
        raise LabelError(f"unknown label {label_id!r}") from None
