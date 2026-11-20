"""Prediction targets, losses and metrics, independent of the model (tranche T4).

See ``docs/targets.md``.
"""

from src.targets.targets import (
    NEUTRAL_BAND_K,
    QUANTILES,
    TARGETS,
    Target,
    TargetError,
    TargetKind,
    get_target,
    softmax,
    softplus,
)

__all__ = [
    "NEUTRAL_BAND_K",
    "QUANTILES",
    "TARGETS",
    "Target",
    "TargetError",
    "TargetKind",
    "get_target",
    "softmax",
    "softplus",
]
