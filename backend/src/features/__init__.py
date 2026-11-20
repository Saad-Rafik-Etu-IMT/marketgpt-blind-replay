"""Versioned, point-in-time feature catalogue and engine (tranche T1).

Pure and deterministic like ``src/core``: no database, network or file access.
See ``docs/features.md``.
"""

from src.features.catalog import BASELINE_FEATURES, DEFAULT_REGISTRY
from src.features.engine import FeatureEngineError, FeatureMatrix, compute, vector_at
from src.features.registry import FeatureRegistry, FeatureRegistryError
from src.features.spec import (
    FeatureGroup,
    FeatureSpec,
    FeatureSpecError,
    MissingPolicy,
    Visibility,
)

__all__ = [
    "BASELINE_FEATURES",
    "DEFAULT_REGISTRY",
    "FeatureEngineError",
    "FeatureGroup",
    "FeatureMatrix",
    "FeatureRegistry",
    "FeatureRegistryError",
    "FeatureSpec",
    "FeatureSpecError",
    "MissingPolicy",
    "Visibility",
    "compute",
    "vector_at",
]
