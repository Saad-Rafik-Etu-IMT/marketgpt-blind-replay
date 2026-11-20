"""Versioned, point-in-time feature catalogue and engine (tranches T1-T2).

Pure and deterministic like ``src/core``: no database, network or file access.
See ``docs/features.md``.
"""

from src.features.catalog import BASELINE_FEATURES, DEFAULT_REGISTRY
from src.features.diagnostics import RedundancyReport, redundancy_report, training_rows
from src.features.engine import FeatureEngineError, FeatureMatrix, compute, vector_at
from src.features.profiles import (
    PROFILES,
    FeatureProfile,
    FeatureProfileError,
    get_profile,
    validate_profiles,
)
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
    "FeatureProfile",
    "FeatureProfileError",
    "FeatureMatrix",
    "FeatureRegistry",
    "FeatureRegistryError",
    "FeatureSpec",
    "FeatureSpecError",
    "MissingPolicy",
    "PROFILES",
    "RedundancyReport",
    "Visibility",
    "compute",
    "get_profile",
    "redundancy_report",
    "training_rows",
    "validate_profiles",
    "vector_at",
]
