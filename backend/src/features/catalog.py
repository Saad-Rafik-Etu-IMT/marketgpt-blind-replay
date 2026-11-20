"""The project's default feature catalogue.

New groups register here; the engine, dataset builder and API read the
catalogue, so adding a feature never requires changing them.
"""

from __future__ import annotations

from src.features.groups import (
    regime,
    returns,
    statistics,
    structure,
    trend,
    volatility,
    volume,
)
from src.features.registry import FeatureRegistry

# The permanent baseline input: 64 adjusted-close log returns (W1 MLP, F3).
BASELINE_FEATURES: tuple[str, ...] = ("returns_window_64",)


def build_default_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    for group in (returns, trend, volatility, structure, volume, statistics, regime):
        for spec in group.specs():
            registry.register(spec)
    registry.validate()
    return registry


DEFAULT_REGISTRY: FeatureRegistry = build_default_registry()
