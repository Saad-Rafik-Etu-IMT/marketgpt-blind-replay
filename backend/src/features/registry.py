"""Versioned feature registry with dependency resolution and fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

from src.features.spec import FeatureGroup, FeatureSpec


class FeatureRegistryError(ValueError):
    """Unknown, duplicate or cyclic feature declarations."""


class FeatureRegistry:
    def __init__(self, specs: Iterable[FeatureSpec] = ()) -> None:
        self._specs: dict[str, FeatureSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.id in self._specs:
            raise FeatureRegistryError(f"feature {spec.id!r} is already registered")
        taken = {c: s.id for s in self._specs.values() for c in s.columns}
        clash = [c for c in spec.columns if c in taken]
        if clash:
            raise FeatureRegistryError(f"{spec.id}: column(s) {clash} already used")
        self._specs[spec.id] = spec
        return spec

    def __contains__(self, feature_id: object) -> bool:
        return feature_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, feature_id: str) -> FeatureSpec:
        try:
            return self._specs[feature_id]
        except KeyError:
            raise FeatureRegistryError(f"unknown feature {feature_id!r}") from None

    def specs(self, group: FeatureGroup | None = None) -> list[FeatureSpec]:
        """Registered specs, sorted by group then id (stable for listings)."""
        chosen = [s for s in self._specs.values() if group is None or s.group is group]
        return sorted(chosen, key=lambda s: (s.group.value, s.id))

    def ids(self, group: FeatureGroup | None = None) -> list[str]:
        return [s.id for s in self.specs(group)]

    def resolve(self, feature_ids: Sequence[str]) -> list[FeatureSpec]:
        """Requested features plus their dependencies, dependencies first.

        Raises on unknown ids, duplicates in the request, cycles and a feature
        whose declared lookback is shorter than one of its dependencies'.
        """
        if len(set(feature_ids)) != len(feature_ids):
            raise FeatureRegistryError("a feature is requested more than once")
        order: list[FeatureSpec] = []
        state: dict[str, str] = {}

        def visit(feature_id: str, path: tuple[str, ...]) -> None:
            if state.get(feature_id) == "done":
                return
            if state.get(feature_id) == "active":
                cycle = " -> ".join((*path, feature_id))
                raise FeatureRegistryError(f"dependency cycle: {cycle}")
            spec = self.get(feature_id)
            state[feature_id] = "active"
            for dependency in spec.dependencies:
                if self.get(dependency).lookback > spec.lookback:
                    raise FeatureRegistryError(
                        f"{spec.id}: lookback {spec.lookback} is shorter than "
                        f"dependency {dependency!r} ({self.get(dependency).lookback})"
                    )
                visit(dependency, (*path, feature_id))
            state[feature_id] = "done"
            order.append(spec)

        for feature_id in feature_ids:
            visit(feature_id, ())
        return order

    def validate(self) -> None:
        """Check every registered feature resolves (dependencies exist, no cycles)."""
        for spec in self._specs.values():
            self.resolve([spec.id])

    def columns(self, feature_ids: Sequence[str]) -> list[str]:
        """Output columns of the requested features, in request order."""
        self.resolve(feature_ids)
        return [c for feature_id in feature_ids for c in self.get(feature_id).columns]

    def lookback(self, feature_ids: Sequence[str]) -> int:
        """Warm-up bars needed before every requested feature is defined."""
        return max((s.lookback for s in self.resolve(feature_ids)), default=0)

    def fingerprint(self, feature_ids: Sequence[str]) -> str:
        """SHA-256 of the feature set: ids, versions, columns and request order.

        Any version bump, reordering or column change yields a new fingerprint,
        so a model artifact can refuse a feature schema it was not trained on.
        """
        payload = {
            "requested": list(feature_ids),
            # Dependencies count too: a new log-return version changes windows.
            "resolved": [
                {
                    "id": spec.id,
                    "version": spec.version,
                    "columns": list(spec.columns),
                    "dependencies": list(spec.dependencies),
                }
                for spec in self.resolve(feature_ids)
            ],
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
