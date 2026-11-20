"""Feature specification: the contract every registered feature declares.

A ``FeatureSpec`` is metadata plus one deterministic NumPy/pandas function.
The engine, not the feature, enforces the point-in-time rules: warm-up rows are
masked, non-finite values after warm-up are refused and hard bounds are checked.

Timing convention (daily bars): a feature value at row ``t`` is the value known
at the close of bar ``t``. It may use bars ``t - lookback .. t`` and nothing
after ``t``. ``lookback`` counts the *earlier* bars required, so a one-day
return has ``lookback = 1`` and the 64-return window has ``lookback = 64``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from src.features.canonical import CANONICAL_FIELDS

ComputeFn = Callable[[pd.DataFrame, Mapping[str, pd.DataFrame]], pd.DataFrame]

_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_COLUMN = re.compile(r"^[a-z][a-z0-9_.\[\]]{0,95}$")


class FeatureSpecError(ValueError):
    """A feature declaration is incomplete or inconsistent."""


class FeatureGroup(str, Enum):
    RETURNS = "returns"
    TREND = "trend"
    VOLATILITY = "volatility"
    PRICE_STRUCTURE = "price_structure"
    VOLUME = "volume"
    STATISTICS = "statistics"
    REGIME = "regime"
    EXTERNAL = "external"
    CALENDAR = "calendar"
    ADVANCED = "advanced"


class Visibility(str, Enum):
    """What a feature reveals, checked against the blind-replay contract.

    - ``returns_only``: computable from the adjusted-close log returns that the
      F3 ``BlindObservation`` already exposes.
    - ``price_relative``: needs other adjusted OHLC fields, but only as
      scale-free ratios (no price level).
    - ``volume_relative``: needs volume, only as ratios to its own history.
    - ``dated``: reveals calendar position; never allowed in blind mode.
    - ``external``: aligned from another series by date; allowed in blind mode
      only once a separate review approves the source.
    """

    RETURNS_ONLY = "returns_only"
    PRICE_RELATIVE = "price_relative"
    VOLUME_RELATIVE = "volume_relative"
    DATED = "dated"
    EXTERNAL = "external"


BLIND_SAFE: frozenset[Visibility] = frozenset(
    {Visibility.RETURNS_ONLY, Visibility.PRICE_RELATIVE, Visibility.VOLUME_RELATIVE}
)


class MissingPolicy(str, Enum):
    """How missing values are handled; values are never silently filled.

    - ``warmup_nan``: NaN before ``lookback`` bars exist, finite afterwards
      (a non-finite value after warm-up is an error).
    - ``allow_nan``: NaN may also appear after warm-up when the definition is
      undefined (e.g. zero variance); the dataset builder drops those rows.
    """

    WARMUP_NAN = "warmup_nan"
    ALLOW_NAN = "allow_nan"


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    version: str
    group: FeatureGroup
    description: str
    definition: str
    required_fields: tuple[str, ...]
    lookback: int
    compute: ComputeFn = field(repr=False, compare=False)
    columns: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    visibility: Visibility = Visibility.RETURNS_ONLY
    frequency: str = "1d"
    availability: str = "close of bar t"
    adjustment: str = "adjusted_close scale (split and dividend adjusted)"
    missing_policy: MissingPolicy = MissingPolicy.WARMUP_NAN
    units: str = "dimensionless"
    expected_range: tuple[float | None, float | None] = (None, None)
    hard_bounds: tuple[float | None, float | None] = (None, None)

    def __post_init__(self) -> None:
        if not _ID.match(self.id):
            raise FeatureSpecError(f"invalid feature id {self.id!r}")
        if not _SEMVER.match(self.version):
            raise FeatureSpecError(f"{self.id}: version must be MAJOR.MINOR.PATCH")
        if not self.description.strip() or not self.definition.strip():
            raise FeatureSpecError(
                f"{self.id}: description and definition are required"
            )
        if not isinstance(self.lookback, int) or isinstance(self.lookback, bool):
            raise FeatureSpecError(f"{self.id}: lookback must be an int")
        if self.lookback < 0:
            raise FeatureSpecError(f"{self.id}: lookback must be >= 0")
        unknown = set(self.required_fields) - CANONICAL_FIELDS
        if unknown:
            raise FeatureSpecError(
                f"{self.id}: unknown source field(s) {sorted(unknown)}"
            )
        if not self.required_fields and not self.dependencies:
            raise FeatureSpecError(f"{self.id}: needs source fields or dependencies")
        if self.id in self.dependencies:
            raise FeatureSpecError(f"{self.id}: a feature cannot depend on itself")
        if not self.columns:
            object.__setattr__(self, "columns", (self.id,))
        if len(set(self.columns)) != len(self.columns):
            raise FeatureSpecError(f"{self.id}: duplicate column names")
        bad = [c for c in self.columns if not _COLUMN.match(c)]
        if bad:
            raise FeatureSpecError(f"{self.id}: invalid column name(s) {bad}")
        for name, (low, high) in (
            ("expected_range", self.expected_range),
            ("hard_bounds", self.hard_bounds),
        ):
            if low is not None and high is not None and low >= high:
                raise FeatureSpecError(f"{self.id}: {name} lower bound must be < upper")
        if not callable(self.compute):
            raise FeatureSpecError(f"{self.id}: compute must be callable")

    @property
    def width(self) -> int:
        return len(self.columns)

    @property
    def blind_safe(self) -> bool:
        return self.visibility in BLIND_SAFE

    def describe(self) -> dict[str, object]:
        """JSON-ready metadata (no callable), for manifests and the API."""
        return {
            "id": self.id,
            "version": self.version,
            "group": self.group.value,
            "description": self.description,
            "definition": self.definition,
            "required_fields": list(self.required_fields),
            "lookback": self.lookback,
            "columns": list(self.columns),
            "dependencies": list(self.dependencies),
            "visibility": self.visibility.value,
            "blind_safe": self.blind_safe,
            "frequency": self.frequency,
            "availability": self.availability,
            "adjustment": self.adjustment,
            "missing_policy": self.missing_policy.value,
            "units": self.units,
            "expected_range": list(self.expected_range),
            "hard_bounds": list(self.hard_bounds),
        }
