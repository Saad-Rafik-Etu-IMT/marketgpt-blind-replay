"""Experiment profiles and the validated experiment specification.

Profiles (configurable starting points, not validated optimums):

| profile      | seeds | max epochs | patience | walk-forward folds | validation read |
| development  | 1     | 10         | preset   | 0                  | optional        |
| research     | 5     | 40         | 7        | 0                  | optional        |
| confirmation | 10    | 60         | 7        | 3                  | always          |

An experiment trains one multi-head model (one head per target) for every
(feature profile, preset, seed). The grid is bounded: at most 60 runs and a
compute budget (``estimate.py``), so no uncontrolled Cartesian explosion.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from src.ai.mlp import PRESETS, TrainConfig, preset
from src.datasets import ContextRule, ScalerKind
from src.experiments.sources import SOURCE_NAME, SYNTHETIC
from src.features import PROFILES as FEATURE_PROFILES
from src.targets import TARGETS

MAX_RUNS = 60
RUN_KEY = re.compile(r"^[a-z0-9_]+__[a-z0-9_]+__s[0-9]+$")


@dataclass(frozen=True)
class ExperimentProfile:
    id: str
    version: str
    seeds: int
    max_epochs: int
    patience: int | None
    walk_forward_folds: int
    report_validation: bool
    description: str


EXPERIMENT_PROFILES: dict[str, ExperimentProfile] = {
    p.id: p
    for p in (
        ExperimentProfile(
            "development",
            "1.0.0",
            1,
            10,
            None,
            0,
            False,
            "Smoke tests and quick iteration.",
        ),
        ExperimentProfile(
            "research",
            "1.0.0",
            5,
            40,
            7,
            0,
            False,
            "Feature-group and architecture comparisons.",
        ),
        ExperimentProfile(
            "confirmation",
            "1.0.0",
            10,
            60,
            7,
            3,
            True,
            "Locked protocol: 10 seeds, walk-forward folds inside training, validation read once.",
        ),
    )
}


@dataclass(frozen=True)
class Run:
    feature_profile: str
    preset: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.feature_profile}__{self.preset}__s{self.seed}"


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9 _.,()-]+$")
    profile: Literal["development", "research", "confirmation"] = "development"
    sources: tuple[str, ...] = Field(default=(SYNTHETIC,), min_length=1, max_length=15)
    primary_asset: str | None = None
    feature_profiles: tuple[str, ...] = Field(
        default=("returns_only_v1",), min_length=1, max_length=8
    )
    presets: tuple[str, ...] = Field(
        default=("w1_baseline",), min_length=1, max_length=5
    )
    targets: tuple[str, ...] = Field(
        default=("direction_binary_v1",), min_length=1, max_length=4
    )
    context: ContextRule = ContextRule.OWN_PARTITION
    scaler: ScalerKind = ScalerKind.STANDARD
    epochs: int | None = Field(default=None, ge=1, le=60)
    seeds: tuple[int, ...] | None = None
    report_validation: bool = False
    max_minutes: int | None = Field(default=None, ge=1, le=24 * 60)

    @model_validator(mode="after")
    def _check(self) -> Self:
        profile = EXPERIMENT_PROFILES[self.profile]
        for name in self.sources:
            if not SOURCE_NAME.match(name):
                raise ValueError(f"invalid source name {name!r}")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("duplicate sources")
        if self.primary_asset is not None and self.primary_asset not in self.sources:
            raise ValueError("primary_asset must be one of the sources")
        for fp in self.feature_profiles:
            if fp not in FEATURE_PROFILES or not FEATURE_PROFILES[fp].available:
                raise ValueError(f"unknown or unavailable feature profile {fp!r}")
        unknown = [p for p in self.presets if p not in PRESETS]
        if unknown:
            raise ValueError(f"unknown preset(s) {unknown}")
        unknown = [t for t in self.targets if t not in TARGETS]
        if unknown:
            raise ValueError(f"unknown target(s) {unknown}")
        for values, what in (
            (self.feature_profiles, "feature profiles"),
            (self.presets, "presets"),
            (self.targets, "targets"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {what}")
        if self.epochs is not None and self.epochs > profile.max_epochs:
            raise ValueError(
                f"{self.profile} allows at most {profile.max_epochs} epochs"
            )
        if self.seeds is not None:
            if len(self.seeds) != profile.seeds or len(set(self.seeds)) != len(
                self.seeds
            ):
                raise ValueError(
                    f"{self.profile} needs exactly {profile.seeds} distinct seeds"
                )
            if any(s < 0 or s > 2**31 - 1 for s in self.seeds):
                raise ValueError("seed outside 0 .. 2^31 - 1")
        if len(self.runs()) > MAX_RUNS:
            raise ValueError(
                f"{len(self.runs())} runs exceed the limit of {MAX_RUNS}; split the grid"
            )
        return self

    @property
    def experiment_profile(self) -> ExperimentProfile:
        return EXPERIMENT_PROFILES[self.profile]

    @property
    def resolved_seeds(self) -> tuple[int, ...]:
        return (
            self.seeds
            if self.seeds is not None
            else tuple(range(self.experiment_profile.seeds))
        )

    @property
    def reads_validation(self) -> bool:
        return self.report_validation or self.experiment_profile.report_validation

    def runs(self) -> list[Run]:
        return [
            Run(fp, ps, seed)
            for fp in self.feature_profiles
            for ps in self.presets
            for seed in self.resolved_seeds
        ]

    def train_config(self, preset_name: str, seed: int) -> TrainConfig:
        """The preset's training config, capped by the profile, seeded per run."""
        _, config = preset(preset_name, self.targets)
        profile = self.experiment_profile
        epochs = (
            self.epochs
            if self.epochs is not None
            else min(config.epochs, profile.max_epochs)
        )
        patience = config.patience if config.patience is not None else profile.patience
        return config.model_copy(
            update={"epochs": epochs, "patience": patience, "seed": seed}
        )

    def fingerprint(self) -> str:
        payload = {
            **self.model_dump(mode="json"),
            "profile_version": self.experiment_profile.version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
