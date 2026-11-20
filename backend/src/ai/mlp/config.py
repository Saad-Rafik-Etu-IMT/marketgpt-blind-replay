"""Validated, JSON-serializable MLP architecture and training configurations."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from src.targets import TARGETS


class Activation(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"


class Norm(str, Enum):
    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"


class OutputInit(str, Enum):
    HE = "he"  # N(0, 2 / fan_in), the W1 output layer
    LECUN = "lecun"  # N(0, 1 / fan_in), the deep_mlp output layer


class Head(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    weight: float = Field(default=1.0, gt=0, le=100)

    @model_validator(mode="after")
    def _known(self) -> Self:
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}")
        return self


class ArchitectureConfig(BaseModel):
    """Shared trunk of hidden layers plus one linear head per target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden: tuple[int, ...] = Field(default=(32,), min_length=0, max_length=8)
    activation: Activation = Activation.RELU
    norm: Norm = Norm.NONE
    dropout: tuple[float, ...] | float = 0.0
    residual: bool = False
    input_skip: float | None = Field(default=None, gt=0, le=10)
    output_init: OutputInit = OutputInit.HE
    heads: tuple[Head, ...] = (Head(target="direction_binary_v1"),)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if any(w < 1 or w > 4096 for w in self.hidden):
            raise ValueError("hidden widths must be in 1..4096")
        rates = self.dropout_rates
        if len(rates) != len(self.hidden) or any(not 0 <= r < 1 for r in rates):
            raise ValueError(
                "dropout: one rate in [0, 1) per hidden layer, or one scalar"
            )
        if not self.heads:
            raise ValueError("at least one head is required")
        if len({h.target for h in self.heads}) != len(self.heads):
            raise ValueError("each target may have only one head")
        if self.input_skip is not None and not self.hidden:
            raise ValueError("input_skip needs at least one hidden layer")
        return self

    @property
    def dropout_rates(self) -> tuple[float, ...]:
        if isinstance(self.dropout, float | int):
            return tuple(float(self.dropout) for _ in self.hidden)
        return tuple(float(r) for r in self.dropout)


class Optimizer(str, Enum):
    SGD = "sgd"
    ADAMW = "adamw"


class Schedule(str, Enum):
    CONSTANT = "constant"
    COSINE = "cosine"


class Clip(str, Enum):
    NONE = "none"
    GLOBAL_NORM = "global_norm"
    VALUE = "value"  # element-wise, as in deep_mlp


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    epochs: int = Field(default=40, ge=1, le=500)
    batch_size: int = Field(default=64, ge=1, le=65536)
    optimizer: Optimizer = Optimizer.ADAMW
    learning_rate: float = Field(default=1e-3, gt=0, le=10)
    momentum: float = Field(default=0.0, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0, le=1)
    schedule: Schedule = Schedule.CONSTANT
    schedule_floor: float = Field(default=0.0, ge=0, le=1)
    warmup_epochs: int = Field(default=0, ge=0, le=100)
    clip: Clip = Clip.NONE
    clip_value: float = Field(default=1.0, gt=0)
    patience: int | None = Field(default=None, ge=1, le=500)
    min_delta: float = Field(default=0.0, ge=0)
    conflict_probe_rows: int = Field(default=512, ge=2, le=100000)

    def learning_rate_at(self, epoch: int) -> float:
        """Epoch-level learning rate (epoch counts from 1)."""
        if self.warmup_epochs and epoch <= self.warmup_epochs:
            return self.learning_rate * epoch / (self.warmup_epochs + 1)
        if self.schedule is Schedule.CONSTANT:
            return self.learning_rate
        span = max(1, self.epochs - self.warmup_epochs)
        progress = (epoch - 1 - self.warmup_epochs) / span
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.learning_rate * (
            self.schedule_floor + (1 - self.schedule_floor) * cosine
        )
