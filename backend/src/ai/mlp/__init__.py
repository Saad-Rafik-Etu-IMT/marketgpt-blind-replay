"""Configurable, multi-head NumPy MLP (tranche T5). See ``docs/mlp.md``."""

from src.ai.mlp.checkpoint import (
    CheckpointError,
    LoadedModel,
    load,
    save,
    weights_digest,
)
from src.ai.mlp.config import (
    Activation,
    ArchitectureConfig,
    Clip,
    Head,
    Norm,
    Optimizer,
    OutputInit,
    Schedule,
    TrainConfig,
)
from src.ai.mlp.network import MLP
from src.ai.mlp.presets import PRESETS, preset
from src.ai.mlp.train import (
    TrainingData,
    TrainingDivergedError,
    TrainResult,
    evaluate_loss,
    gradient_conflicts,
    train,
)

__all__ = [
    "MLP",
    "PRESETS",
    "Activation",
    "ArchitectureConfig",
    "CheckpointError",
    "Clip",
    "Head",
    "LoadedModel",
    "Norm",
    "Optimizer",
    "OutputInit",
    "Schedule",
    "TrainConfig",
    "TrainResult",
    "TrainingData",
    "TrainingDivergedError",
    "evaluate_loss",
    "gradient_conflicts",
    "load",
    "preset",
    "save",
    "train",
    "weights_digest",
]
