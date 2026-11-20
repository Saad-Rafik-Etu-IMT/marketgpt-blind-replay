"""Named starting configurations. Candidates, not predetermined winners.

- ``w1_baseline``: the permanent baseline, 64 -> 32 -> 2 ReLU, plain SGD
  (lr 0.05, batch 64, 40 epochs), He init: reproduces ``learning/mlp/step8_train``.
- ``small`` / ``medium`` / ``large``: 32, 64 -> 32 and 128 -> 64 -> 32 hidden
  units with AdamW, cosine schedule, dropout 0.1, weight decay 1e-2, global-norm
  clipping and early stopping (patience 7). Starting points for T7 comparisons.
- ``deep_v1``: the architecture of ``learning/mlp/deep_mlp.py`` (128 -> 64 ->
  32, BatchNorm, dropout 0.25 on the first two layers, 0.35 input shortcut,
  LeCun output init) with its optimizer (AdamW 1.5e-3, decay 2e-3, cosine to
  15%, element-wise clipping at 2). It is fed with the T2/T3 features rather
  than ``extract_blind_features``.
"""

from __future__ import annotations

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


def _heads(targets: tuple[str, ...]) -> tuple[Head, ...]:
    return tuple(Head(target=t) for t in targets)


def _modern(
    hidden: tuple[int, ...], targets: tuple[str, ...]
) -> tuple[ArchitectureConfig, TrainConfig]:
    return (
        ArchitectureConfig(
            hidden=hidden,
            activation=Activation.RELU,
            dropout=0.1,
            heads=_heads(targets),
        ),
        TrainConfig(
            optimizer=Optimizer.ADAMW,
            learning_rate=1e-3,
            weight_decay=1e-2,
            schedule=Schedule.COSINE,
            schedule_floor=0.1,
            clip=Clip.GLOBAL_NORM,
            clip_value=1.0,
            patience=7,
            epochs=40,
        ),
    )


def preset(
    name: str, targets: tuple[str, ...] = ("direction_binary_v1",)
) -> tuple[ArchitectureConfig, TrainConfig]:
    if name == "w1_baseline":
        return (
            ArchitectureConfig(
                hidden=(32,), heads=_heads(targets), output_init=OutputInit.HE
            ),
            TrainConfig(
                optimizer=Optimizer.SGD, learning_rate=0.05, batch_size=64, epochs=40
            ),
        )
    if name == "small":
        return _modern((32,), targets)
    if name == "medium":
        return _modern((64, 32), targets)
    if name == "large":
        return _modern((128, 64, 32), targets)
    if name == "deep_v1":
        return (
            ArchitectureConfig(
                hidden=(128, 64, 32),
                norm=Norm.BATCH,
                dropout=(0.25, 0.25, 0.0),
                input_skip=0.35,
                output_init=OutputInit.LECUN,
                heads=_heads(targets),
            ),
            TrainConfig(
                optimizer=Optimizer.ADAMW,
                learning_rate=1.5e-3,
                weight_decay=2e-3,
                schedule=Schedule.COSINE,
                schedule_floor=0.15,
                clip=Clip.VALUE,
                clip_value=2.0,
                epochs=40,
            ),
        )
    raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")


PRESETS: tuple[str, ...] = ("w1_baseline", "small", "medium", "large", "deep_v1")
