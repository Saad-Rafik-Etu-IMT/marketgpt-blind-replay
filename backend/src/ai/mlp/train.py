"""Deterministic training: mini-batches, SGD or AdamW, early stopping, best restore.

One ``np.random.default_rng(seed)`` drives the epoch shuffles and dropout masks
(as in W1 and ``deep_mlp``); the model's own seed drives initialization. The
selection set (inner-validation) picks the best epoch; the validation set is
never passed here.

Multi-task: the total loss is ``sum_k weight_k * loss_k``. Losses of
continuous targets are already standardized (T4), so they are comparable. Each
epoch records per-head losses and the cosine similarity between heads' trunk
gradients on a fixed probe batch: a negative value signals conflicting
gradients (possible negative transfer).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from src.ai.mlp.config import Clip, Norm, Optimizer, TrainConfig
from src.ai.mlp.network import MLP, Params, is_decayed, is_trainable
from src.targets import get_target


class TrainingDivergedError(RuntimeError):
    """A loss, gradient or parameter became NaN or infinite."""

    def __init__(self, message: str, history: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.history = history


@dataclass(frozen=True)
class TrainingData:
    """Inputs and per-target labels for training and selection rows."""

    x_train: np.ndarray
    y_train: dict[str, np.ndarray]
    x_select: np.ndarray
    y_select: dict[str, np.ndarray]
    target_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_datasets(
        cls,
        datasets: list[Any],
        train_part: str = "inner_train",
        select_part: str = "inner_validation",
    ) -> TrainingData:
        """One dataset per target, built with the same rows (see ``align_rows``)."""
        if not datasets:
            raise ValueError("at least one dataset is required")
        first = datasets[0]
        for other in datasets[1:]:
            for part in (train_part, select_part):
                if not first.parts[part].index.equals(
                    other.parts[part].index
                ) or not np.array_equal(first.parts[part].x, other.parts[part].x):
                    raise ValueError(
                        "datasets disagree on rows or inputs; build them with the same spec "
                        "and min_horizon set to the longest target horizon"
                    )
        names = [
            d.target.id if d.target is not None else "direction_binary_v1"
            for d in datasets
        ]
        if len(set(names)) != len(names):
            raise ValueError("one dataset per target")
        return cls(
            x_train=first.parts[train_part].x,
            y_train={
                n: d.parts[train_part].y for n, d in zip(names, datasets, strict=True)
            },
            x_select=first.parts[select_part].x,
            y_select={
                n: d.parts[select_part].y for n, d in zip(names, datasets, strict=True)
            },
            target_states={
                n: d.target_state or {} for n, d in zip(names, datasets, strict=True)
            },
        )


@dataclass
class TrainResult:
    params: Params
    history: list[dict[str, Any]]
    best_epoch: int
    stopped_early: bool
    steps: int


def _losses(
    model: MLP, outputs: dict[str, np.ndarray], y: dict[str, np.ndarray], states
) -> dict[str, tuple[float, np.ndarray]]:
    return {
        t: get_target(t).loss(z, y[t], states.get(t, {})) for t, z in outputs.items()
    }


def evaluate_loss(
    model: MLP, x: np.ndarray, y: dict[str, np.ndarray], states: dict[str, Any]
) -> dict[str, float]:
    outputs = model.predict(x)
    per = {t: v for t, (v, _) in _losses(model, outputs, y, states).items()}
    weights = {h.target: h.weight for h in model.config.heads}
    per["total"] = float(sum(weights[t] * v for t, v in per.items()))
    return per


def gradient_conflicts(
    model: MLP, x: np.ndarray, y: dict[str, np.ndarray], states
) -> dict[str, float]:
    """Cosine similarity of each pair of heads' gradients on the shared trunk."""
    if len(model.config.heads) < 2 or not model.config.hidden:
        return {}
    outputs, cache = model.forward(x, training=False)
    vectors = {}
    for head in model.config.heads:
        _, grad = get_target(head.target).loss(
            outputs[head.target], y[head.target], states.get(head.target, {})
        )
        g = model.backward(cache, {head.target: head.weight * grad}, trunk_only=True)
        vectors[head.target] = np.concatenate([g[k].ravel() for k in sorted(g)])
    out = {}
    for a, b in combinations(sorted(vectors), 2):
        na, nb = np.linalg.norm(vectors[a]), np.linalg.norm(vectors[b])
        out[f"{a}|{b}"] = (
            float(vectors[a] @ vectors[b] / (na * nb))
            if na > 0 and nb > 0
            else float("nan")
        )
    return out


def _all_finite(arrays) -> bool:
    return all(np.isfinite(a).all() for a in arrays)


def train(
    model: MLP,
    data: TrainingData,
    config: TrainConfig,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Train ``model`` in place; ``on_epoch(record)`` may raise to cancel (T6 orchestration)."""
    if len(data.x_train) == 0 or len(data.x_select) == 0:
        raise ValueError("training and selection sets must be non-empty")
    missing = {h.target for h in model.config.heads} - set(data.y_train)
    if missing:
        raise ValueError(f"no labels for head(s) {sorted(missing)}")
    rng = np.random.default_rng(config.seed)
    states = data.target_states
    weights = {h.target: h.weight for h in model.config.heads}
    trainable = [k for k in model.params if is_trainable(k)]
    moment1 = {k: np.zeros_like(model.params[k]) for k in trainable}
    moment2 = {k: np.zeros_like(model.params[k]) for k in trainable}
    skip_singletons = model.config.norm is Norm.BATCH
    probe = slice(0, min(config.conflict_probe_rows, len(data.x_select)))
    probe_y = {t: v[probe] for t, v in data.y_select.items()}

    history: list[dict[str, Any]] = []
    best_total, best_params, best_epoch, waited, step = np.inf, None, 0, 0, 0
    n = len(data.x_train)
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        lr = config.learning_rate_at(epoch)
        order = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            batch = order[start : start + config.batch_size]
            if skip_singletons and len(batch) < 2:
                continue
            step += 1
            # Divergence is detected explicitly below, so overflow warnings are silenced.
            with np.errstate(over="ignore", invalid="ignore"):
                outputs, cache = model.forward(
                    data.x_train[batch], training=True, rng=rng
                )
            if not _all_finite(outputs.values()):
                raise TrainingDivergedError(
                    f"non-finite model output at epoch {epoch}, step {step}", history
                )
            losses = _losses(
                model, outputs, {t: v[batch] for t, v in data.y_train.items()}, states
            )
            grads = model.backward(
                cache, {t: weights[t] * g for t, (_, g) in losses.items()}
            )
            if not _all_finite(
                [np.array([v for v, _ in losses.values()]), *grads.values()]
            ):
                raise TrainingDivergedError(
                    f"non-finite loss or gradient at epoch {epoch}, step {step}",
                    history,
                )
            _step(model.params, grads, moment1, moment2, config, lr, step)
            if not _all_finite(model.params[k] for k in trainable):
                raise TrainingDivergedError(
                    f"non-finite parameters at epoch {epoch}, step {step}", history
                )

        train_losses = evaluate_loss(model, data.x_train, data.y_train, states)
        select_losses = evaluate_loss(model, data.x_select, data.y_select, states)
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": train_losses["total"],
            "select_loss": select_losses["total"],
            "train_loss_per_head": {
                k: v for k, v in train_losses.items() if k != "total"
            },
            "select_loss_per_head": {
                k: v for k, v in select_losses.items() if k != "total"
            },
            "gradient_cosine": gradient_conflicts(
                model, data.x_select[probe], probe_y, states
            ),
        }
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)
        if not np.isfinite(record["select_loss"]):
            raise TrainingDivergedError(
                f"non-finite selection loss at epoch {epoch}", history
            )
        if record["select_loss"] < best_total - config.min_delta:
            best_total, best_epoch, waited = record["select_loss"], epoch, 0
            best_params = {k: v.copy() for k, v in model.params.items()}
        else:
            waited += 1
            if config.patience is not None and waited >= config.patience:
                stopped_early = True
                break

    assert best_params is not None
    model.params = best_params
    for record in history:
        record["selected"] = record["epoch"] == best_epoch
    return TrainResult(
        params=best_params,
        history=history,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        steps=step,
    )


def _step(
    params: Params,
    grads: Params,
    m: Params,
    v: Params,
    config: TrainConfig,
    lr: float,
    step: int,
) -> None:
    if config.clip is Clip.GLOBAL_NORM:
        norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
        if norm > config.clip_value:
            grads = {k: g * (config.clip_value / norm) for k, g in grads.items()}
    elif config.clip is Clip.VALUE:
        grads = {
            k: np.clip(g, -config.clip_value, config.clip_value)
            for k, g in grads.items()
        }
    if config.optimizer is Optimizer.SGD:
        for k, g in grads.items():
            if config.weight_decay and is_decayed(k):
                params[k] *= 1.0 - lr * config.weight_decay
            if config.momentum:
                m[k] = config.momentum * m[k] + g
                g = m[k]
            params[k] = params[k] - lr * g
        return
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    correction1, correction2 = 1.0 - beta1**step, 1.0 - beta2**step
    for k, g in grads.items():
        m[k] = beta1 * m[k] + (1.0 - beta1) * g
        v[k] = beta2 * v[k] + (1.0 - beta2) * (g * g)
        if config.weight_decay and is_decayed(k):
            params[k] *= 1.0 - lr * config.weight_decay
        params[k] -= lr * (m[k] / correction1) / (np.sqrt(v[k] / correction2) + eps)
