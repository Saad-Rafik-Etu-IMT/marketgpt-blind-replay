"""NumPy building blocks with explicit forward and backward passes.

Conventions follow the W1 sprint and ``deep_mlp`` so presets reproduce them:
He-normal weights, BatchNorm momentum 0.1 and eps 1e-5 with population batch
variance, inverted dropout drawn with ``rng.random(shape) >= rate``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.ai.mlp.config import Activation

BN_MOMENTUM, NORM_EPS = 0.1, 1e-5
_GELU_C = math.sqrt(2.0 / math.pi)


def activate(z: np.ndarray, kind: Activation) -> np.ndarray:
    if kind is Activation.RELU:
        return np.maximum(z, 0.0)
    if kind is Activation.TANH:
        return np.tanh(z)
    return 0.5 * z * (1.0 + np.tanh(_GELU_C * (z + 0.044715 * z**3)))


def activate_backward(dout: np.ndarray, z: np.ndarray, kind: Activation) -> np.ndarray:
    if kind is Activation.RELU:
        return dout * (z > 0)
    if kind is Activation.TANH:
        return dout * (1.0 - np.tanh(z) ** 2)
    inner = _GELU_C * (z + 0.044715 * z**3)
    t = np.tanh(inner)
    derivative = 0.5 * (1 + t) + 0.5 * z * (1 - t**2) * _GELU_C * (
        1 + 3 * 0.044715 * z**2
    )
    return dout * derivative


def batch_norm_forward(
    z: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    training: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Batch statistics in training (running stats updated in place), running stats otherwise."""
    if training and len(z) > 1:
        mu, var = z.mean(axis=0), z.var(axis=0)
        running_mean[:] = (1.0 - BN_MOMENTUM) * running_mean + BN_MOMENTUM * mu
        running_var[:] = (1.0 - BN_MOMENTUM) * running_var + BN_MOMENTUM * var
    else:
        mu, var = running_mean, running_var
    inv_std = 1.0 / np.sqrt(var + NORM_EPS)
    z_hat = (z - mu) * inv_std
    batch = training and len(z) > 1
    return gamma * z_hat + beta, {
        "z_hat": z_hat,
        "inv_std": inv_std,
        "gamma": gamma,
        "batch": batch,
    }


def batch_norm_backward(
    dout: np.ndarray, cache: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    z_hat, inv_std, gamma = cache["z_hat"], cache["inv_std"], cache["gamma"]
    dgamma, dbeta = (dout * z_hat).sum(axis=0), dout.sum(axis=0)
    dz_hat = dout * gamma
    if not cache["batch"]:  # running statistics are constants
        return dz_hat * inv_std, dgamma, dbeta
    m = len(dout)
    dz = (inv_std / m) * (
        m * dz_hat - dz_hat.sum(axis=0) - z_hat * (dz_hat * z_hat).sum(axis=0)
    )
    return dz, dgamma, dbeta


def layer_norm_forward(
    z: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    mu = z.mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(z.var(axis=1, keepdims=True) + NORM_EPS)
    z_hat = (z - mu) * inv_std
    return gamma * z_hat + beta, {"z_hat": z_hat, "inv_std": inv_std, "gamma": gamma}


def layer_norm_backward(
    dout: np.ndarray, cache: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    z_hat, inv_std, gamma = cache["z_hat"], cache["inv_std"], cache["gamma"]
    dgamma, dbeta = (dout * z_hat).sum(axis=0), dout.sum(axis=0)
    dz_hat = dout * gamma
    d = z_hat.shape[1]
    dz = (inv_std / d) * (
        d * dz_hat
        - dz_hat.sum(axis=1, keepdims=True)
        - z_hat * (dz_hat * z_hat).sum(axis=1, keepdims=True)
    )
    return dz, dgamma, dbeta


def dropout_mask(
    shape: tuple[int, ...], rate: float, rng: np.random.Generator
) -> np.ndarray:
    return (rng.random(shape) >= rate).astype(np.float64) / (1.0 - rate)
