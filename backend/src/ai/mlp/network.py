"""Configurable multi-head MLP: shared trunk, one linear head per target.

Block i: ``z = h @ W + b`` -> optional norm -> activation -> optional residual
(added when the block keeps the width) -> inverted dropout. An optional linear
input shortcut ``scale * x @ W_skip`` is added to the last hidden output
(``deep_mlp`` style). Each head is linear; its target (T4) supplies the loss
and the gradient with respect to the head's raw output.

Parameters live in one flat dict with stable names (``hidden.0.W``,
``head.direction_binary_v1.b``, ...), which is also the checkpoint layout.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.ai.mlp.config import ArchitectureConfig, Norm, OutputInit
from src.ai.mlp.layers import (
    activate,
    activate_backward,
    batch_norm_backward,
    batch_norm_forward,
    dropout_mask,
    layer_norm_backward,
    layer_norm_forward,
)
from src.targets import get_target

Params = dict[str, np.ndarray]
NON_TRAINABLE = ("running_mean", "running_var")


def is_trainable(name: str) -> bool:
    return not name.endswith(NON_TRAINABLE)


def is_decayed(name: str) -> bool:
    """Weight decay applies to weight matrices only, never to biases or norms."""
    return name.endswith(".W")


class MLP:
    def __init__(self, n_in: int, config: ArchitectureConfig, seed: int = 0) -> None:
        if n_in < 1:
            raise ValueError("n_in must be >= 1")
        self.n_in, self.config = n_in, config
        self.params: Params = {}
        rng = np.random.default_rng(seed)
        width = n_in
        for i, out in enumerate(config.hidden):
            self.params[f"hidden.{i}.W"] = rng.normal(
                0.0, math.sqrt(2.0 / width), size=(width, out)
            )
            self.params[f"hidden.{i}.b"] = np.zeros(out)
            if config.norm is not Norm.NONE:
                self.params[f"hidden.{i}.gamma"] = np.ones(out)
                self.params[f"hidden.{i}.beta"] = np.zeros(out)
            if config.norm is Norm.BATCH:
                self.params[f"hidden.{i}.running_mean"] = np.zeros(out)
                self.params[f"hidden.{i}.running_var"] = np.ones(out)
            width = out
        if config.input_skip is not None:
            self.params["skip.W"] = rng.normal(
                0.0, math.sqrt(0.5 / n_in), size=(n_in, width)
            )
        scale = 2.0 if config.output_init is OutputInit.HE else 1.0
        for head in config.heads:
            dim = get_target(head.target).output_dim
            self.params[f"head.{head.target}.W"] = rng.normal(
                0.0, math.sqrt(scale / width), size=(width, dim)
            )
            self.params[f"head.{head.target}.b"] = np.zeros(dim)

    @property
    def parameter_count(self) -> int:
        return int(sum(v.size for k, v in self.params.items() if is_trainable(k)))

    def _flat(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x.reshape(len(x), -1) if x.ndim == 3 else x
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(
                f"expected inputs with {self.n_in} features, got shape {x.shape}"
            )
        return x

    def forward(
        self,
        x: np.ndarray,
        *,
        training: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Raw head outputs ``{target: z}`` and the cache for ``backward``."""
        x = self._flat(x)
        cfg, p = self.config, self.params
        h, blocks = x, []
        for i, rate in enumerate(cfg.dropout_rates):
            z = h @ p[f"hidden.{i}.W"] + p[f"hidden.{i}.b"]
            norm_cache = None
            if cfg.norm is Norm.BATCH:
                zn, norm_cache = batch_norm_forward(
                    z,
                    p[f"hidden.{i}.gamma"],
                    p[f"hidden.{i}.beta"],
                    p[f"hidden.{i}.running_mean"],
                    p[f"hidden.{i}.running_var"],
                    training,
                )
            elif cfg.norm is Norm.LAYER:
                zn, norm_cache = layer_norm_forward(
                    z, p[f"hidden.{i}.gamma"], p[f"hidden.{i}.beta"]
                )
            else:
                zn = z
            a = activate(zn, cfg.activation)
            residual = cfg.residual and h.shape[1] == a.shape[1]
            if residual:
                a = a + h
            mask = (
                dropout_mask(a.shape, rate, rng)
                if training and rate > 0 and rng is not None
                else None
            )
            blocks.append(
                {
                    "h_in": h,
                    "zn": zn,
                    "norm": norm_cache,
                    "mask": mask,
                    "residual": residual,
                }
            )
            h = a * mask if mask is not None else a
        if cfg.input_skip is not None:
            h = h + cfg.input_skip * (x @ p["skip.W"])
        outputs = {
            hd.target: h @ p[f"head.{hd.target}.W"] + p[f"head.{hd.target}.b"]
            for hd in cfg.heads
        }
        return outputs, {"x": x, "h": h, "blocks": blocks}

    def backward(
        self,
        cache: dict[str, Any],
        d_outputs: dict[str, np.ndarray],
        *,
        trunk_only: bool = False,
    ) -> Params:
        """Gradients of the loss given d loss / d head outputs (already weighted)."""
        cfg, p = self.config, self.params
        grads: Params = {}
        h = cache["h"]
        dh = np.zeros_like(h)
        for target, dz in d_outputs.items():
            if not trunk_only:
                grads[f"head.{target}.W"] = h.T @ dz
                grads[f"head.{target}.b"] = dz.sum(axis=0)
            dh = dh + dz @ p[f"head.{target}.W"].T
        if cfg.input_skip is not None:
            grads["skip.W"] = cfg.input_skip * (cache["x"].T @ dh)
        for i in range(len(cache["blocks"]) - 1, -1, -1):
            block = cache["blocks"][i]
            da = dh * block["mask"] if block["mask"] is not None else dh
            dh_in = da if block["residual"] else None
            dzn = activate_backward(da, block["zn"], cfg.activation)
            if cfg.norm is Norm.BATCH:
                dz, grads[f"hidden.{i}.gamma"], grads[f"hidden.{i}.beta"] = (
                    batch_norm_backward(dzn, block["norm"])
                )
            elif cfg.norm is Norm.LAYER:
                dz, grads[f"hidden.{i}.gamma"], grads[f"hidden.{i}.beta"] = (
                    layer_norm_backward(dzn, block["norm"])
                )
            else:
                dz = dzn
            grads[f"hidden.{i}.W"] = block["h_in"].T @ dz
            grads[f"hidden.{i}.b"] = dz.sum(axis=0)
            if i > 0 or trunk_only:
                back = dz @ p[f"hidden.{i}.W"].T
                dh = back + dh_in if dh_in is not None else back
        return grads

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Raw head outputs in evaluation mode (no dropout, running BatchNorm statistics)."""
        return self.forward(x, training=False)[0]

    def decode(
        self, x: np.ndarray, states: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Predictions in economic units for every head (``states`` = fitted target states)."""
        outputs = self.predict(x)
        return {
            t: get_target(t).decode(z, states.get(t, {})) for t, z in outputs.items()
        }
