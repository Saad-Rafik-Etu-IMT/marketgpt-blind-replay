"""Prediction-target registry, independent of the model architecture (T4).

A target turns raw future quantities into training labels and defines, for a
model's raw output ``z`` of shape ``[N, output_dim]``:

- ``loss(z, y, state) -> (value, grad)``: mean loss and its gradient d/dz,
  in plain NumPy so any model (the T5 MLP included) can back-propagate it;
- ``decode(z, state)``: predictions in economic units;
- ``evaluate(z, y, raw, sigma, state)``: target-specific metrics.

Fitted state (scales, class shares, regime cut-offs) comes from ``fit`` on
training rows only and is JSON-serializable for the dataset manifest.
Continuous targets are standardized with training statistics, so models work
in unit scale and every loss is comparable across targets (multi-task, T5).

Forecasting stays separate from trading: no target is a Buy/Sell/Hold action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.targets import metrics as m
from src.targets.quantities import (
    close_cumulative,
    executed_bars,
    executed_cumulative,
    future_volatility,
)

State = dict[str, Any]
NEUTRAL_BAND_K = 0.25  # neutral if |r| <= 0.25 * sigma20(t) * sqrt(h); predeclared
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
CURRENT_TEST_GAP_BARS, CURRENT_EMBARGO = 21, 5  # measured on SPY with config/split.json


class TargetError(ValueError):
    """Unknown target, invalid output shape or unusable training labels."""


class TargetKind(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    VOLATILITY = "volatility"
    QUANTILE = "quantile"


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _finite_dict(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            out[key] = _finite_dict(value)
        elif isinstance(value, float) and not np.isfinite(value):
            out[key] = None
        else:
            out[key] = value
    return out


class Target(ABC):
    id: str
    version: str = "1.0.0"
    kind: TargetKind
    interpretation: str
    definition: str
    horizons: tuple[int, ...]
    output_names: tuple[str, ...]
    parameterization: str
    loss_name: str
    metric_names: tuple[str, ...]
    calibration: str = "not applicable"
    needs_sigma: bool = False

    @property
    def output_dim(self) -> int:
        return len(self.output_names)

    @property
    def horizon_bars(self) -> int:
        """Bars after the decision bar that the label reads."""
        return max(executed_bars(h) for h in self.horizons)

    @property
    def development_only(self) -> bool:
        """True when the label cannot cross the current validation/test gap.

        With ``config/split.json`` the gap holds 21 bars (January 2020) and the
        embargo is 5, so labels reading more than 16 bars cannot be scored on
        the sealed test set. The builder enforces the exact rule on the data.
        """
        return self.horizon_bars + CURRENT_EMBARGO > CURRENT_TEST_GAP_BARS

    @abstractmethod
    def raw(self, bars: pd.DataFrame) -> np.ndarray:
        """Raw future quantities, shape [n_bars, k]; NaN where undefined."""

    def fit(self, raw: np.ndarray, sigma: np.ndarray) -> State:
        return {}

    @abstractmethod
    def encode(
        self, raw: np.ndarray, sigma: np.ndarray, state: State
    ) -> np.ndarray: ...

    @abstractmethod
    def loss(
        self, z: np.ndarray, y: np.ndarray, state: State
    ) -> tuple[float, np.ndarray]: ...

    @abstractmethod
    def decode(self, z: np.ndarray, state: State) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def evaluate(
        self,
        z: np.ndarray,
        y: np.ndarray,
        raw: np.ndarray,
        sigma: np.ndarray,
        state: State,
    ) -> dict[str, Any]: ...

    def check_output(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.output_dim:
            raise TargetError(f"{self.id}: model output must be [N, {self.output_dim}]")
        if not np.isfinite(z).all():
            raise TargetError(f"{self.id}: model output contains non-finite values")
        return z

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "kind": self.kind.value,
            "interpretation": self.interpretation,
            "definition": self.definition,
            "horizons": list(self.horizons),
            "horizon_bars": self.horizon_bars,
            "output_names": list(self.output_names),
            "output_dim": self.output_dim,
            "parameterization": self.parameterization,
            "loss": self.loss_name,
            "metrics": list(self.metric_names),
            "calibration": self.calibration,
            "needs_sigma": self.needs_sigma,
            "development_only": self.development_only,
            "overlapping_labels": self.horizon_bars > 2,
            "min_bootstrap_block": max(10, self.horizon_bars),
        }


# ---------------------------------------------------------------- classification
class _Classifier(Target):
    kind = TargetKind.CLASSIFICATION
    parameterization = "softmax over logits"
    loss_name = "cross-entropy"
    calibration = "required: reliability bins and ECE on validation"
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier",
        "ece",
        "baseline_class_frequency_log_loss",
    )

    def fit(self, raw: np.ndarray, sigma: np.ndarray) -> State:
        y = self.encode(raw, sigma, {})
        if len(y) == 0:
            raise TargetError(f"{self.id}: no training labels")
        shares = np.bincount(y, minlength=self.output_dim) / len(y)
        return {"train_class_shares": [float(s) for s in shares]}

    def loss(self, z, y, state):
        z = self.check_output(z)
        y = np.asarray(y, int)
        p = softmax(z)
        value = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-300, None))))
        grad = p.copy()
        grad[np.arange(len(y)), y] -= 1.0
        return value, grad / len(y)

    def decode(self, z, state):
        p = softmax(self.check_output(z))
        out = {f"p_{name}": p[:, i] for i, name in enumerate(self.output_names)}
        out["class"] = p.argmax(axis=1)
        return out

    def evaluate(self, z, y, raw, sigma, state):
        p = softmax(self.check_output(z))
        y = np.asarray(y, int)
        pred = p.argmax(axis=1)
        shares = np.asarray(
            state.get(
                "train_class_shares", np.full(self.output_dim, 1 / self.output_dim)
            )
        )
        result: dict[str, Any] = {
            "count": int(len(y)),
            "class_shares": [float(np.mean(y == c)) for c in range(self.output_dim)],
            "accuracy": m.accuracy(pred, y),
            "balanced_accuracy": m.balanced_accuracy(pred, y, self.output_dim),
            "macro_f1": m.macro_f1(pred, y, self.output_dim),
            "log_loss": m.log_loss(p, y),
            "brier": m.brier(p, y),
            "ece": m.expected_calibration_error(p.max(axis=1), pred == y),
            "baseline_class_frequency_log_loss": m.log_loss(
                np.tile(shares, (len(y), 1)), y
            ),
            "baseline_majority_accuracy": m.accuracy(
                np.full(len(y), int(np.argmax(shares))), y
            ),
            "confusion": m.confusion(pred, y, self.output_dim),
        }
        if self.output_dim == 2:
            result["auc"] = m.auc(p[:, 1], y)
            result["reliability"] = m.reliability_bins(p[:, 1], y == 1)
        return _finite_dict(result)


class DirectionBinary(_Classifier):
    output_names = ("down", "up")

    def __init__(self, target_id: str, source: str) -> None:
        self.id = target_id
        self.source = source
        self.horizons = (1,)
        if source == "executed":
            self.interpretation = "Will the next executed session return (next open to following open) be positive?"
            self.definition = (
                "up = log(adj_open[t+2] / adj_open[t+1]) > 0; zero counts as down"
            )
        else:
            self.interpretation = (
                "Will tomorrow's close be above today's close? (W1 label)"
            )
            self.definition = (
                "up = log(adj_close[t+1] / adj_close[t]) > 0; zero counts as down"
            )

    @property
    def horizon_bars(self) -> int:
        return 2 if self.source == "executed" else 1

    def raw(self, bars):
        values = (
            executed_cumulative(bars, 1)
            if self.source == "executed"
            else close_cumulative(bars, 1)
        )
        return values[:, None]

    def encode(self, raw, sigma, state):
        return (np.asarray(raw)[:, 0] > 0).astype(np.int64)


class DirectionThreeClass(_Classifier):
    output_names = ("down", "neutral", "up")
    needs_sigma = True
    id = "direction_3class_v1"
    horizons = (1,)
    interpretation = "Down, flat or up next executed session, relative to the asset's recent volatility."
    definition = (
        f"r = log(adj_open[t+2] / adj_open[t+1]); band = {NEUTRAL_BAND_K} * realized_vol_20[t] * sqrt(1); "
        "down if r < -band, up if r > band, else neutral (band known at t, not fitted)"
    )

    def raw(self, bars):
        return executed_cumulative(bars, 1)[:, None]

    def encode(self, raw, sigma, state):
        r = np.asarray(raw)[:, 0]
        band = NEUTRAL_BAND_K * np.asarray(sigma) * np.sqrt(self.horizons[0])
        return np.where(r < -band, 0, np.where(r > band, 2, 1)).astype(np.int64)


# ---------------------------------------------------------------- continuous
class _Standardized(Target):
    """Continuous labels standardized with training mean and std (per column)."""

    def raw_columns(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, float)

    def fit(self, raw, sigma):
        values = self.raw_columns(raw)
        if len(values) < 2:
            raise TargetError(f"{self.id}: at least two training labels are needed")
        std = values.std(axis=0)
        if (std <= 0).any():
            raise TargetError(f"{self.id}: constant training labels")
        state: State = {"mean": values.mean(axis=0).tolist(), "std": std.tolist()}
        if self.needs_sigma:
            q = np.quantile(np.asarray(sigma, float), [1 / 3, 2 / 3])
            state["sigma_terciles"] = [float(q[0]), float(q[1])]
        return state

    def encode(self, raw, sigma, state):
        return (self.raw_columns(raw) - np.asarray(state["mean"])) / np.asarray(
            state["std"]
        )

    def to_raw(self, standardized: np.ndarray, state: State) -> np.ndarray:
        return np.asarray(state["mean"]) + np.asarray(state["std"]) * standardized


class ExpectedReturn(_Standardized):
    kind = TargetKind.REGRESSION
    parameterization = "linear output in training-standardized units"
    metric_names = (
        "mae_bp",
        "rmse_bp",
        "loss",
        "r2_vs_train_mean",
        "rank_ic",
        "directional_accuracy",
    )

    def __init__(self, horizons: tuple[int, ...], loss: str = "huber") -> None:
        if loss not in ("huber", "mse"):
            raise TargetError("loss must be 'huber' or 'mse'")
        self.horizons = horizons
        self.loss_kind = loss
        self.loss_name = (
            "Huber (delta = 1 training std)" if loss == "huber" else "squared error / 2"
        )
        multi = len(horizons) > 1
        suffix = "" if loss == "huber" else "_mse"
        self.id = (
            "multi_horizon_v1" if multi else f"expected_return_h{horizons[0]}"
        ) + suffix
        self.output_names = tuple(f"return_h{h}" for h in horizons)
        self.interpretation = (
            "Conditional expected executed log return over "
            + ", ".join(f"{h}" for h in horizons)
            + " session(s), E[r(t+h) | information at t]."
        )
        self.definition = "; ".join(
            f"r_h{h} = log(adj_open[t+1+{h}] / adj_open[t+1])" for h in horizons
        )

    def raw(self, bars):
        return np.column_stack([executed_cumulative(bars, h) for h in self.horizons])

    def loss(self, z, y, state):
        z = self.check_output(z)
        e = z - np.asarray(y, float)
        n, k = e.shape
        if self.loss_kind == "huber":
            a = np.abs(e)
            value = np.where(a <= 1, 0.5 * e**2, a - 0.5)
            grad = np.clip(e, -1, 1)
        else:
            value, grad = 0.5 * e**2, e
        return float(value.mean()), grad / (n * k)

    def decode(self, z, state):
        predictions = self.to_raw(self.check_output(z), state)
        return {name: predictions[:, i] for i, name in enumerate(self.output_names)}

    def evaluate(self, z, y, raw, sigma, state):
        z = self.check_output(z)
        predictions = self.to_raw(z, state)
        truth = np.asarray(raw, float)
        loss_value, _ = self.loss(z, y, state)
        per = {}
        for i, h in enumerate(self.horizons):
            yt, yp = truth[:, i], predictions[:, i]
            per[f"h{h}"] = {
                "mae_bp": 1e4 * m.mae(yt, yp),
                "rmse_bp": 1e4 * m.rmse(yt, yp),
                "baseline_train_mean_mae_bp": 1e4
                * m.mae(yt, np.full(len(yt), state["mean"][i])),
                "baseline_zero_mae_bp": 1e4 * m.mae(yt, np.zeros(len(yt))),
                "r2_vs_train_mean": m.r2_out_of_sample(yt, yp, state["mean"][i]),
                "r2_vs_zero": m.r2_out_of_sample(yt, yp, 0.0),
                "rank_ic": m.rank_correlation(yt, yp),
                "directional_accuracy": m.directional_accuracy(yt, yp),
            }
        result: dict[str, Any] = {
            "count": int(len(truth)),
            "loss": loss_value,
            "units": "basis points of log return",
        }
        result.update(per if len(self.horizons) > 1 else per[f"h{self.horizons[0]}"])
        return _finite_dict(result)


class Volatility(_Standardized):
    kind = TargetKind.VOLATILITY
    needs_sigma = True
    parameterization = "standardized log-volatility; decoded sigma = exp(.) > 0"
    loss_name = "squared error on standardized log-volatility / 2"
    metric_names = (
        "mae",
        "rmse",
        "rmse_log",
        "qlike",
        "naive_mae",
        "mae_ratio_vs_naive",
    )

    def __init__(self, h: int) -> None:
        self.horizons = (h,)
        self.id = f"volatility_h{h}"
        self.output_names = (f"log_vol_h{h}",)
        self.interpretation = f"Realized volatility of the next {h} sessions (per-session std, not annualized)."
        self.definition = (
            f"std(ddof=1) of close-to-close log returns of bars t+1 .. t+{h}; "
            "model predicts the standardized log; naive baseline = realized_vol_20[t]"
        )

    @property
    def horizon_bars(self) -> int:
        return self.horizons[0]

    def raw(self, bars):
        return future_volatility(bars, self.horizons[0])[:, None]

    def raw_columns(self, raw):
        return np.log(np.asarray(raw, float))

    def loss(self, z, y, state):
        z = self.check_output(z)
        e = z - np.asarray(y, float)
        return float(np.mean(0.5 * e**2)), e / e.size

    def decode(self, z, state):
        return {"volatility": np.exp(self.to_raw(self.check_output(z), state)[:, 0])}

    def evaluate(self, z, y, raw, sigma, state):
        z = self.check_output(z)
        sigma_hat = self.decode(z, state)["volatility"]
        truth = np.asarray(raw, float)[:, 0]
        naive = np.asarray(sigma, float)
        result = {
            "count": int(len(truth)),
            "units": "per-session std of daily log returns",
            "mae": m.mae(truth, sigma_hat),
            "rmse": m.rmse(truth, sigma_hat),
            "rmse_log": m.rmse(np.log(truth), np.log(sigma_hat)),
            "qlike": m.qlike(truth, sigma_hat),
            "naive_mae": m.mae(truth, naive),
            "naive_qlike": m.qlike(truth, naive),
        }
        result["mae_ratio_vs_naive"] = (
            result["mae"] / result["naive_mae"]
            if result["naive_mae"] > 0
            else float("nan")
        )
        return _finite_dict(result)


class Quantiles(_Standardized):
    kind = TargetKind.QUANTILE
    needs_sigma = True
    parameterization = "q50 = z0; q10 = z0 - softplus(z1); q90 = z0 + softplus(z2) (no crossing by construction)"
    loss_name = "mean pinball loss over the three quantiles (standardized units)"
    calibration = (
        "required: empirical coverage of [Q10, Q90] overall and by volatility tercile"
    )
    metric_names = (
        "pinball",
        "coverage_80",
        "mean_width_bp",
        "crossing_rate",
        "coverage_by_volatility",
    )
    output_names = ("center", "lower_gap", "upper_gap")

    def __init__(self, h: int) -> None:
        self.horizons = (h,)
        self.id = f"quantiles_h{h}"
        self.interpretation = (
            f"10th, 50th and 90th percentiles of the next {h}-session executed return; "
            "three quantiles, not a full distribution."
        )
        self.definition = f"r = log(adj_open[t+1+{h}] / adj_open[t+1]); pinball loss at tau = 0.1, 0.5, 0.9"

    def raw(self, bars):
        return executed_cumulative(bars, self.horizons[0])[:, None]

    def _standardized_quantiles(self, z: np.ndarray) -> np.ndarray:
        center = z[:, 0]
        return np.column_stack(
            [center - softplus(z[:, 1]), center, center + softplus(z[:, 2])]
        )

    def loss(self, z, y, state):
        z = self.check_output(z)
        q = self._standardized_quantiles(z)
        target = np.asarray(y, float).reshape(-1, 1)
        taus = np.asarray(QUANTILES)
        diff = target - q
        value = np.maximum(taus * diff, (taus - 1) * diff)
        dq = ((diff < 0).astype(float) - taus) / q.size
        grad = np.column_stack(
            [dq.sum(axis=1), -dq[:, 0] * sigmoid(z[:, 1]), dq[:, 2] * sigmoid(z[:, 2])]
        )
        return float(value.mean()), grad

    def decode(self, z, state):
        q = self._standardized_quantiles(self.check_output(z))
        raw = state["mean"][0] + state["std"][0] * q
        return {"q10": raw[:, 0], "q50": raw[:, 1], "q90": raw[:, 2]}

    def evaluate(self, z, y, raw, sigma, state):
        z = self.check_output(z)
        d = self.decode(z, state)
        truth = np.asarray(raw, float)[:, 0]
        sigma = np.asarray(sigma, float)
        low, high = state["sigma_terciles"]
        by_regime = {}
        for name, mask in (
            ("low", sigma <= low),
            ("mid", (sigma > low) & (sigma <= high)),
            ("high", sigma > high),
        ):
            by_regime[name] = {
                "count": int(mask.sum()),
                "coverage_80": (
                    m.coverage(truth[mask], d["q10"][mask], d["q90"][mask])
                    if mask.any()
                    else float("nan")
                ),
            }
        result = {
            "count": int(len(truth)),
            "pinball": {
                f"q{int(t * 100)}": m.pinball(truth, d[f"q{int(t * 100)}"], t)
                for t in QUANTILES
            },
            "coverage_80": m.coverage(truth, d["q10"], d["q90"]),
            "below_q10": float(np.mean(truth < d["q10"])),
            "above_q90": float(np.mean(truth > d["q90"])),
            "mean_width_bp": 1e4 * float(np.mean(d["q90"] - d["q10"])),
            "crossing_rate": float(
                np.mean((d["q10"] > d["q50"]) | (d["q50"] > d["q90"]))
            ),
            "coverage_by_volatility": by_regime,
        }
        return _finite_dict(result)


TARGETS: dict[str, Target] = {
    t.id: t
    for t in (
        DirectionBinary("direction_binary_v1", "executed"),
        DirectionBinary("direction_binary_close_v1", "close"),
        DirectionThreeClass(),
        ExpectedReturn((1,)),
        ExpectedReturn((5,)),
        ExpectedReturn((20,)),
        ExpectedReturn((1,), loss="mse"),
        ExpectedReturn((5,), loss="mse"),
        Volatility(5),
        Volatility(20),
        Quantiles(1),
        ExpectedReturn((1, 5, 20)),
    )
}


def get_target(target_id: str) -> Target:
    try:
        return TARGETS[target_id]
    except KeyError:
        raise TargetError(f"unknown target {target_id!r}") from None
