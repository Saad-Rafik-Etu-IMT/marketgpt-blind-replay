"""Isolated three-bar replay for an MLP that scores entry choices by net P&L."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from src.data.split import SplitConfig, build_examples, load_split_config

HORIZON = 3
OFFSETS = (1, 3, 5)
BASELINE_OFFSET = 3
OHLC = ["open", "high", "low", "close"]
# Hold out the old development/validation dates for this study, not the
# benchmark's sealed test period.
RESEARCH_SPLIT = SplitConfig(
    train_end=date(2014, 12, 31),
    validation_start=date(2015, 4, 1),
    validation_end=date(2016, 12, 31),
    test_start=date(2017, 4, 1),
    test_end=date(2019, 12, 31),
    label_horizon=HORIZON,
)


@dataclass(frozen=True)
class Choice:
    target: Literal["flat", "long", "short"]
    order_type: Literal["market", "limit"] | None = None
    offset_ticks: int | None = None

    @property
    def name(self) -> str:
        if self.offset_ticks is not None:
            return f"{self.target}_limit_{self.offset_ticks}"
        return f"{self.target}_market" if self.order_type == "market" else self.target


CHOICES = (
    Choice("flat"),
    Choice("long", "market"),
    Choice("long", "limit", 1),
    Choice("long", "limit", 3),
    Choice("long", "limit", 5),
    Choice("short", "market"),
    Choice("short", "limit", 1),
    Choice("short", "limit", 3),
    Choice("short", "limit", 5),
)


@dataclass(frozen=True)
class Costs:
    tick_size: float = 0.01
    spread_bps: float = 1.0
    fee_per_unit: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.tick_size) or self.tick_size <= 0:
            raise ValueError("tick_size must be positive and finite")
        if not 0 <= self.spread_bps < 20_000:
            raise ValueError("spread_bps must be in [0, 20000)")
        if not math.isfinite(self.fee_per_unit) or self.fee_per_unit < 0:
            raise ValueError("fee_per_unit must be nonnegative and finite")


@dataclass(frozen=True)
class Outcome:
    net_pnl: float
    fill_bar: int | None = None
    limit_price: float | None = None


class Metrics(TypedDict):
    total_net_pnl: float
    mean_net_pnl_per_window: float
    orders: int
    fills: int
    expired_limits: int
    choices: dict[str, int]


class Report(TypedDict):
    split: dict[str, object]
    costs: dict[str, float]
    seed: int
    epochs: int
    windows: dict[str, int]
    fixed_limit_ticks: int
    mlp: Metrics
    fixed_offset_baseline: Metrics


def simulate(
    choice: Choice, visible_close: float, future: np.ndarray, costs: Costs
) -> Outcome:
    """Score one unit from flat; future rows are N+1..N+3 in OHLC order."""
    if future.shape != (HORIZON, len(OHLC)):
        raise ValueError("entry scoring requires exactly three OHLC bars")
    if choice.target == "flat":
        return Outcome(0.0)
    side = 1 if choice.target == "long" else -1
    half_spread = costs.spread_bps / 20_000
    entry_quote_factor = 1 + side * half_spread
    exit_price = float(future[-1, 3]) * (1 - side * half_spread)
    limit_price = None
    offset = choice.offset_ticks
    if choice.order_type == "market":
        entry_price = float(future[0, 0]) * entry_quote_factor
        fill_bar = 1
    elif choice.order_type == "limit" and offset is not None and offset in OFFSETS:
        tick = Decimal(str(costs.tick_size))
        rounding = ROUND_FLOOR if side == 1 else ROUND_CEILING
        ticks = (Decimal(str(visible_close)) / tick).to_integral_value(rounding)
        ticks -= side * offset
        limit_price = float(ticks * tick)
        if limit_price <= 0:
            raise ValueError("buy limit is nonpositive; check price and tick size")
        # Require a full extra tick through the limit on the executable quote.
        threshold = float((ticks - side) * tick)
        extreme = future[:, 2 if side == 1 else 1] * entry_quote_factor
        eligible = np.flatnonzero(
            extreme <= threshold if side == 1 else extreme >= threshold
        )
        if len(eligible) == 0:
            return Outcome(0.0, limit_price=limit_price)
        fill_bar = int(eligible[0]) + 1
        entry_price = limit_price
    else:
        raise ValueError(f"invalid entry choice: {choice}")
    pnl = side * (exit_price - entry_price) - 2 * costs.fee_per_unit
    return Outcome(pnl, fill_bar, limit_price)


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    net_pnl: np.ndarray
    filled: np.ndarray


def _dataset(
    returns: np.ndarray,
    bars: np.ndarray,
    indices: tuple[int, ...],
    lookback: int,
    costs: Costs,
    *,
    nonoverlapping: bool = False,
) -> Dataset:
    rows = []
    pnl = []
    filled = []
    next_entry = 0
    for t in indices:
        if t < next_entry:
            continue
        rows.append(returns[t - lookback + 1 : t + 1])
        outcomes = [
            simulate(choice, bars[t, 3], bars[t + 1 : t + HORIZON + 1], costs)
            for choice in CHOICES
        ]
        pnl.append([outcome.net_pnl for outcome in outcomes])
        filled.append([outcome.fill_bar is not None for outcome in outcomes])
        if nonoverlapping:
            next_entry = t + HORIZON
    if not rows:
        raise ValueError("each split needs eligible decision windows")
    return Dataset(np.stack(rows), np.asarray(pnl), np.asarray(filled))


@dataclass(frozen=True)
class OrderMLP:
    mean: float
    std: float
    scale: float
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return one expected net-P&L score per choice, in currency per unit."""
        w1, b1, w2, b2 = self.weights
        hidden = np.maximum(0, ((x - self.mean) / self.std) @ w1 + b1)
        return (hidden @ w2 + b2) * self.scale


def fit(train: Dataset, validation: Dataset, *, epochs: int, seed: int) -> OrderMLP:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    rng = np.random.default_rng(seed)
    width = train.x.shape[1]
    weights = (
        rng.normal(0, math.sqrt(2 / width), (width, 32)),
        np.zeros(32),
        np.zeros((32, len(CHOICES))),
        np.zeros(len(CHOICES)),
    )
    mean, std = float(train.x.mean()), float(train.x.std()) or 1.0
    scale = float(np.sqrt(np.mean(train.net_pnl**2))) or 1.0
    x, y = (train.x - mean) / std, train.net_pnl / scale
    best_loss, best_weights = math.inf, None
    w1, b1, w2, b2 = weights
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 64):
            batch = order[start : start + 64]
            inputs = x[batch]
            z = inputs @ w1 + b1
            hidden = np.maximum(0, z)
            error = hidden @ w2 + b2 - y[batch]
            grad = 2 * error / error.size
            grad_hidden = (grad @ w2.T) * (z > 0)
            w1 -= 0.01 * (inputs.T @ grad_hidden)
            b1 -= 0.01 * grad_hidden.sum(axis=0)
            w2 -= 0.01 * (hidden.T @ grad)
            b2 -= 0.01 * grad.sum(axis=0)
        predicted = OrderMLP(mean, std, scale, weights).predict(validation.x)
        loss = float(np.mean(((predicted - validation.net_pnl) / scale) ** 2))
        if not math.isfinite(loss):
            raise RuntimeError("order MLP training diverged")
        if loss < best_loss:
            best_loss = loss
            best_weights = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
    assert best_weights is not None
    return OrderMLP(mean, std, scale, best_weights)


def _choose(scores: np.ndarray, *, fixed_offset: bool) -> np.ndarray:
    allowed = np.asarray(
        [
            i
            for i, choice in enumerate(CHOICES)
            if not fixed_offset
            or choice.target == "flat"
            or choice.offset_ticks == BASELINE_OFFSET
        ]
    )
    # Flat has known zero reward; never open a trade on a tie with flat.
    ranked = scores[:, allowed].copy()
    ranked[:, 0] = 0.0
    return allowed[np.argmax(ranked, axis=1)]


def _metrics(test: Dataset, choices: np.ndarray) -> Metrics:
    rows = np.arange(len(choices))
    filled = test.filled[rows, choices]
    pnl = test.net_pnl[rows, choices]
    limits = np.asarray([choice.order_type == "limit" for choice in CHOICES])[choices]
    return {
        "total_net_pnl": float(pnl.sum()),
        "mean_net_pnl_per_window": float(pnl.mean()),
        "orders": int(np.count_nonzero(choices)),
        "fills": int(filled.sum()),
        "expired_limits": int(np.count_nonzero(limits & ~filled)),
        "choices": {
            choice.name: int(np.count_nonzero(choices == i))
            for i, choice in enumerate(CHOICES)
        },
    }


def run(
    frame: pd.DataFrame,
    split: SplitConfig = RESEARCH_SPLIT,
    costs: Costs = Costs(),
    *,
    epochs: int = 40,
    seed: int = 0,
) -> Report:
    if split.label_horizon != HORIZON or split.test_end is None:
        raise ValueError("use a three-bar split with an explicit test_end")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("date column must not contain missing dates")
    in_range = dates.dt.date <= split.test_end
    days = dates[in_range].dt.date.tolist()
    bars = frame.loc[in_range, OHLC].to_numpy(dtype=float)
    adjusted = frame.loc[in_range, "adjusted_close"].to_numpy(dtype=float)
    if (
        not np.isfinite(bars).all()
        or (bars <= 0).any()
        or (bars[:, 2] > bars[:, [0, 1, 3]].min(axis=1)).any()
        or (bars[:, 1] < bars[:, [0, 3]].max(axis=1)).any()
        or not np.isfinite(adjusted).all()
        or (adjusted <= 0).any()
    ):
        raise ValueError(
            "research replay requires valid positive OHLC and adjusted_close"
        )

    # L returns need L+1 closes; enforce the boundary on the extra input bar too.
    input_split = split.model_copy(update={"lookback": split.lookback + 1})
    indices = build_examples(days, input_split)
    returns = np.diff(np.log(adjusted), prepend=np.nan)
    train = _dataset(returns, bars, indices.train, split.lookback, costs)
    val = _dataset(
        returns, bars, indices.validation, split.lookback, costs, nonoverlapping=True
    )
    model = fit(train, val, epochs=epochs, seed=seed)
    test = _dataset(
        returns, bars, indices.test, split.lookback, costs, nonoverlapping=True
    )
    scores = model.predict(test.x)
    if not np.isfinite(scores).all():
        raise RuntimeError("order MLP produced nonfinite holdout scores")
    return {
        "split": split.model_dump(mode="json"),
        "costs": asdict(costs),
        "seed": seed,
        "epochs": epochs,
        "windows": {
            "train": len(train.x),
            "validation": len(val.x),
            "test": len(test.x),
        },
        "fixed_limit_ticks": BASELINE_OFFSET,
        "mlp": _metrics(test, _choose(scores, fixed_offset=False)),
        "fixed_offset_baseline": _metrics(test, _choose(scores, fixed_offset=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--split", type=Path, help="custom three-bar split with test_end"
    )
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--fee-per-unit", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run(
        pd.read_csv(args.csv),
        load_split_config(args.split) if args.split else RESEARCH_SPLIT,
        Costs(args.tick_size, args.spread_bps, args.fee_per_unit),
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
