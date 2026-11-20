"""Evaluation protocol shared by the study pipeline (training audit, 2026-11-20).

- ``executed_returns``: the label is the return a decision actually earns in
  the replay (decided at a close, filled at the next open, marked at the
  following open), not the close-to-close return.
- ``inner_split``: a chronological inner-validation set carved from TRAIN by a
  calendar cutoff shared by every asset, with whole-window and embargo rules
  mirroring the train/validation split. Every tuning choice uses it; the
  validation set is evaluated once.
- ``HoldPolicy`` / ``fit_policy``: a default-long decision rule with a minimum
  holding period, its flat threshold chosen on inner-validation.
- Small NumPy statistics used for the pooled report.

Indexing: ``values[t] = log(adj_close[t+1] / adj_close[t])``, so example ``t``
is decided at the close of bar ``t + 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

LABEL_HORIZON = 2  # value index t+2 = bar t+3, the open the executed return ends on
INNER_FRACTION = 0.2
MIN_HOLD = 5


# --------------------------------------------------------------------------- labels
def adjusted_opens(frame) -> np.ndarray:
    """Opens on the adjusted-close scale (the research replay's total-return units)."""
    return (
        frame.open.to_numpy(dtype=float)
        * frame.adjusted_close.to_numpy(dtype=float)
        / frame.close.to_numpy(dtype=float)
    )


def executed_returns(opens: np.ndarray, indices) -> np.ndarray:
    """log(O[t+3] / O[t+2]): decided at close t+1, filled at open t+2, held one bar."""
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return np.zeros(0)
    return np.log(opens[idx + 3] / opens[idx + 2])


# --------------------------------------------------------------------------- split
def inner_cutoff(train_dates: list[date], fraction: float = INNER_FRACTION) -> date:
    """Calendar date where inner-validation starts (last ``fraction`` of train)."""
    if not train_dates:
        raise ValueError("No training examples to split")
    return train_dates[int(len(train_dates) * (1 - fraction))]


def inner_split(
    train, bar_dates: list[date], cutoff: date, *, lookback: int = 64, embargo: int = 5
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split train examples at ``cutoff`` with the same rules as the main split.

    Inner-train keeps examples whose last label bar (``t + 3``) is at least
    ``embargo`` bars before the first bar on/after ``cutoff``. Inner-validation
    keeps examples whose whole input window (first close at bar ``t - 63``) is
    on/after ``cutoff``. Examples straddling the cutoff are dropped.
    """
    first = next((i for i, d in enumerate(bar_dates) if d >= cutoff), len(bar_dates))
    inner_train = tuple(t for t in train if t + 3 < first - embargo)
    inner_val = tuple(t for t in train if t - (lookback - 1) >= first)
    return inner_train, inner_val


# --------------------------------------------------------------------------- policy
@dataclass
class HoldPolicy:
    """Long unless P(up) < ``flat_below``; every switch is held ``min_hold`` decisions."""

    flat_below: float = 0.0
    min_hold: int = MIN_HOLD
    position: int = 0
    held: int = field(default=10**9)

    def step(self, p_up: float) -> int:
        wanted = 0 if p_up < self.flat_below else 1
        if wanted != self.position and self.held >= self.min_hold:
            self.position, self.held = wanted, 0
        self.held += 1
        return self.position


def policy_returns(
    p: np.ndarray, returns: np.ndarray, flat_below: float, min_hold: int, cost: float
) -> np.ndarray:
    """Per-decision net log returns of HoldPolicy on one chronological sequence."""
    policy = HoldPolicy(flat_below, min_hold)
    out, previous = np.zeros(len(p)), 0
    for i, (prob, ret) in enumerate(zip(p, returns, strict=True)):
        pos = policy.step(float(prob))
        out[i] = pos * ret - cost * abs(pos - previous)
        previous = pos
    return out


def simulate_policy(
    p: np.ndarray, returns: np.ndarray, flat_below: float, min_hold: int, cost: float
) -> float:
    """Net log return of HoldPolicy on one chronological sequence."""
    return float(policy_returns(p, returns, flat_below, min_hold, cost).sum())


def fit_policy(
    sequences: list[tuple[np.ndarray, np.ndarray]],
    *,
    min_hold: int = MIN_HOLD,
    cost: float = 0.5e-4,
    min_t: float = 2.0,
) -> dict[str, float]:
    """Choose ``flat_below`` on inner-validation sequences (one per asset).

    Default long: candidates are the 5th..50th percentiles of the pooled
    probabilities, and the best one is adopted only if its per-decision gain
    over always-long has a t-statistic of at least ``min_t``. Otherwise a
    threshold search over noise would always find an in-sample "edge".
    """
    pooled = np.concatenate([p for p, _ in sequences]) if sequences else np.zeros(0)
    base = (
        np.concatenate([policy_returns(p, r, 0.0, min_hold, cost) for p, r in sequences])
        if sequences
        else np.zeros(0)
    )
    best_tau, best_t, best_total = 0.0, -math.inf, float(base.sum())
    if len(pooled) > 1:
        for tau in sorted(set(np.quantile(pooled, np.linspace(0.05, 0.5, 10)).tolist())):
            series = np.concatenate(
                [policy_returns(p, r, tau, min_hold, cost) for p, r in sequences]
            )
            gain = series - base
            sd = float(gain.std(ddof=1))
            t = float(gain.mean() / (sd / math.sqrt(len(gain)))) if sd > 0 else -math.inf
            if t > best_t:
                best_t, best_candidate, candidate_total = t, float(tau), float(series.sum())
        if best_t >= min_t:
            best_tau, best_total = best_candidate, candidate_total
    return dict(
        flat_below=best_tau,
        min_hold=min_hold,
        min_t=min_t,
        best_candidate_t=best_t if math.isfinite(best_t) else None,
        inner_log_return=best_total,
        inner_always_long_log_return=float(base.sum()),
    )


# --------------------------------------------------------------------------- statistics
def macro_f1(pred, y) -> float:
    pred, y = np.asarray(pred, int), np.asarray(y, int)
    scores = []
    for cls in (0, 1):
        denom = int((pred == cls).sum() + (y == cls).sum())
        scores.append(2 * int(((pred == cls) & (y == cls)).sum()) / denom if denom else 0.0)
    return float(np.mean(scores))


def auc(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def date_block_bootstrap(
    dates: list, stat, *arrays, n_boot: int = 1000, block: int = 10, seed: int = 0
) -> list[float]:
    """95% CI resampling blocks of consecutive CALENDAR dates.

    All assets' decisions on a date stay together, so cross-asset correlation
    on the same day is not mistaken for independent evidence.
    """
    uniq = sorted(set(dates))
    pos = {d: i for i, d in enumerate(uniq)}
    by_date: list[list[int]] = [[] for _ in uniq]
    for i, d in enumerate(dates):
        by_date[pos[d]].append(i)
    rng = np.random.default_rng(seed)
    n = len(uniq)
    n_blocks = (n + block - 1) // block
    values = []
    for _ in range(n_boot):
        starts = rng.integers(n, size=n_blocks)
        days = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        idx = np.fromiter((i for d in days for i in by_date[d]), dtype=int)
        values.append(stat(*(np.asarray(a)[idx] for a in arrays)))
    lo, hi = np.nanquantile(values, [0.025, 0.975])
    return [float(lo), float(hi)]


def _finite(value):
    """JSON-safe: NaN (e.g. AUC with one class) becomes None, recursively."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pooled_report(p, y, dates, assets) -> dict:
    """Classification evidence on the pooled validation set (one evaluation)."""
    return _finite(_pooled_report(p, y, dates, assets))


def _pooled_report(p, y, dates, assets) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, int)
    pred = (p >= 0.5).astype(int)
    up = float(y.mean())
    diff = (pred == y).astype(float) - y.astype(float)
    return dict(
        assets=sorted(set(assets)),
        count=int(len(y)),
        dates=len(set(dates)),
        validationUp=up,
        accuracy=float(np.mean(pred == y)),
        alwaysUpAccuracy=up,
        accuracyMinusAlwaysUpInterval=date_block_bootstrap(dates, np.mean, diff),
        macroF1=macro_f1(pred, y),
        auc=auc(p, y),
        aucInterval=date_block_bootstrap(dates, auc, p, y),
        logLoss=log_loss(p, y),
        constantLogLoss=log_loss(np.full(len(y), up), y),
        perAsset={
            a: dict(
                count=int(m.sum()),
                accuracy=float(np.mean(pred[m] == y[m])),
                alwaysUpAccuracy=float(y[m].mean()),
                auc=auc(p[m], y[m]),
            )
            for a in sorted(set(assets))
            for m in [np.asarray(assets) == a]
        },
    )
