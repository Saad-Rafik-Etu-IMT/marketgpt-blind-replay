"""Target-specific evaluation metrics (pure NumPy, no fitted state).

``macro_f1``, ``auc`` and ``log_loss`` match ``learning/mlp/protocol.py`` on
binary inputs (tested). Undefined values (one class only, constant series)
are NaN, never 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# ------------------------------------------------------------ classification
def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(y))) if len(y) else float("nan")


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    pred, y = np.asarray(pred), np.asarray(y)
    recalls = [np.mean(pred[y == c] == c) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int = 2) -> float:
    pred, y = np.asarray(pred, int), np.asarray(y, int)
    scores = []
    for c in range(n_classes):
        denom = int((pred == c).sum() + (y == c).sum())
        scores.append(2 * int(((pred == c) & (y == c)).sum()) / denom if denom else 0.0)
    return float(np.mean(scores))


def log_loss(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood; ``probabilities`` is [N, C]."""
    p = np.clip(np.asarray(probabilities, float), EPS, 1 - EPS)
    return float(-np.mean(np.log(p[np.arange(len(y)), np.asarray(y, int)])))


def brier(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean over rows of sum_c (p_c - 1[y = c])^2."""
    p = np.asarray(probabilities, float)
    onehot = np.eye(p.shape[1])[np.asarray(y, int)]
    return float(np.mean(((p - onehot) ** 2).sum(axis=1)))


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width bins on [0, 1]; confidence 1.0 falls in the last bin."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            out.append(
                {
                    "low": float(edges[b]),
                    "high": float(edges[b + 1]),
                    "count": int(mask.sum()),
                    "confidence": float(confidence[mask].mean()),
                    "frequency": float(correct[mask].mean()),
                }
            )
    return out


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    n = len(confidence)
    if n == 0:
        return float("nan")
    return float(
        sum(
            b["count"] / n * abs(b["frequency"] - b["confidence"])
            for b in reliability_bins(confidence, correct, bins)
        )
    )


def auc(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def confusion(pred: np.ndarray, y: np.ndarray, n_classes: int) -> list[list[int]]:
    """Rows = true class, columns = predicted class."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return matrix.tolist()


# ---------------------------------------------------------------- regression
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def r2_out_of_sample(y: np.ndarray, yhat: np.ndarray, benchmark: float) -> float:
    """1 - SSE(model) / SSE(constant benchmark fitted on train). Negative = worse."""
    y = np.asarray(y, float)
    denom = float(np.sum((y - benchmark) ** 2))
    return (
        1 - float(np.sum((y - np.asarray(yhat)) ** 2)) / denom
        if denom > 0
        else float("nan")
    )


def rank_correlation(y: np.ndarray, yhat: np.ndarray) -> float:
    """Spearman correlation (information coefficient); NaN if either side is constant."""
    a = pd.Series(np.asarray(y, float)).rank().to_numpy()
    b = pd.Series(np.asarray(yhat, float)).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def directional_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """Share of rows with y != 0 where sign(yhat) == sign(y)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = y != 0
    return (
        float(np.mean(np.sign(yhat[mask]) == np.sign(y[mask])))
        if mask.any()
        else float("nan")
    )


# ------------------------------------------------------------------ quantiles
def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    diff = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(low)) & (y <= np.asarray(high))))


# ---------------------------------------------------------------- volatility
def qlike(sigma: np.ndarray, sigma_hat: np.ndarray) -> float:
    """QLIKE loss on variances: mean(r - log r - 1), r = sigma^2 / sigma_hat^2; 0 is perfect."""
    ratio = (np.asarray(sigma, float) / np.asarray(sigma_hat, float)) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))
