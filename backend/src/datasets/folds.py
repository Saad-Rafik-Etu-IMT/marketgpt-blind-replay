"""Chronological folds carved from the TRAINING examples only.

Validation is evaluated once and the test set stays sealed, so every tuning
loop (epoch, threshold, hyperparameters, feature groups) runs on these folds.

Cut-offs are calendar dates shared by every asset; per asset, the cut-off
becomes the first bar on or after that date (``first``):

- the fold's training side keeps examples whose LAST LABEL bar is more than
  ``embargo`` bars before ``first`` (purge + embargo);
- the evaluated side keeps examples whose decision bar is on or after
  ``first`` (``available_history``) or whose whole input window is
  (``own_partition``), and whose last label bar is before the next cut-off.

``inner_split`` reproduces ``learning/mlp/protocol.py::inner_split``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from src.datasets.examples import AssetExamples, ContextRule

INNER_FRACTION = 0.2


class FoldError(ValueError):
    """Not enough training history for the requested folds."""


@dataclass(frozen=True)
class Fold:
    """Example bar indices of one asset for one chronological fold."""

    train: tuple[int, ...]
    evaluate: tuple[int, ...]


def _first_bar(dates: Sequence[date], cutoff: date) -> int:
    return next((i for i, d in enumerate(dates) if d >= cutoff), len(dates))


def inner_cutoff(
    decision_dates: Sequence[date], fraction: float = INNER_FRACTION
) -> date:
    """Calendar date starting the last ``fraction`` of the training decisions."""
    if not decision_dates:
        raise FoldError("no training examples to split")
    if not 0 < fraction < 1:
        raise FoldError("fraction must be in (0, 1)")
    ordered = sorted(decision_dates)
    return ordered[int(len(ordered) * (1 - fraction))]


def split_at(
    examples: AssetExamples,
    candidates: Sequence[int],
    start: date,
    end: date | None,
    *,
    embargo: int,
    context: ContextRule,
) -> Fold:
    """Train on ``candidates`` before ``start`` (purged), evaluate on ``[start, end)``."""
    first = _first_bar(examples.dates, start)
    stop = len(examples.dates) if end is None else _first_bar(examples.dates, end)
    train = tuple(t for t in candidates if examples.last_label(t) < first - embargo)
    if context is ContextRule.OWN_PARTITION:
        begins = examples.first_input
    else:

        def begins(t: int) -> int:
            return t

    evaluate = tuple(
        t for t in candidates if begins(t) >= first and examples.last_label(t) < stop
    )
    return Fold(train=train, evaluate=evaluate)


def inner_split(
    examples: AssetExamples, cutoff: date, *, embargo: int, context: ContextRule
) -> Fold:
    """Inner-train / inner-validation inside the training examples."""
    return split_at(
        examples, examples.train, cutoff, None, embargo=embargo, context=context
    )


def walk_forward_cutoffs(
    decision_dates: Sequence[date], folds: int, min_train_fraction: float
) -> list[date]:
    """``folds + 1`` calendar boundaries: fold k evaluates [cut[k], cut[k + 1])."""
    if folds < 1:
        raise FoldError("folds must be >= 1")
    if not 0 < min_train_fraction < 1:
        raise FoldError("min_train_fraction must be in (0, 1)")
    ordered = sorted(set(decision_dates))
    if len(ordered) < 2 * (folds + 1):
        raise FoldError("not enough training dates for the requested folds")
    start = int(len(ordered) * min_train_fraction)
    step = (len(ordered) - start) / folds
    cuts = [ordered[start + int(round(k * step))] for k in range(folds)]
    return cuts + [ordered[-1]]


def walk_forward(
    examples: AssetExamples,
    cutoffs: Sequence[date],
    *,
    embargo: int,
    context: ContextRule,
) -> list[Fold]:
    """Expanding-window folds over the training examples.

    The last fold's end is exclusive of nothing: it runs to the last training
    decision (inclusive), so every evaluated example is a training example.
    """
    folds = []
    for k in range(len(cutoffs) - 1):
        end = None if k == len(cutoffs) - 2 else cutoffs[k + 1]
        folds.append(
            split_at(
                examples,
                examples.train,
                cutoffs[k],
                end,
                embargo=embargo,
                context=context,
            )
        )
    return folds
