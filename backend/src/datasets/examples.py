"""Decision-bar selection per partition, with a configurable input-context rule.

Indexing: example ``t`` is decided at the close of bar ``t``. Its inputs use
bars ``t - lookback .. t``; its label uses bars ``t + 1 .. t + horizon``.

Two context rules (decision recorded 2026-11-20, option 3: configurable):

- ``own_partition`` (default, the README invariant): train and validation
  examples keep inputs AND label inside their own partition; test examples
  may use earlier bars as inputs (``split.py`` warm-up rule).
- ``available_history``: inputs may use any earlier bar, in every partition,
  because those prices were public at the decision date. Decision and label
  bars must still lie in the example's own partition, and gaps still hold
  ``horizon + embargo`` bars, so no label crosses a partition boundary.

Neither rule lets fitted state (scalers, thresholds) see validation or test
rows: fitting is the builder's job and uses training rows only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from src.data.split import Split, SplitConfig, SplitError, assign_splits, validate_gaps


class ContextRule(str, Enum):
    OWN_PARTITION = "own_partition"
    AVAILABLE_HISTORY = "available_history"


@dataclass(frozen=True)
class AssetExamples:
    """Selected decision bars of one asset, by partition."""

    dates: tuple[date, ...]
    lookback: int
    horizon: int
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def first_input(self, t: int) -> int:
        return t - self.lookback

    def last_label(self, t: int) -> int:
        return t + self.horizon


def select_examples(
    dates: Sequence[date],
    config: SplitConfig,
    *,
    lookback: int,
    horizon: int,
    context: ContextRule = ContextRule.OWN_PARTITION,
) -> AssetExamples:
    """Decision bars whose inputs and label satisfy ``context`` and the split."""
    if lookback < 0 or horizon < 1:
        raise SplitError("lookback must be >= 0 and horizon >= 1")
    labels = assign_splits(dates, config)
    validate_gaps(dates, config.model_copy(update={"label_horizon": horizon}))
    chosen: dict[Split, list[int]] = {
        Split.TRAIN: [],
        Split.VALIDATION: [],
        Split.TEST: [],
    }
    for t, split in enumerate(labels):
        if split not in chosen:
            continue
        first, last = t - lookback, t + horizon
        if first < 0 or last >= len(labels):
            continue
        own_inputs = context is ContextRule.OWN_PARTITION and split is not Split.TEST
        checked_from = first if own_inputs else t
        if all(labels[i] is split for i in range(checked_from, last + 1)):
            chosen[split].append(t)
    return AssetExamples(
        dates=tuple(d if not hasattr(d, "date") else d.date() for d in dates),
        lookback=lookback,
        horizon=horizon,
        train=tuple(chosen[Split.TRAIN]),
        validation=tuple(chosen[Split.VALIDATION]),
        test=tuple(chosen[Split.TEST]),
    )
