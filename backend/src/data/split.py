"""F2: chronological train/validation/test split with embargoed gaps.

Pure functions over a strictly increasing sequence of bar dates for one asset.
No database, network or market data is touched here; boundaries come from a
JSON config so they are never hard-coded.

Example semantics, for a decision bar at index ``t``:

- inputs are bars ``t - lookback + 1 .. t``;
- the label covers bars ``t + 1 .. t + label_horizon``.

A train/validation example is kept only if its inputs *and* label bars all lie
inside its own set. A test example must have its decision and label bars inside
the test set; its inputs may reach back before ``test_start`` as warm-up
context (past data at decision time), but those warm-up bars are never scored.
Each gap must contain at least ``label_horizon + embargo`` bars of the asset.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from src.config import BACKEND_DIR

DEFAULT_SPLIT_CONFIG_PATH: Path = BACKEND_DIR / "config" / "split.json"


class SplitError(ValueError):
    """Raised when a split config or the dates it is applied to are invalid."""


class Split(str, Enum):
    TRAIN = "train"
    GAP = "gap"
    VALIDATION = "validation"
    TEST = "test"


def _weekdays_between(after: date, before: date) -> int:
    """Weekdays strictly between two dates: an upper bound on trading bars."""
    count = 0
    day = after + timedelta(days=1)
    while day < before:
        if day.weekday() < 5:
            count += 1
        day += timedelta(days=1)
    return count


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    train_start: date | None = None
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date | None = None
    lookback: int = Field(default=64, ge=1)
    label_horizon: int = Field(default=1, ge=1)
    embargo: int = Field(default=5, ge=0)

    @property
    def required_gap_bars(self) -> int:
        return self.label_horizon + self.embargo

    @model_validator(mode="after")
    def _check_boundaries(self) -> Self:
        if self.train_start is not None and self.train_start > self.train_end:
            raise ValueError("train_start must be on or before train_end")
        if self.validation_start > self.validation_end:
            raise ValueError("validation_start must be on or before validation_end")
        if self.test_end is not None and self.test_start > self.test_end:
            raise ValueError("test_start must be on or before test_end")
        if self.validation_start <= self.train_end:
            raise ValueError("validation must start after train_end (sets overlap or are unordered)")
        if self.test_start <= self.validation_end:
            raise ValueError("test must start after validation_end (sets overlap or are unordered)")
        # Weekdays are an upper bound on trading bars, so failing here means the
        # gap can never satisfy the rule; the exact check runs against real bars.
        for name, after, before in (
            ("train/validation", self.train_end, self.validation_start),
            ("validation/test", self.validation_end, self.test_start),
        ):
            if _weekdays_between(after, before) < self.required_gap_bars:
                raise ValueError(
                    f"{name} gap is too short: needs at least "
                    f"label_horizon + embargo = {self.required_gap_bars} bars"
                )
        return self


def load_split_config(path: Path = DEFAULT_SPLIT_CONFIG_PATH) -> SplitConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SplitConfig.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SplitError(f"invalid split config {path}: {exc}") from exc


def _normalize_dates(dates: Sequence[date]) -> list[date]:
    normalized = [d.date() if isinstance(d, datetime) else d for d in dates]
    for previous, current in zip(normalized, normalized[1:], strict=False):
        if current <= previous:
            raise SplitError(f"dates must be strictly increasing: {previous} then {current}")
    return normalized


def _split_of(day: date, config: SplitConfig) -> Split | None:
    if day <= config.train_end:
        if config.train_start is not None and day < config.train_start:
            return None
        return Split.TRAIN
    if day < config.validation_start:
        return Split.GAP
    if day <= config.validation_end:
        return Split.VALIDATION
    if day < config.test_start:
        return Split.GAP
    if config.test_end is not None and day > config.test_end:
        return None
    return Split.TEST


def assign_splits(dates: Sequence[date], config: SplitConfig) -> list[Split | None]:
    """Label each bar; ``None`` marks bars outside every configured range."""
    return [_split_of(day, config) for day in _normalize_dates(dates)]


def validate_gaps(dates: Sequence[date], config: SplitConfig) -> None:
    """Each gap the asset spans must hold at least ``label_horizon + embargo`` bars."""
    normalized = _normalize_dates(dates)
    for name, after, before in (
        ("train/validation", config.train_end, config.validation_start),
        ("validation/test", config.validation_end, config.test_start),
    ):
        # A gap only separates sets if the asset has bars on both sides of it.
        if not normalized or normalized[0] > after or normalized[-1] < before:
            continue
        bars = sum(1 for day in normalized if after < day < before)
        if bars < config.required_gap_bars:
            raise SplitError(
                f"{name} gap has {bars} bars; needs at least "
                f"label_horizon + embargo = {config.required_gap_bars}"
            )


@dataclass(frozen=True)
class SplitExamples:
    """Decision-bar indices of the valid examples in each set."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def build_examples(dates: Sequence[date], config: SplitConfig) -> SplitExamples:
    labels = assign_splits(dates, config)
    validate_gaps(dates, config)
    lookback, horizon = config.lookback, config.label_horizon
    selected: dict[Split, list[int]] = {
        Split.TRAIN: [],
        Split.VALIDATION: [],
        Split.TEST: [],
    }

    for t, split in enumerate(labels):
        if split not in selected:
            continue
        first_input, last_label = t - lookback + 1, t + horizon
        if first_input < 0 or last_label >= len(labels):
            continue
        # Test inputs may use earlier bars as warm-up; only scored bars must be test.
        first_checked = t if split is Split.TEST else first_input
        if all(labels[i] is split for i in range(first_checked, last_label + 1)):
            selected[split].append(t)

    return SplitExamples(
        train=tuple(selected[Split.TRAIN]),
        validation=tuple(selected[Split.VALIDATION]),
        test=tuple(selected[Split.TEST]),
    )
