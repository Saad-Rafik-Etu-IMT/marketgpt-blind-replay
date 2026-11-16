"""F2 split tests: synthetic weekday calendars only, no network or market data."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.split import (
    DEFAULT_SPLIT_CONFIG_PATH,
    Split,
    SplitConfig,
    SplitError,
    assign_splits,
    build_examples,
    load_split_config,
    validate_gaps,
)


def weekdays(start: date, end: date) -> list[date]:
    days: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


SMALL = {
    "train_start": None,
    "train_end": "2020-03-31",
    "validation_start": "2020-05-01",
    "validation_end": "2020-08-31",
    "test_start": "2020-10-01",
    "test_end": "2020-12-31",
    "lookback": 5,
    "label_horizon": 2,
    "embargo": 3,
}


@pytest.fixture
def small_config() -> SplitConfig:
    return SplitConfig.model_validate(SMALL)


@pytest.fixture
def small_dates() -> list[date]:
    return weekdays(date(2020, 1, 1), date(2021, 2, 26))


def window(t: int, config: SplitConfig) -> range:
    return range(t - config.lookback + 1, t + config.label_horizon + 1)


# --- Config -----------------------------------------------------------------


def test_repository_config_loads_agreed_boundaries() -> None:
    config = load_split_config()
    assert DEFAULT_SPLIT_CONFIG_PATH.name == "split.json"
    assert config.train_start is None
    assert config.train_end == date(2016, 12, 31)
    assert config.validation_start == date(2017, 4, 1)
    assert config.validation_end == date(2019, 12, 31)
    assert config.test_start == date(2020, 2, 1)
    assert config.test_end is None
    assert (config.lookback, config.label_horizon, config.embargo) == (64, 1, 5)
    assert config.required_gap_bars == 6


@pytest.mark.parametrize(
    "override",
    [
        {"validation_start": "2020-03-31"},  # overlaps train_end
        {"validation_start": "2020-02-01"},  # validation starts inside train
        {"validation_end": "2020-04-30"},  # validation end before its start
        {"test_start": "2020-08-31"},  # overlaps validation_end
        {"test_start": "2020-07-01"},  # test inside validation
        {"test_end": "2020-09-30"},  # test end before its start
        {"train_start": "2020-04-01"},  # train start after train end
        {"validation_start": "2020-04-03"},  # gap of 2 weekdays < 2 + 3
        {"test_start": "2020-09-03"},  # gap of 2 weekdays < 2 + 3
        {"lookback": 0},
        {"label_horizon": 0},
        {"embargo": -1},
        {"unexpected": 1},
    ],
)
def test_config_rejects_invalid_boundaries(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SplitConfig.model_validate({**SMALL, **override})


def test_config_gap_requirement_ignores_lookback() -> None:
    # 5 weekdays = label_horizon + embargo is enough even with a huge lookback.
    config = SplitConfig.model_validate(
        {**SMALL, "train_end": "2020-04-03", "validation_start": "2020-04-11", "lookback": 500}
    )
    assert config.required_gap_bars == 5


def test_load_split_config_reports_bad_files(tmp_path: Path) -> None:
    with pytest.raises(SplitError):
        load_split_config(tmp_path / "missing.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SplitError):
        load_split_config(broken)

    overlapping = tmp_path / "overlap.json"
    overlapping.write_text(json.dumps({**SMALL, "validation_start": "2020-01-15"}), encoding="utf-8")
    with pytest.raises(SplitError):
        load_split_config(overlapping)


# --- Assignment and gaps ----------------------------------------------------


def test_sets_do_not_overlap_and_are_ordered(
    small_config: SplitConfig, small_dates: list[date]
) -> None:
    labels = assign_splits(small_dates, small_config)
    by_set = {
        split: [d for d, label in zip(small_dates, labels, strict=True) if label is split]
        for split in (Split.TRAIN, Split.VALIDATION, Split.TEST)
    }
    assert all(by_set.values())
    assert max(by_set[Split.TRAIN]) < min(by_set[Split.VALIDATION])
    assert max(by_set[Split.VALIDATION]) < min(by_set[Split.TEST])
    assert not set(by_set[Split.TRAIN]) & set(by_set[Split.VALIDATION])
    assert not set(by_set[Split.VALIDATION]) & set(by_set[Split.TEST])
    # Bars after test_end belong to no set.
    assert all(label is None for d, label in zip(small_dates, labels, strict=True) if d > date(2020, 12, 31))


def test_bars_before_train_start_are_unassigned(small_dates: list[date]) -> None:
    config = SplitConfig.model_validate({**SMALL, "train_start": "2020-02-03"})
    labels = assign_splits(small_dates, config)
    assert labels[0] is None
    assert labels[small_dates.index(date(2020, 2, 3))] is Split.TRAIN


@pytest.mark.parametrize("gap", ["train/validation", "validation/test"])
def test_gap_shorter_than_horizon_plus_embargo_is_rejected(
    small_config: SplitConfig, small_dates: list[date], gap: str
) -> None:
    after, before = (
        (small_config.train_end, small_config.validation_start)
        if gap == "train/validation"
        else (small_config.validation_end, small_config.test_start)
    )
    gap_days = [d for d in small_dates if after < d < before]
    required = small_config.required_gap_bars

    exact = [d for d in small_dates if d not in gap_days[required:]]
    validate_gaps(exact, small_config)
    build_examples(exact, small_config)

    too_short = [d for d in small_dates if d not in gap_days[required - 1 :]]
    with pytest.raises(SplitError, match=gap):
        validate_gaps(too_short, small_config)
    with pytest.raises(SplitError):
        build_examples(too_short, small_config)


def test_gap_check_skipped_when_asset_does_not_span_it(small_config: SplitConfig) -> None:
    late_listing = weekdays(date(2020, 6, 1), date(2020, 12, 31))
    validate_gaps(late_listing, small_config)
    examples = build_examples(late_listing, small_config)
    assert examples.train == ()
    assert examples.validation and examples.test
    validate_gaps([], small_config)


def test_dates_must_be_strictly_increasing(small_config: SplitConfig) -> None:
    with pytest.raises(SplitError):
        assign_splits([date(2020, 1, 3), date(2020, 1, 2)], small_config)
    with pytest.raises(SplitError):
        assign_splits([date(2020, 1, 2), date(2020, 1, 2)], small_config)


# --- Example windows --------------------------------------------------------


def test_train_and_validation_windows_never_cross_a_boundary(
    small_config: SplitConfig, small_dates: list[date]
) -> None:
    labels = assign_splits(small_dates, small_config)
    examples = build_examples(small_dates, small_config)
    for split, indices in ((Split.TRAIN, examples.train), (Split.VALIDATION, examples.validation)):
        assert indices
        for t in indices:
            assert all(labels[i] is split for i in window(t, small_config))


def test_window_edges_are_exact(small_config: SplitConfig, small_dates: list[date]) -> None:
    labels = assign_splits(small_dates, small_config)
    examples = build_examples(small_dates, small_config)
    lookback, horizon = small_config.lookback, small_config.label_horizon

    train_idx = [i for i, s in enumerate(labels) if s is Split.TRAIN]
    val_idx = [i for i, s in enumerate(labels) if s is Split.VALIDATION]
    test_idx = [i for i, s in enumerate(labels) if s is Split.TEST]

    # Earliest example needs a full look-back; latest needs its label inside the set.
    assert examples.train[0] == train_idx[0] + lookback - 1
    assert examples.train[-1] == train_idx[-1] - horizon
    assert examples.validation[0] == val_idx[0] + lookback - 1
    assert examples.validation[-1] == val_idx[-1] - horizon
    # Every in-set bar between those edges is kept, contiguous sets lose nothing else.
    assert examples.train == tuple(range(train_idx[0] + lookback - 1, train_idx[-1] - horizon + 1))
    # Boundary-crossing candidates one bar beyond each edge are dropped, not reassigned.
    dropped = {examples.train[-1] + 1, examples.validation[0] - 1, examples.validation[-1] + 1}
    for t in dropped:
        assert t not in examples.train + examples.validation + examples.test
    assert examples.test[-1] == test_idx[-1] - horizon


def test_test_examples_use_warm_up_but_are_scored_from_test_start(
    small_config: SplitConfig, small_dates: list[date]
) -> None:
    labels = assign_splits(small_dates, small_config)
    examples = build_examples(small_dates, small_config)
    first_test = labels.index(Split.TEST)

    # The first test decision is the first test bar: its look-back is warm-up.
    assert examples.test[0] == first_test
    assert any(labels[i] is not Split.TEST for i in window(first_test, small_config))
    for t in examples.test:
        assert small_dates[t] >= small_config.test_start
        assert all(labels[i] is Split.TEST for i in range(t, t + small_config.label_horizon + 1))

    # Warm-up (gap) bars are never decisions in any set.
    all_examples = set(examples.train + examples.validation + examples.test)
    assert all(labels[t] is not Split.GAP for t in all_examples)
    assert not set(examples.test) & set(examples.train + examples.validation)


def test_test_examples_need_a_full_look_back(small_config: SplitConfig) -> None:
    starts_in_test = weekdays(date(2020, 10, 1), date(2020, 12, 31))
    examples = build_examples(starts_in_test, small_config)
    assert examples.test[0] == small_config.lookback - 1


def test_split_is_deterministic(small_config: SplitConfig, small_dates: list[date]) -> None:
    as_datetimes = [datetime(d.year, d.month, d.day) for d in small_dates]
    first = build_examples(small_dates, small_config)
    assert build_examples(small_dates, small_config) == first
    assert build_examples(tuple(small_dates), small_config) == first
    assert build_examples(as_datetimes, small_config) == first
    assert assign_splits(as_datetimes, small_config) == assign_splits(small_dates, small_config)


def test_repository_config_on_synthetic_calendar() -> None:
    config = load_split_config()
    dates = weekdays(date(2010, 1, 4), date(2022, 12, 30))
    labels = assign_splits(dates, config)
    examples = build_examples(dates, config)
    assert examples.train and examples.validation and examples.test
    for split, indices in ((Split.TRAIN, examples.train), (Split.VALIDATION, examples.validation)):
        for t in indices:
            assert all(labels[i] is split for i in window(t, config))
    assert min(dates[t] for t in examples.test) >= config.test_start
