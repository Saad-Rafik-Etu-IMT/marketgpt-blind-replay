"""Tests for OHLCV row-level validation — the ``OHLCVBar`` schema.

Phase 1, Module 1. Written before the schema exists (TDD: red first).
``OHLCVBar`` *is* the per-row validator: constructing one enforces every
OHLC integrity invariant. As later modules land, full-CSV validation
tests join this file.
"""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas.market import OHLCVBar
from src.services.data_validator import validate_csv_content, validate_csv_file


def _valid_bar_kwargs(**overrides: object) -> dict[str, object]:
    """A known-good OHLCV bar. Override individual fields per test."""
    base: dict[str, object] = {
        "date": date(2023, 1, 3),
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 102.0,
        "volume": 1_000_000,
        "adjusted_close": 102.0,
    }
    base.update(overrides)
    return base


# --- Acceptance cases -------------------------------------------------------


def test_valid_bar_is_accepted() -> None:
    """A well-formed bar constructs without error."""
    bar = OHLCVBar(**_valid_bar_kwargs())
    assert bar.open == 100.0
    assert bar.high == 105.0
    assert bar.volume == 1_000_000


def test_adjusted_close_is_optional() -> None:
    """adjusted_close may be omitted entirely and defaults to None."""
    kwargs = _valid_bar_kwargs()
    del kwargs["adjusted_close"]
    bar = OHLCVBar(**kwargs)
    assert bar.adjusted_close is None


def test_date_coerced_from_iso_string() -> None:
    """A 'YYYY-MM-DD' string is parsed into a date (CSV ingestion path)."""
    bar = OHLCVBar(**_valid_bar_kwargs(date="2023-01-03"))
    assert bar.date == date(2023, 1, 3)


def test_flat_bar_is_accepted() -> None:
    """A zero-range bar (open == high == low == close) is valid (e.g. a halt)."""
    bar = OHLCVBar(**_valid_bar_kwargs(open=100.0, high=100.0, low=100.0, close=100.0))
    assert bar.high == bar.low == 100.0


def test_zero_volume_is_accepted() -> None:
    """Zero volume is valid — an illiquid or halted session."""
    bar = OHLCVBar(**_valid_bar_kwargs(volume=0))
    assert bar.volume == 0


def test_high_equal_to_max_open_close_is_accepted() -> None:
    """high == max(open, close) is an inclusive boundary."""
    bar = OHLCVBar(**_valid_bar_kwargs(open=100.0, close=105.0, high=105.0, low=98.0))
    assert bar.high == 105.0


def test_low_equal_to_min_open_close_is_accepted() -> None:
    """low == min(open, close) is an inclusive boundary."""
    bar = OHLCVBar(**_valid_bar_kwargs(open=100.0, close=105.0, low=100.0, high=106.0))
    assert bar.low == 100.0


# --- Rejection cases: OHLC integrity ---------------------------------------


def test_rejects_high_below_open() -> None:
    """high < open violates high >= max(open, close)."""
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(open=110.0, high=105.0))


def test_rejects_high_below_close() -> None:
    """high < close violates high >= max(open, close)."""
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(close=110.0, high=105.0))


def test_rejects_low_above_open() -> None:
    """low > open violates low <= min(open, close)."""
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(open=90.0, low=95.0))


def test_rejects_low_above_close() -> None:
    """low > close violates low <= min(open, close)."""
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(close=90.0, low=95.0))


def test_rejects_inverted_high_low() -> None:
    """An inverted bar (high < low) is rejected."""
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(high=95.0, low=100.0, open=97.0, close=98.0))


# --- Rejection cases: non-negativity ---------------------------------------


def test_rejects_negative_open() -> None:
    """A negative open price is rejected (Field ge=0)."""
    # low lowered to -1.0 so ONLY the open >= 0 constraint fails.
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(open=-1.0, low=-1.0))


def test_rejects_negative_close() -> None:
    """A negative close price is rejected (Field ge=0)."""
    # low lowered to -1.0 so ONLY the close >= 0 constraint fails.
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(close=-1.0, low=-1.0))


def test_rejects_negative_volume() -> None:
    """Negative volume is rejected (Field ge=0)."""
    with pytest.raises(ValidationError):
        OHLCVBar(**_valid_bar_kwargs(volume=-1))


# ===========================================================================
# CSV-level validation — Phase 1, Module 2 (data_validator.py)
# ===========================================================================

_HEADER = "date,open,high,low,close,volume,adjusted_close"

# A reusable well-formed data row (becomes CSV line 2 when used alone).
_GOOD_ROW = "2023-01-03,100,105,98,102,1000000,102"


def _csv(*data_rows: str, header: str = _HEADER) -> str:
    """Assemble CSV text from a header and zero or more data rows."""
    return "\n".join([header, *data_rows]) + "\n"


# --- Acceptance -------------------------------------------------------------


def test_valid_csv_passes() -> None:
    """A clean multi-row CSV validates with zero errors."""
    report = validate_csv_content(
        _csv(
            "2023-01-03,100,105,98,102,1000000,102",
            "2023-01-04,102,108,101,107,1200000,107",
            "2023-01-05,107,109,104,105,900000,105",
        )
    )
    assert report.is_valid is True
    assert report.total_rows == 3
    assert report.valid_rows == 3
    assert report.errors == []


def test_valid_csv_returns_parsed_bars() -> None:
    """A valid report carries the parsed OHLCVBar objects."""
    report = validate_csv_content(_csv(_GOOD_ROW))
    assert len(report.bars) == 1
    assert isinstance(report.bars[0], OHLCVBar)
    assert report.bars[0].close == 102.0


# --- File-level failures ----------------------------------------------------


def test_detects_missing_required_column() -> None:
    """A file missing a required column fails before row validation."""
    report = validate_csv_content(
        _csv(
            "2023-01-03,100,105,98,102,102",
            header="date,open,high,low,close,adjusted_close",
        )
    )
    assert report.is_valid is False
    assert any("volume" in err.message for err in report.errors)


def test_empty_csv_is_invalid() -> None:
    """Empty content (no header) is reported as a file-level failure."""
    report = validate_csv_content("")
    assert report.is_valid is False
    assert report.errors


def test_header_only_csv_is_invalid() -> None:
    """A header with no data rows is invalid — nothing to load."""
    report = validate_csv_content(_csv())
    assert report.is_valid is False
    assert any("no data" in err.message.lower() for err in report.errors)


# --- Row-level failures -----------------------------------------------------


def test_detects_missing_value_in_row() -> None:
    """An empty cell in a required column is reported for that field."""
    report = validate_csv_content(_csv("2023-01-03,,105,98,102,1000000,102"))
    assert report.is_valid is False
    assert any(err.field == "open" for err in report.errors)


def test_short_row_reported_as_missing_values() -> None:
    """A row with fewer fields than the header yields missing-value errors."""
    report = validate_csv_content(_csv("2023-01-03,100,105"))
    assert report.is_valid is False
    assert any(err.field in {"close", "volume"} for err in report.errors)


def test_detects_integrity_violation_row() -> None:
    """A row violating OHLC integrity is rejected (high < close here)."""
    report = validate_csv_content(_csv("2023-01-03,100,95,98,102,1000000,102"))
    assert report.is_valid is False
    assert report.valid_rows == 0


def test_detects_negative_volume_row() -> None:
    """A negative volume cell is reported for the volume field."""
    report = validate_csv_content(_csv("2023-01-03,100,105,98,102,-5,102"))
    assert report.is_valid is False
    assert any(err.field == "volume" for err in report.errors)


def test_detects_non_numeric_value() -> None:
    """A non-numeric price cell is reported for that field."""
    report = validate_csv_content(_csv("2023-01-03,abc,105,98,102,1000000,102"))
    assert report.is_valid is False
    assert any(err.field == "open" for err in report.errors)


def test_detects_bad_date_format() -> None:
    """An unparseable date is reported for the date field."""
    report = validate_csv_content(_csv("not-a-date,100,105,98,102,1000000,102"))
    assert report.is_valid is False
    assert any(err.field == "date" for err in report.errors)


def test_detects_duplicate_date() -> None:
    """Two rows sharing a date trigger a duplicate-date error."""
    report = validate_csv_content(
        _csv(
            "2023-01-03,100,105,98,102,1000000,102",
            "2023-01-03,101,106,99,103,1100000,103",
        )
    )
    assert report.is_valid is False
    assert any("duplicate" in err.message.lower() for err in report.errors)


def test_detects_non_ascending_dates() -> None:
    """A date that precedes the previous row triggers an ordering error."""
    report = validate_csv_content(
        _csv(
            "2023-01-04,100,105,98,102,1000000,102",
            "2023-01-03,101,106,99,103,1100000,103",
        )
    )
    assert report.is_valid is False
    assert any("not strictly after" in err.message for err in report.errors)


# --- Aggregation behaviour --------------------------------------------------


def test_collects_multiple_errors_without_stopping() -> None:
    """Validation does not stop at the first bad row — all are reported."""
    report = validate_csv_content(
        _csv(
            "2023-01-03,100,95,98,102,1000000,102",  # line 2: bad high
            "2023-01-04,100,105,98,102,-5,102",  # line 3: bad volume
        )
    )
    assert report.is_valid is False
    assert {2, 3} <= {err.row_number for err in report.errors}


def test_row_numbers_match_csv_lines() -> None:
    """Error row numbers are 1-based CSV line numbers (header is line 1)."""
    report = validate_csv_content(
        _csv(
            "2023-01-03,100,105,98,102,1000000,102",  # line 2: valid
            "2023-01-04,100,95,98,102,1000000,102",  # line 3: bad high
        )
    )
    assert all(err.row_number == 3 for err in report.errors)


def test_valid_rows_counted_with_mixed_rows() -> None:
    """valid_rows counts only fully valid rows; total_rows counts all."""
    report = validate_csv_content(
        _csv(
            "2023-01-03,100,105,98,102,1000000,102",  # valid
            "2023-01-04,100,95,98,102,1000000,102",  # invalid
            "2023-01-05,107,109,104,105,900000,105",  # valid
        )
    )
    assert report.total_rows == 3
    assert report.valid_rows == 2
    assert len(report.bars) == 2


# --- File-path entry point --------------------------------------------------


def test_validate_csv_file_reads_from_disk(tmp_path: Path) -> None:
    """validate_csv_file reads and validates a file on disk."""
    path = tmp_path / "bars.csv"
    path.write_text(_csv(_GOOD_ROW), encoding="utf-8")
    report = validate_csv_file(path)
    assert report.is_valid is True
    assert report.valid_rows == 1


def test_validate_csv_file_missing_file(tmp_path: Path) -> None:
    """A nonexistent path yields a file-level failure, not an exception."""
    report = validate_csv_file(tmp_path / "does_not_exist.csv")
    assert report.is_valid is False
    assert any("not found" in err.message.lower() for err in report.errors)
