"""Tests for data_loader — Phase 1, Module 4.

Persists validated OHLCV bars into the database with strict
all-or-nothing semantics:

  * If the validation report is not valid → load nothing.
  * If any (instrument, date) already exists in the DB → load nothing.
  * On success, every bar is committed in a single transaction.
"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.market import OHLCVBarORM
from src.schemas.market import OHLCVBar, RowError, ValidationReport
from src.services.data_loader import load_bars


def _report(*bars: OHLCVBar, is_valid: bool = True) -> ValidationReport:
    """Build a ValidationReport carrying the given bars."""
    return ValidationReport(
        is_valid=is_valid,
        total_rows=len(bars),
        valid_rows=len(bars) if is_valid else 0,
        errors=[] if is_valid else [RowError(row_number=2, message="forced")],
        bars=list(bars) if is_valid else [],
    )


def _bar(day: int, **overrides: object) -> OHLCVBar:
    """A well-formed bar on 2023-01-{day}."""
    return OHLCVBar(
        date=date(2023, 1, day),
        open=overrides.get("open", 100.0),
        high=overrides.get("high", 105.0),
        low=overrides.get("low", 98.0),
        close=overrides.get("close", 102.0),
        volume=overrides.get("volume", 1_000_000),
        adjusted_close=overrides.get("adjusted_close", 102.0),
    )


# --- Happy path -------------------------------------------------------------


def test_load_bars_persists_all_rows(db_session: Session) -> None:
    """A valid report's bars are persisted under the given instrument."""
    result = load_bars(db_session, "SPY", _report(_bar(3), _bar(4), _bar(5)))

    assert result.success is True
    assert result.loaded_count == 3
    assert result.error is None

    stored = db_session.execute(select(OHLCVBarORM)).scalars().all()
    assert len(stored) == 3
    assert {b.instrument for b in stored} == {"SPY"}
    assert {b.date for b in stored} == {
        date(2023, 1, 3),
        date(2023, 1, 4),
        date(2023, 1, 5),
    }


def test_load_bars_preserves_field_values(db_session: Session) -> None:
    """Every domain field is persisted with the same value."""
    result = load_bars(db_session, "SPY", _report(_bar(3)))
    assert result.success is True

    stored = db_session.execute(select(OHLCVBarORM)).scalar_one()
    assert stored.open == 100.0
    assert stored.high == 105.0
    assert stored.low == 98.0
    assert stored.close == 102.0
    assert stored.volume == 1_000_000
    assert stored.adjusted_close == 102.0


def test_loads_for_different_instruments_independently(db_session: Session) -> None:
    """Same date across different instruments coexists."""
    assert load_bars(db_session, "SPY", _report(_bar(3))).success is True
    assert load_bars(db_session, "QQQ", _report(_bar(3))).success is True

    rows = (
        db_session.execute(
            select(OHLCVBarORM.instrument).order_by(OHLCVBarORM.instrument)
        )
        .scalars()
        .all()
    )
    assert rows == ["QQQ", "SPY"]


# --- Refusal: invalid report -----------------------------------------------


def test_refuses_invalid_report(db_session: Session) -> None:
    """A report with is_valid=False is refused — nothing is persisted."""
    bad_report = _report(_bar(3), is_valid=False)
    result = load_bars(db_session, "SPY", bad_report)

    assert result.success is False
    assert result.loaded_count == 0
    assert result.error is not None
    assert "invalid" in result.error.lower()

    assert db_session.execute(select(OHLCVBarORM)).first() is None


def test_refuses_empty_report(db_session: Session) -> None:
    """A valid-but-empty report (no bars) is refused — nothing to load."""
    result = load_bars(db_session, "SPY", _report())
    assert result.success is False
    assert result.loaded_count == 0
    assert result.error is not None


# --- Refusal: conflicts -----------------------------------------------------


def test_refuses_when_date_already_loaded(db_session: Session) -> None:
    """If any (instrument, date) is already in DB, the whole load is refused."""
    # Initial load.
    assert load_bars(db_session, "SPY", _report(_bar(3), _bar(4))).success is True

    # A subsequent load whose first bar collides is fully rejected.
    result = load_bars(db_session, "SPY", _report(_bar(4), _bar(5)))
    assert result.success is False
    assert result.loaded_count == 0
    assert result.error is not None
    assert "conflict" in result.error.lower() or "exists" in result.error.lower()
    assert "2023-01-04" in result.error  # the offending date is named

    # The non-conflicting bar (day 5) must not have been written.
    dates = (
        db_session.execute(select(OHLCVBarORM.date).order_by(OHLCVBarORM.date))
        .scalars()
        .all()
    )
    assert dates == [date(2023, 1, 3), date(2023, 1, 4)]


def test_conflict_check_is_instrument_scoped(db_session: Session) -> None:
    """A date present for SPY does not block the same date for QQQ."""
    assert load_bars(db_session, "SPY", _report(_bar(3))).success is True
    assert load_bars(db_session, "QQQ", _report(_bar(3))).success is True


# --- Transactional integrity ----------------------------------------------


def test_loaded_count_matches_report(db_session: Session) -> None:
    """loaded_count is exactly len(report.bars) on success."""
    result = load_bars(db_session, "SPY", _report(_bar(3), _bar(4), _bar(5), _bar(6)))
    assert result.success is True
    assert result.loaded_count == 4
