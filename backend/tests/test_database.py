"""Tests for the persistence foundation — Phase 1, Module 3.

Covers:
  * Engine + session factory build cleanly from a URL.
  * ``Base.metadata.create_all`` creates the OHLCV bars table.
  * A bar round-trips through the ORM with all fields intact.
  * Unique constraint on (instrument, date) is enforced.
  * SQLite file engines enable WAL mode (per architecture decision).

Sessions are scoped per-test via the ``db_session`` fixture in
``conftest.py``: a fresh in-memory SQLite for every test, fully isolated.
"""

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database import Base, build_engine, build_session_factory
from src.models.market import OHLCVBarORM

# --- Engine / metadata ------------------------------------------------------


def test_engine_builds_from_url() -> None:
    """build_engine returns a working SQLAlchemy engine for an in-memory DB."""
    engine = build_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar_one()
    assert result == 1


def test_create_all_creates_ohlcv_bars_table() -> None:
    """Base.metadata.create_all materialises the ohlcv_bars table."""
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    table_names = inspect(engine).get_table_names()
    assert "ohlcv_bars" in table_names


def test_ohlcv_bars_table_has_expected_columns() -> None:
    """The ORM table exposes every domain field plus a surrogate id."""
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    column_names = {col["name"] for col in inspect(engine).get_columns("ohlcv_bars")}
    assert column_names >= {
        "id",
        "instrument",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
    }


# --- Round-trip -------------------------------------------------------------


def test_bar_round_trip(db_session: Session) -> None:
    """A persisted bar reads back with every field equal."""
    bar = OHLCVBarORM(
        instrument="SPY",
        date=date(2023, 1, 3),
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=1_000_000,
        adjusted_close=102.0,
    )
    db_session.add(bar)
    db_session.commit()

    fetched = db_session.execute(select(OHLCVBarORM)).scalar_one()
    assert fetched.instrument == "SPY"
    assert fetched.date == date(2023, 1, 3)
    assert fetched.open == 100.0
    assert fetched.high == 105.0
    assert fetched.low == 98.0
    assert fetched.close == 102.0
    assert fetched.volume == 1_000_000
    assert fetched.adjusted_close == 102.0
    assert fetched.id is not None


def test_adjusted_close_nullable(db_session: Session) -> None:
    """adjusted_close persists as NULL when not provided."""
    db_session.add(
        OHLCVBarORM(
            instrument="SPY",
            date=date(2023, 1, 3),
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=1_000_000,
        )
    )
    db_session.commit()
    fetched = db_session.execute(select(OHLCVBarORM)).scalar_one()
    assert fetched.adjusted_close is None


# --- Constraints ------------------------------------------------------------


def test_unique_instrument_date(db_session: Session) -> None:
    """Two bars with the same (instrument, date) violate the unique constraint."""
    common = dict(
        instrument="SPY",
        date=date(2023, 1, 3),
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=1_000_000,
    )
    db_session.add(OHLCVBarORM(**common))
    db_session.commit()

    db_session.add(OHLCVBarORM(**common))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_same_date_different_instruments_allowed(db_session: Session) -> None:
    """The unique constraint is composite — same date is fine across instruments."""
    common = dict(
        date=date(2023, 1, 3),
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=1_000_000,
    )
    db_session.add(OHLCVBarORM(instrument="SPY", **common))
    db_session.add(OHLCVBarORM(instrument="QQQ", **common))
    db_session.commit()

    instruments = (
        db_session.execute(
            select(OHLCVBarORM.instrument).order_by(OHLCVBarORM.instrument)
        )
        .scalars()
        .all()
    )
    assert instruments == ["QQQ", "SPY"]


# --- Session factory --------------------------------------------------------


def test_session_factory_yields_working_session() -> None:
    """build_session_factory returns a factory producing usable Sessions."""
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        assert isinstance(session, Session)
        assert session.execute(text("SELECT 1")).scalar_one() == 1


# --- WAL mode on file-backed SQLite ----------------------------------------


def test_wal_mode_enabled_for_file_sqlite(tmp_path: Path) -> None:
    """File-backed SQLite engines enable WAL mode per architecture.md."""
    db_path = tmp_path / "trading_lab.sqlite"
    engine = build_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar_one()
    assert str(journal_mode).lower() == "wal"


def test_memory_sqlite_does_not_require_wal() -> None:
    """In-memory SQLite cannot use WAL; the engine must not crash trying."""
    engine = build_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar_one()
    assert str(journal_mode).lower() != "wal"
