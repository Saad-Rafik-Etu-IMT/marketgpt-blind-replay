"""Persistence of validated OHLCV bars.

Bridges Module 2's ``ValidationReport`` and Module 3's ORM. Loads are
strictly **all-or-nothing**:

  1. If the report is not valid, refuse — nothing is persisted.
  2. If the report is empty, refuse — nothing to do.
  3. If any (instrument, date) in the report already exists in the DB,
     refuse the entire load and name the first offending date.
  4. Otherwise, persist every bar inside a single transaction.

Rationale: silently skipping duplicates or silently overwriting would
corrupt the research record. A re-upload requires explicit intent and
must come through a future ``replace`` endpoint.

Phase 1, Module 4.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.models.market import OHLCVBarORM
from src.schemas.market import ValidationReport


@dataclass(frozen=True)
class LoadResult:
    """Outcome of a load attempt.

    ``success`` is True only when every bar in the report was persisted.
    On failure, ``error`` carries a human-readable explanation and
    ``loaded_count`` is 0.
    """

    success: bool
    loaded_count: int
    error: str | None = None


def load_bars(
    session: Session, instrument: str, report: ValidationReport
) -> LoadResult:
    """Persist a validated report's bars under ``instrument``."""
    refusal = _refusal_reason(session, instrument, report)
    if refusal is not None:
        return LoadResult(success=False, loaded_count=0, error=refusal)

    try:
        session.add_all(
            OHLCVBarORM(
                instrument=instrument,
                date=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                adjusted_close=bar.adjusted_close,
            )
            for bar in report.bars
        )
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        return LoadResult(success=False, loaded_count=0, error=f"database error: {exc}")

    return LoadResult(success=True, loaded_count=len(report.bars))


def _refusal_reason(
    session: Session, instrument: str, report: ValidationReport
) -> str | None:
    """Return a refusal reason string, or None if the load should proceed."""
    if not report.is_valid:
        return "refusing to load: report is invalid"
    if not report.bars:
        return "refusing to load: report contains no bars"

    conflicting_date = _first_conflicting_date(session, instrument, report)
    if conflicting_date is not None:
        return (
            f"conflict: {instrument} already has data for {conflicting_date}; "
            "no rows loaded"
        )
    return None


def _first_conflicting_date(
    session: Session, instrument: str, report: ValidationReport
) -> str | None:
    """Return the ISO date of the first already-loaded bar, or None."""
    incoming_dates = [bar.date for bar in report.bars]
    existing = (
        session.execute(
            select(OHLCVBarORM.date)
            .where(OHLCVBarORM.instrument == instrument)
            .where(OHLCVBarORM.date.in_(incoming_dates))
        )
        .scalars()
        .all()
    )
    if not existing:
        return None
    # Name the earliest conflicting date — stable and diagnosable.
    return min(existing).isoformat()


def has_quarantined_wti(db: Session) -> bool:
    """Detect the previously bundled SPY copy, without deleting user data."""
    rows = list(
        db.scalars(
            select(OHLCVBarORM)
            .where(OHLCVBarORM.instrument == "CL=F")
            .order_by(OHLCVBarORM.date)
            .limit(5)
        )
    )
    if len(rows) < 5:
        return False
    for row in rows:
        spy = db.scalar(
            select(OHLCVBarORM).where(
                OHLCVBarORM.instrument == "SPY", OHLCVBarORM.date == row.date
            )
        )
        if spy is None or any(
            getattr(spy, field) != getattr(row, field)
            for field in ("open", "high", "low", "close", "volume")
        ):
            return False
    return True
