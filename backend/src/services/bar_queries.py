"""Shared, ordered market-data queries for one replay session."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.market import OHLCVBarORM
from src.models.session import ReplaySessionORM


def bar_at(db: Session, session: ReplaySessionORM, index: int) -> OHLCVBarORM | None:
    """Fetch only the requested bar instead of materializing the timeline."""
    if index < 0:
        return None
    return db.scalar(
        select(OHLCVBarORM)
        .where(OHLCVBarORM.instrument == session.instrument)
        .where(OHLCVBarORM.date >= session.start_date)
        .where(OHLCVBarORM.date <= session.end_date)
        .order_by(OHLCVBarORM.date)
        .offset(index)
        .limit(1)
    )
