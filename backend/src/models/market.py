"""ORM model for OHLCV market-data bars.

Mirrors the ``OHLCVBar`` Pydantic schema (§4.1) plus an ``instrument``
discriminator: a single table stores bars for every instrument. A
composite unique constraint on ``(instrument, date)`` prevents duplicate
ingestion of the same trading day for the same instrument.
"""

from datetime import date as date_type

from sqlalchemy import Date, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class OHLCVBarORM(Base):
    """A persisted OHLCV bar."""

    __tablename__ = "ohlcv_bars"
    __table_args__ = (
        UniqueConstraint("instrument", "date", name="uq_ohlcv_instrument_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    date: Mapped[date_type] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    adjusted_close: Mapped[float | None] = mapped_column(Float, nullable=True)
