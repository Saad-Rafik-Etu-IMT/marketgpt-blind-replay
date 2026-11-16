"""ORM model for replay sessions.

A ReplaySession owns one piece of state that must never be wrong: the
``current_bar_index`` pointer (§5 invariant 6). It is the sole place
"now" is defined in the system.
"""

from datetime import date as date_type
from datetime import datetime

from sqlalchemy import Date, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class ReplaySessionORM(Base):
    """A named replay session on a specific instrument and date range."""

    __tablename__ = "replay_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    start_date: Mapped[date_type] = mapped_column(Date, nullable=False)
    end_date: Mapped[date_type] = mapped_column(Date, nullable=False)

    # Pointer into the bars array (ordered by date) within [start, end].
    # 0 = the first bar is "now"; the cockpit shows bars [0..N] inclusive.
    current_bar_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # active | paused | completed
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")

    initial_cash: Mapped[float] = mapped_column(
        Float, nullable=False, default=100_000.0
    )

    # Contract multiplier (point value) applied to every position in this
    # session. 1.0 for cash equities; e.g. 50.0 for an E-mini S&P future.
    multiplier: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    simulation_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    fee_per_unit: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_bps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
