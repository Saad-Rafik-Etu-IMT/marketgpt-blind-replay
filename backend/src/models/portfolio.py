"""ORM models for positions and portfolio snapshots."""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PositionORM(Base):
    """Current position for a tradeable contract within one session.

    One row per (session, contract_key), using the instrument symbol as
    the contract key. Updated in place as fills land.
    """

    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("session_id", "contract_key", name="uq_pos_session_contract"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("replay_sessions.session_id"), nullable=False, index=True
    )
    instrument: Mapped[str] = mapped_column(String(32), nullable=False)

    # Stable identity of the traded contract: the session instrument.
    contract_key: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    instrument_kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="underlying"
    )

    # long | short | flat
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_entry_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Contract multiplier (point value). 1.0 for cash equities; e.g. 50.0
    # for an E-mini S&P 500 future. Carried per position so P&L scales.
    multiplier: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)


class PortfolioSnapshotORM(Base):
    """Equity-curve snapshot taken after every bar transition."""

    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        UniqueConstraint("session_id", "bar_index", name="uq_snap_session_bar"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("replay_sessions.session_id"), nullable=False, index=True
    )
    bar_index: Mapped[int] = mapped_column(Integer, nullable=False)

    cash: Mapped[float] = mapped_column(Float, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    total_realized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    total_unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
