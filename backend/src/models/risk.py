"""ORM model for per-session risk limits."""

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class RiskLimitsORM(Base):
    """Active risk thresholds — one row per session, defaults from §4.5."""

    __tablename__ = "risk_limits"

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("replay_sessions.session_id"),
        primary_key=True,
    )
    max_drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)
    max_risk_per_trade_pct: Mapped[float] = mapped_column(
        Float, nullable=False, default=2.0
    )
    max_open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    max_exposure_pct: Mapped[float] = mapped_column(
        Float, nullable=False, default=100.0
    )
