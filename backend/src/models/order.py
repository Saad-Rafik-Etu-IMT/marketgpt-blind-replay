"""ORM model for orders."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class OrderORM(Base):
    """A simulated order against a replay session."""

    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("replay_sessions.session_id"), nullable=False, index=True
    )

    side: Mapped[str] = mapped_column(String(8), nullable=False)  # buy | sell
    order_type: Mapped[str] = mapped_column(
        String(8), nullable=False
    )  # market | limit | stop
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)

    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    instrument_kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="underlying"
    )

    # Optional protective children submitted alongside the parent.
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)

    # pending | filled | partially_filled | cancelled | rejected
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # Machine-readable code set only when status == "rejected" (e.g. INSUFFICIENT_CASH).
    rejection_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    submitted_at_bar: Mapped[int] = mapped_column(Integer, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    rationale: Mapped[str] = mapped_column(Text, nullable=False)

    # OCO linkage. ``parent_order_id`` is set on child SL/TP orders; the
    # ``oco_pair_id`` is a shared marker so the two siblings can find
    # each other when one fills.
    parent_order_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("orders.order_id"), nullable=True
    )
    oco_pair_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
