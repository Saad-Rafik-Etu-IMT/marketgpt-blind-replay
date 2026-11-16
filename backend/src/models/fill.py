"""ORM model for fills."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class FillORM(Base):
    """A single fill against an order."""

    __tablename__ = "fills"

    fill_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("orders.order_id"), nullable=False, index=True
    )

    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    fill_bar: Mapped[int] = mapped_column(Integer, nullable=False)

    # fill_price - the reference price (limit/stop) or 0.0 for market.
    slippage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
