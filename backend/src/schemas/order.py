"""Pydantic schemas for orders and fills."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop"]
OrderStatus = Literal["pending", "filled", "partially_filled", "cancelled", "rejected"]
InstrumentKind = Literal["underlying"]


class CreateOrderRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    """Inputs for POST /api/orders."""

    session_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int = Field(gt=0)
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    rationale: str = Field(min_length=1, max_length=2000)

    instrument_kind: InstrumentKind = "underlying"

    @model_validator(mode="after")
    def _require_prices_for_typed_orders(self) -> Self:
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop_price is required for stop orders")
        return self

class OrderRead(BaseModel):
    """Response shape for orders."""

    order_id: str
    session_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: float | None = None
    stop_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    status: OrderStatus
    submitted_at_bar: int
    submitted_at: datetime
    rationale: str
    parent_order_id: str | None = None
    oco_pair_id: str | None = None

    instrument_kind: InstrumentKind = "underlying"


class FillRead(BaseModel):
    """Response shape for fills."""

    fill_id: str
    order_id: str
    fill_price: float
    fill_quantity: int
    fill_bar: int
    slippage: float
    fee: float = 0.0
    timestamp: datetime
