"""Portfolio schemas — read models for the routes layer."""

from typing import Literal

from pydantic import BaseModel, Field

PositionSide = Literal["long", "short", "flat"]
InstrumentKind = Literal["underlying"]


class PositionRead(BaseModel):
    instrument: str
    side: PositionSide
    quantity: int
    avg_entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    multiplier: float = 1.0

    instrument_kind: InstrumentKind = "underlying"


class PortfolioSnapshotRead(BaseModel):
    session_id: str
    bar_index: int = Field(ge=0)
    total_fees: float = 0.0
    cash: float
    equity: float
    positions: list[PositionRead]
    total_realized_pnl: float
    total_unrealized_pnl: float
    drawdown_pct: float


