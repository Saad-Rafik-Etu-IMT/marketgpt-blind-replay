"""Schemas for replay sessions and step/jump requests."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SessionStatus = Literal["active", "paused", "completed"]


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    instrument: str = Field(min_length=1, max_length=32)
    start_date: date
    end_date: date
    initial_cash: float = Field(default=100_000.0, gt=0)
    multiplier: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fee_per_unit: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    spread_bps: float = Field(default=0.0, ge=0, le=1000, allow_inf_nan=False)


class StepRequest(BaseModel):
    n: int = Field(default=1, ge=1, description="Number of bars to advance.")


class JumpRequest(BaseModel):
    target_index: int = Field(ge=0, description="Bar index to jump to.")


class ReplaySessionRead(BaseModel):
    """Public view of a replay session — what every endpoint returns."""

    session_id: str
    simulation_version: int = 2
    instrument: str
    start_date: date
    end_date: date
    current_bar_index: int = Field(ge=0)
    status: SessionStatus
    initial_cash: float
    created_at: datetime
    total_bars: int = Field(ge=0, description="Bars available in this session.")
    multiplier: float = 1.0
    fee_per_unit: float = 0.0
    spread_bps: float = 0.0
