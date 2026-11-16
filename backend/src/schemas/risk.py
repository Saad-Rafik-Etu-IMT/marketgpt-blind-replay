"""Risk schemas — read/write contracts for the risk routes."""

from pydantic import BaseModel, Field


class RiskMetricsRead(BaseModel):
    """Per-session risk metrics computed at the current bar."""

    max_drawdown_pct: float = Field(ge=0)
    current_drawdown_pct: float = Field(ge=0)
    sharpe_ratio: float | None = None
    win_rate: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    expectancy: float | None = None
    risk_per_trade_pct: float = Field(ge=0)
    max_position_exposure_pct: float = Field(ge=0)


class RiskLimitsRead(BaseModel):
    """Active risk-limit configuration for a session."""

    max_drawdown_pct: float = Field(gt=0)
    max_risk_per_trade_pct: float = Field(gt=0)
    max_open_positions: int = Field(gt=0)
    max_exposure_pct: float = Field(gt=0)


class RiskLimitsUpdate(BaseModel):
    """Partial update payload — every field optional."""

    max_drawdown_pct: float | None = Field(default=None, gt=0)
    max_risk_per_trade_pct: float | None = Field(default=None, gt=0)
    max_open_positions: int | None = Field(default=None, gt=0)
    max_exposure_pct: float | None = Field(default=None, gt=0)


