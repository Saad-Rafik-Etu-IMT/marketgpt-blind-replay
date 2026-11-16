"""Risk service.

Wraps the pure metric functions with session-scoped DB reads, and
manages the session's risk-limit configuration (with default-on-read).

§5 invariant 4: metrics see only bars 0..current_bar_index. This is
guaranteed structurally — the only equity-curve source is the
portfolio_snapshots table, and snapshots are only ever written for the
session's current bar (so the table itself can never contain a future-bar
snapshot).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.risk_limits import RiskLimits as CoreRiskLimits
from src.core.risk_metrics import (
    average_loss,
    average_win,
    drawdown_series,
    expectancy,
    sharpe,
    win_rate,
)
from src.models.fill import FillORM
from src.models.order import OrderORM
from src.models.portfolio import PortfolioSnapshotORM, PositionORM
from src.models.risk import RiskLimitsORM
from src.models.session import ReplaySessionORM
from src.schemas.risk import RiskLimitsRead, RiskLimitsUpdate, RiskMetricsRead


def _equity_series(db: Session, session_id: str) -> list[float]:
    """Read the session's equity curve from portfolio_snapshots.

    Bounded structurally to bars 0..current_bar_index (§5 invariant 4):
    snapshots only ever exist for the session's current bar.
    """
    snapshots = (
        db.execute(
            select(PortfolioSnapshotORM)
            .where(PortfolioSnapshotORM.session_id == session_id)
            .order_by(PortfolioSnapshotORM.bar_index)
        )
        .scalars()
        .all()
    )
    return [s.equity for s in snapshots]


def _bar_over_bar_returns(equity_series: list[float]) -> list[float]:
    """Simple bar-over-bar equity returns, skipping any zero-equity base."""
    return [
        (equity_series[i] - equity_series[i - 1]) / equity_series[i - 1]
        for i in range(1, len(equity_series))
        if equity_series[i - 1] != 0
    ]


def compute_metrics(db: Session, session_id: str) -> RiskMetricsRead:
    """Compute current risk metrics for ``session_id``.

    Pulls the equity curve from portfolio_snapshots (which is bounded
    structurally by the session's current_bar_index — see §5 invariant 4).
    """
    equity_series = _equity_series(db, session_id)
    dd_series = drawdown_series(equity_series)
    current_dd = dd_series[-1] * 100.0 if dd_series else 0.0
    max_dd = (max(dd_series) if dd_series else 0.0) * 100.0

    # Returns: simple bar-over-bar equity change.
    returns = _bar_over_bar_returns(equity_series)

    trade_pnls = _realized_pnl_per_round_trip(db, session_id)

    session_orm = db.get(ReplaySessionORM, session_id)
    current_equity = (
        equity_series[-1]
        if equity_series
        else (session_orm.initial_cash if session_orm else 0.0)
    )

    # Risk-per-trade and exposure %s are point-in-time snapshots from
    # the latest position data — used by the cockpit's risk panel.
    positions = (
        db.execute(select(PositionORM).where(PositionORM.session_id == session_id))
        .scalars()
        .all()
    )
    from src.services.portfolio_service import _marks_at_bar

    marks = (
        _marks_at_bar(db, session_id, session_orm.current_bar_index)
        if session_orm
        else {}
    )
    total_notional = sum(
        p.quantity
        * abs(marks.get(p.contract_key or p.instrument, p.avg_entry_price))
        * p.multiplier
        for p in positions
    )
    exposure_pct = (
        (total_notional / current_equity * 100.0) if current_equity > 0 else 0.0
    )

    return RiskMetricsRead(
        max_drawdown_pct=max_dd,
        current_drawdown_pct=current_dd,
        sharpe_ratio=sharpe(returns),
        win_rate=win_rate(trade_pnls),
        avg_win=average_win(trade_pnls),
        avg_loss=average_loss(trade_pnls),
        expectancy=expectancy(trade_pnls),
        risk_per_trade_pct=0.0,  # populated by the pre-trade gate, not here
        max_position_exposure_pct=exposure_pct,
    )


# --- Limits CRUD ----------------------------------------------------------


def get_limits(db: Session, session_id: str) -> RiskLimitsRead:
    """Return the session's active limits, creating defaults on first read."""
    orm = _get_or_create_limits(db, session_id)
    return _to_read(orm)


def update_limits(
    db: Session, session_id: str, payload: RiskLimitsUpdate
) -> RiskLimitsRead:
    orm = _get_or_create_limits(db, session_id)
    if payload.max_drawdown_pct is not None:
        orm.max_drawdown_pct = payload.max_drawdown_pct
    if payload.max_risk_per_trade_pct is not None:
        orm.max_risk_per_trade_pct = payload.max_risk_per_trade_pct
    if payload.max_open_positions is not None:
        orm.max_open_positions = payload.max_open_positions
    if payload.max_exposure_pct is not None:
        orm.max_exposure_pct = payload.max_exposure_pct
    db.commit()
    return _to_read(orm)


def core_limits_for(db: Session, session_id: str) -> CoreRiskLimits:
    """Convert the session's limits to the pure ``CoreRiskLimits`` dataclass."""
    orm = _get_or_create_limits(db, session_id)
    return CoreRiskLimits(
        max_drawdown_pct=orm.max_drawdown_pct,
        max_risk_per_trade_pct=orm.max_risk_per_trade_pct,
        max_open_positions=orm.max_open_positions,
        max_exposure_pct=orm.max_exposure_pct,
    )


# --- Auto-pause check ----------------------------------------------------


def should_auto_pause(db: Session, session_id: str) -> bool:
    """True when current drawdown ≥ max_drawdown_pct."""
    metrics = compute_metrics(db, session_id)
    limits = get_limits(db, session_id)
    return metrics.current_drawdown_pct >= limits.max_drawdown_pct


# --- Internals ----------------------------------------------------------


def _get_or_create_limits(db: Session, session_id: str) -> RiskLimitsORM:
    orm = db.get(RiskLimitsORM, session_id)
    if orm is not None:
        return orm
    orm = RiskLimitsORM(session_id=session_id)
    db.add(orm)
    db.commit()
    return orm


def _to_read(orm: RiskLimitsORM) -> RiskLimitsRead:
    return RiskLimitsRead(
        max_drawdown_pct=orm.max_drawdown_pct,
        max_risk_per_trade_pct=orm.max_risk_per_trade_pct,
        max_open_positions=orm.max_open_positions,
        max_exposure_pct=orm.max_exposure_pct,
    )


def _realized_pnl_per_round_trip(db: Session, session_id: str) -> list[float]:
    """Approximate per-trade P&L by pairing fills.

    V1 simplification: we treat each closing fill as a trade and use
    the position's running realized_pnl delta. Since the broker writes
    fills in order and `apply_trade` updates ``realized_pnl`` only on
    closing trades, summing the realized_pnl deltas reconstructs the
    per-close P&L sequence.
    """
    # Order fills by order_id submission order to keep things stable.
    fills_with_orders = db.execute(
        select(FillORM, OrderORM)
        .join(OrderORM, FillORM.order_id == OrderORM.order_id)
        .where(OrderORM.session_id == session_id)
        .order_by(FillORM.fill_bar, FillORM.timestamp)
    ).all()

    # Replay the position state to extract realized deltas.
    from src.core.pnl import PositionState, apply_trade

    pnls: list[float] = []
    position_by_inst: dict[str, PositionState] = {}
    session_orm = db.get(ReplaySessionORM, session_id)
    instrument = session_orm.instrument if session_orm else "?"

    for fill, order in fills_with_orders:
        instrument = session_orm.instrument
        mult = session_orm.multiplier
        pos = position_by_inst.get(
            instrument,
            PositionState(
                instrument=instrument,
                side="flat",
                quantity=0,
                avg_entry_price=0.0,
                realized_pnl=0.0,
                multiplier=mult,
            ),
        )
        closing = (pos.side == "long" and order.side == "sell") or (
            pos.side == "short" and order.side == "buy"
        )
        before = pos.realized_pnl
        pos = apply_trade(
            pos, side=order.side, quantity=fill.fill_quantity, price=fill.fill_price
        )
        position_by_inst[instrument] = pos
        delta = pos.realized_pnl - before
        if closing:
            pnls.append(delta)  # gross closing-fill P&L, including break-even
    return pnls
